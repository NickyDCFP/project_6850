"""Thin, retrying clients for OpenAlex and Semantic Scholar.

Optional keys, read from the environment or .env: OPENALEX_API_KEY (without one OpenAlex allows
about 1,000 list calls a day) and SEMANTIC_SCHOLAR_API_KEY (without one requests share a public
pool and are often throttled).
"""
import os
import time

import requests

from collection.common import REPO_ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, '.env'))
except ImportError:
    pass

OPENALEX = 'https://api.openalex.org'
S2 = 'https://api.semanticscholar.org/graph/v1'
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'project_6850-collection (academic research)'


def get_json(url, params=None, json_body=None, headers=None, tries=8, min_interval=0.0):
    """GET (or POST when json_body is given) with backoff on 429/5xx. Returns parsed JSON or None on 404."""
    for attempt in range(tries):
        if min_interval:
            time.sleep(min_interval)
        try:
            if json_body is None:
                r = SESSION.get(url, params=params, headers=headers, timeout=60)
            else:
                r = SESSION.post(url, params=params, json=json_body, headers=headers, timeout=120)
        except requests.RequestException:
            time.sleep(2 ** min(attempt, 6))
            continue
        if r.status_code == 404 or (r.status_code == 400 and 'No valid paper ids' in r.text):
            return None  # Semantic Scholar answers 400 when no id in a batch is known
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get('Retry-After') or 2 ** min(attempt + 1, 7))
            time.sleep(min(wait, 300))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f'giving up on {url} after {tries} tries')


# ---------------------------------------------------------------- OpenAlex

OPENALEX_FIELDS = ('id,doi,ids,title,publication_year,primary_location,abstract_inverted_index,'
                   'cited_by_count,counts_by_year,referenced_works,authorships')


def _openalex_params(params):
    key = os.getenv('OPENALEX_API_KEY')
    return {**params, **({'api_key': key} if key else {})}


def openalex_list(filter_, select=OPENALEX_FIELDS):
    """Every work matching an OpenAlex filter string, paging with a cursor (200 per call)."""
    works, cursor = [], '*'
    while cursor:
        page = get_json(f'{OPENALEX}/works', _openalex_params(
            {'filter': filter_, 'select': select, 'per-page': 200, 'cursor': cursor}))
        works.extend(page['results'])
        cursor = page['meta'].get('next_cursor') if page['results'] else None
    return works


def openalex_search_title(title, select=OPENALEX_FIELDS):
    page = get_json(f'{OPENALEX}/works', _openalex_params(
        {'filter': f'title.search:{title.replace(",", " ").replace("|", " ")}', 'select': select, 'per-page': 5}))
    return page['results'] if page else []


def openalex_abstract(work):
    inv = work.get('abstract_inverted_index')
    if not inv:
        return ''
    words = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return ' '.join(w for _, w in words)


# ---------------------------------------------------------------- Semantic Scholar

def _s2_headers():
    key = os.getenv('SEMANTIC_SCHOLAR_API_KEY') or os.getenv('SEMANTIC_API_KEY')
    return {'x-api-key': key} if key else None


def _s2_interval():
    return 1.05 if _s2_headers() else 3.0


def s2_batch(ids, fields):
    """Look up to 500 ids ('DOI:..', 'ACL:..', 'ARXIV:..', 'CorpusId:..', paperId) at once; None for misses."""
    out = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        out.extend(get_json(f'{S2}/paper/batch', {'fields': fields}, json_body={'ids': chunk},
                            headers=_s2_headers(), min_interval=_s2_interval()) or [None] * len(chunk))
    return out


def s2_match(title, fields):
    """Best title match, or None."""
    res = get_json(f'{S2}/paper/search/match', {'query': title, 'fields': fields}, headers=_s2_headers(),
                   min_interval=_s2_interval())
    return (res or {}).get('data', [None])[0]


def s2_references(paper_id, fields='paperId,externalIds,title,year,venue'):
    """All references of one paper (paged, 1,000 per call)."""
    refs, offset = [], 0
    while offset is not None:
        page = get_json(f'{S2}/paper/{paper_id}/references', {'fields': fields, 'limit': 1000, 'offset': offset},
                        headers=_s2_headers(), min_interval=_s2_interval())
        if not page:
            break
        refs.extend(d['citedPaper'] for d in page.get('data', []))
        offset = page.get('next')
    return refs

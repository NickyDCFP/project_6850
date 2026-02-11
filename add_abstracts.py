#!/usr/bin/env python3
"""
Lightweight abstract-filling helpers focusing on OpenAlex, arXiv, OpenReview and HTML fallbacks.

This version intentionally omits Semantic Scholar calls per user request.
"""
import json
import re
import os
import time
import requests
from requests.exceptions import RequestException
from requests.utils import requote_uri
from dotenv import load_dotenv

load_dotenv()
# prefer tqdm for progress bars; provide a tiny fallback so script still works without the package
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        class _T:
            def __init__(self, it):
                self._it = list(it)
                self.total = len(self._it)
            def __iter__(self):
                for x in self._it:
                    yield x
            def __len__(self):
                return self.total
            def write(self, msg):
                print(msg)
            def set_description(self, desc):
                pass
        return _T(iterable)

# verbosity helper available regardless of tqdm availability
def is_verbose() -> bool:
    """Return True only when ADD_ABSTRACTS_VERBOSE is explicitly set to a truthy value (1/true/yes).

    This avoids treating '0' as truthy which caused noisy logs even when the user set ADD_ABSTRACTS_VERBOSE=0.
    """
    v = os.environ.get('ADD_ABSTRACTS_VERBOSE', '')
    return str(v).lower() in ('1', 'true', 'yes')


# runtime stats across a run
stats = {
    'semantic_calls': 0,
    'openalex_calls': 0,
    'arxiv_calls': 0,
    'openreview_calls': 0,
    'html_calls': 0,
    'recovered': 0,
    'candidates': 0,
}


def request_with_retries(url, method='get', timeout=20, headers=None, params=None, max_retries=2, backoff_factor=0.6, retry_on_status=(429, 500, 502, 503, 504)):
    """Make an HTTP request with retries. Treat statuses in `retry_on_status` as retryable.

    Uses exponential backoff with jitter between retries.
    """
    verbose = is_verbose()
    attempt = 0
    while attempt <= max_retries:
        try:
            if method.lower() == 'get':
                if verbose:
                    print(f"REQUEST GET {url} params={params} attempt={attempt}")
                r = requests.get(url, timeout=timeout, headers=headers or {}, params=params)
            else:
                if verbose:
                    print(f"REQUEST {method.upper()} {url} params={params} attempt={attempt}")
                r = requests.request(method, url, timeout=timeout, headers=headers or {}, params=params)

            status = getattr(r, 'status_code', None)
            if verbose:
                print(f"REQUEST OK {url} status={status}")

            if status and status in retry_on_status:
                # exponential backoff with jitter
                wait = backoff_factor * (2 ** attempt)
                jitter = wait * 0.1 * (0.5 + os.urandom(1)[0] / 255.0)
                total_wait = wait + jitter
                if verbose:
                    print(f"REQUEST RETRYABLE STATUS {status} for {url}; sleeping {total_wait:.2f}s")
                attempt += 1
                if attempt > max_retries:
                    return r
                time.sleep(total_wait)
                continue

            return r
        except RequestException as e:
            wait = backoff_factor * (2 ** attempt)
            jitter = wait * 0.1 * (0.5 + os.urandom(1)[0] / 255.0)
            total_wait = wait + jitter
            if verbose:
                print(f"REQUEST ERROR {url} attempt={attempt} error={e} retrying_in={total_wait}s")
            attempt += 1
            if attempt > max_retries:
                return None
            time.sleep(total_wait)
    return None


def get_arxiv_abstract(ee: str) -> str:
    if not ee or 'arxiv.org' not in ee:
        return ''
    try:
        m = re.search(r'arxiv\.org/(?:abs|pdf)/([^/?#]+)', ee)
        if not m:
            return ''
        arx_id = m.group(1).replace('.pdf', '')
        url = f'http://export.arxiv.org/api/query?search_query=id:{arx_id}'
        r = request_with_retries(url, timeout=15)
        if not r or not getattr(r, 'ok', False):
            return ''
        m2 = re.search(r'<summary>([\s\S]*?)</summary>', r.text)
        if m2:
            summary = re.sub(r'\s+', ' ', m2.group(1)).strip()
            return summary
    except Exception:
        return ''
    return ''


def get_semanticscholar_by_doi(doi: str) -> str:
    """Use Semantic Scholar API to fetch paper by DOI. Requires SEMANTIC_API_KEY in env."""
    key = os.environ.get('SEMANTIC_API_KEY')
    if not key or not doi:
        return ''
    try:
        # Simple per-process throttle: Semantic Scholar allows ~1 request/sec per key.
        global _semantic_last_request
        try:
            _semantic_last_request
        except NameError:
            _semantic_last_request = 0.0
        now = time.time()
        elapsed = now - _semantic_last_request
        if elapsed < 1.05:
            time.sleep(1.05 - elapsed)
        _semantic_last_request = time.time()
        url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{requote_uri(doi)}'
        params = {'fields': 'abstract'}
        headers = {'x-api-key': key, 'User-Agent': 'ethics_v_safety/1.0'}
        r = request_with_retries(url, timeout=15, headers=headers, params=params, max_retries=3)
        if not r or not getattr(r, 'ok', False):
            return ''
        data = r.json()
        abstract = data.get('abstract') or ''
        if isinstance(abstract, str):
            return re.sub(r'\s+', ' ', abstract).strip()
    except Exception:
        return ''
    return ''


def search_semanticscholar_by_title(title: str) -> str:
    """Use Semantic Scholar title search (graph API) as fallback. Requires API key."""
    key = os.environ.get('SEMANTIC_API_KEY')
    if not key or not title:
        return ''
    try:
        # Throttle to respect ~1 req/sec rate for Semantic Scholar API keys
        global _semantic_last_request
        try:
            _semantic_last_request
        except NameError:
            _semantic_last_request = 0.0
        now = time.time()
        elapsed = now - _semantic_last_request
        if elapsed < 1.05:
            time.sleep(1.05 - elapsed)
        _semantic_last_request = time.time()
        url = 'https://api.semanticscholar.org/graph/v1/paper/search'
        params = {'query': title, 'limit': 1, 'fields': 'abstract,title'}
        headers = {'x-api-key': key, 'User-Agent': 'ethics_v_safety/1.0'}

        verbose = is_verbose()
        if verbose:
            print(f"--- SEMANTIC SCHOLAR DIAGNOSTIC ---")
            print(f"API Key Used: {key[:4]}...{key[-4:]}")
            print(f"Request URL: {url}")
            print(f"Request Params: {params}")

        r = request_with_retries(url, timeout=15, headers=headers, params=params, max_retries=3)

        if verbose:
            print(f"Response Status Code: {r.status_code if r else 'N/A'}")
            print(f"Raw Response Text: {r.text if r else 'No response'}")

        if not r or not getattr(r, 'ok', False):
            return ''
        data = r.json()
        items = data.get('data') or data.get('results') or []
        if not items:
            return ''
        it = items[0]
        abstract = it.get('abstract') or ''
        if isinstance(abstract, str):
            return re.sub(r'\s+', ' ', abstract).strip()
    except Exception:
        return ''
    return ''


def get_openreview_abstract(url: str) -> str:
    if not url or 'openreview.net' not in url:
        return ''
    try:
        m = re.search(r'[?&]id=([^&#]+)', url)
        note_id = m.group(1) if m else ''
        if not note_id:
            return ''
        api = f'https://api.openreview.net/notes?id={note_id}'
        r = request_with_retries(api, timeout=15)
        if not r or not getattr(r, 'ok', False):
            return ''
        data = r.json()
        notes = data.get('notes') or ([data.get('note')] if data.get('note') else [])
        for n in notes:
            content = n.get('content', {})
            abstract = content.get('abstract') or content.get('tl;dr') or ''
            if isinstance(abstract, dict):
                abstract = abstract.get('value', '')
            if isinstance(abstract, str) and abstract.strip():
                return re.sub(r'\s+', ' ', abstract).strip()
    except Exception:
        return ''
    return ''


def try_html_abstract(url: str) -> str:
    if not url:
        return ''
    try:
        r = request_with_retries(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if not r or not getattr(r, 'ok', False):
            return ''
        html = r.text
        m = re.search(r'<meta[^>]+name=["\']citation_abstract["\'][^>]*content=["\']([^"\']+)["\']', html, flags=re.I)
        if m:
            return re.sub(r'\s+', ' ', m.group(1)).strip()
        m2 = re.search(r'Abstract\s*[:\n\r\s>]*<p[^>]*>([^<]{40,1000})', html, flags=re.I)
        if m2:
            return re.sub(r'<[^>]+>', ' ', m2.group(1)).strip()
        m3 = re.findall(r'<p[^>]*>([^<]{80,2000})</p>', html, flags=re.I)
        if m3:
            return re.sub(r'<[^>]+>', ' ', m3[0]).strip()
    except Exception:
        return ''
    return ''


def get_openalex_by_doi(doi: str, max_retries: int = 2) -> str:
    if not doi:
        return ''
    try:
        q = requote_uri(doi)
        url = f'https://api.openalex.org/works/doi:{q}'
        r = request_with_retries(url, timeout=15, headers={'User-Agent': 'ethics_v_safety/1.0'}, max_retries=max_retries, retry_on_status=(429,500,502,503,504))
        if not r or not getattr(r, 'ok', False):
            return ''
        data = r.json()
        abstract = ''
        if data.get('abstract_inverted_index'):
            inv = data.get('abstract_inverted_index') or {}
            max_idx = max((max(idxs) for idxs in inv.values()), default=-1)
            tokens = [''] * (max_idx + 1)
            for tok, idxs in inv.items():
                for i in idxs:
                    if 0 <= i < len(tokens):
                        tokens[i] = tok
            abstract = ' '.join(t for t in tokens if t)
        else:
            abstract = data.get('abstract') or ''
        if isinstance(abstract, str):
            return re.sub(r'\s+', ' ', abstract).strip()
    except Exception:
        return ''
    return ''


def search_openalex_by_title(title: str, max_retries: int = 2) -> str:
    if not title:
        return ''
    try:
        url = f'https://api.openalex.org/works?search={requote_uri(title)}&per-page=1'
        r = request_with_retries(url, timeout=15, headers={'User-Agent': 'ethics_v_safety/1.0'}, max_retries=max_retries, retry_on_status=(429,500,502,503,504))
        if not r or not getattr(r, 'ok', False):
            return ''
        data = r.json()
        items = data.get('results') or []
        if not items:
            return ''
        it = items[0]
        abstract = ''
        if it.get('abstract_inverted_index'):
            inv = it.get('abstract_inverted_index') or {}
            max_idx = max((max(idxs) for idxs in inv.values()), default=-1)
            tokens = [''] * (max_idx + 1)
            for tok, idxs in inv.items():
                for i in idxs:
                    if 0 <= i < len(tokens):
                        tokens[i] = tok
            abstract = ' '.join(t for t in tokens if t)
        else:
            abstract = it.get('abstract') or ''
        if isinstance(abstract, str):
            return re.sub(r'\s+', ' ', abstract).strip()
    except Exception:
        return ''
    return ''


def fill_abstract_for_record(rec: dict) -> dict:
    """Fill abstract using DOI (OpenAlex), title (OpenAlex), arXiv, OpenReview, then HTML."""
    try:
        if rec.get('abstract') and len(str(rec.get('abstract') or '')) >= 40:
            return rec
        doi = (rec.get('doi') or '').strip()
        ee = (rec.get('ee') or '')
        url = (rec.get('url') or '')
        title = (rec.get('title') or '')
        verbose = is_verbose()
        load_dotenv()
        ss_key = os.environ.get('SEMANTIC_API_KEY')
        if doi:
            # Prefer Semantic Scholar if an API key is present (rate 1/sec).
            if ss_key:
                if verbose:
                    print(f"Trying Semantic Scholar by DOI for {doi}")
                # record call
                try:
                    stats['semantic_calls'] += 1
                except Exception:
                    pass
                abstract = get_semanticscholar_by_doi(doi)
                if abstract and len(abstract) > 40:
                    rec['abstract'] = abstract
                    rec.setdefault('enriched_from', []).append('semanticscholar_doi')
                    try:
                        stats['recovered'] += 1
                    except Exception:
                        pass
                    return rec
            if verbose:
                print(f"Trying OpenAlex by DOI for {doi}")
            # OpenAlex can rate-limit -- allow more retries and treat 429 as retryable
            try:
                stats['openalex_calls'] += 1
            except Exception:
                pass
            abstract = get_openalex_by_doi(doi, max_retries=4)
            if abstract and len(abstract) > 40:
                rec['abstract'] = abstract
                rec.setdefault('enriched_from', []).append('openalex_doi')
                try:
                    stats['recovered'] += 1
                except Exception:
                    pass
                return rec

        if title:
            # Try Semantic Scholar title search first if key is present
            if ss_key:
                if verbose:
                    print(f"Trying Semantic Scholar title search for: {title[:100]}")
                try:
                    stats['semantic_calls'] += 1
                except Exception as e:
                    print(e)
                    pass
                abstract = search_semanticscholar_by_title(title)
                if abstract and len(abstract) > 40:
                    rec['abstract'] = abstract
                    rec.setdefault('enriched_from', []).append('semanticscholar_title')
                    try:
                        stats['recovered'] += 1
                    except Exception:
                        pass
                    return rec
            if verbose:
                print(f"Trying OpenAlex title search for: {title[:100]}")
            try:
                stats['openalex_calls'] += 1
            except Exception:
                pass
            abstract = search_openalex_by_title(title, max_retries=4)
            if abstract and len(abstract) > 40:
                rec['abstract'] = abstract
                rec.setdefault('enriched_from', []).append('openalex_title')
                try:
                    stats['recovered'] += 1
                except Exception:
                    pass
                return rec

        if ee and 'arxiv.org' in ee:
            if verbose:
                print(f"Trying arXiv for: {ee}")
            try:
                stats['arxiv_calls'] += 1
            except Exception:
                pass
            abstract = get_arxiv_abstract(ee)
            if abstract and len(abstract) > 40:
                rec['abstract'] = abstract
                rec.setdefault('enriched_from', []).append('arxiv')
                try:
                    stats['recovered'] += 1
                except Exception:
                    pass
                return rec

        if url and 'openreview.net' in url:
            if verbose:
                print(f"Trying OpenReview for: {url}")
            try:
                stats['openreview_calls'] += 1
            except Exception:
                pass
            abstract = get_openreview_abstract(url)
            if abstract and len(abstract) > 40:
                rec['abstract'] = abstract
                rec.setdefault('enriched_from', []).append('openreview')
                try:
                    stats['recovered'] += 1
                except Exception:
                    pass
                return rec

        # Last resort: try to scrape a valid URL from ee or url field
        html_url_to_try = ''
        if ee and ee.startswith('http'):
            html_url_to_try = ee
        elif url and url.startswith('http'):
            html_url_to_try = url

        if html_url_to_try:
            if verbose:
                print(f"Trying HTML fallback for: {html_url_to_try}")
            try:
                stats['html_calls'] += 1
            except Exception:
                pass
            abstract = try_html_abstract(html_url_to_try)
            if abstract and len(abstract) > 40:
                rec['abstract'] = abstract
                rec.setdefault('enriched_from', []).append('html')
                try:
                    stats['recovered'] += 1
                except Exception:
                    pass
                return rec

        return rec
    except Exception:
        return rec


def probe_conf_year(backup_path: str, conf: str, year: int, limit: int = 0):
    if not os.path.exists(backup_path):
        print(f"Backup file not found: {backup_path}")
        return
    token = conf.lower()
    found = []
    with open(backup_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            v = str(obj.get('venue') or obj.get('booktitle') or obj.get('venue_raw') or '')
            y = int(obj.get('year') or 0) if str(obj.get('year') or '').isdigit() else None
            key = str(obj.get('key') or obj.get('id') or obj.get('paper_id') or '')
            url = str(obj.get('url') or obj.get('ee') or '')
            if y != year:
                continue
            if token in v.lower() or token in key.lower() or token in url.lower():
                found.append(obj)
    if not found:
        print(f"No entries found for {conf} {year} in {backup_path}")
        return
    # Filter out records that already have sufficiently long abstracts
    to_process = [r for r in found if not (r.get('abstract') and len(str(r.get('abstract') or '')) >= 40)]
    stats['candidates'] = len(to_process)
    pbar = tqdm(to_process, total=len(to_process), desc=f"{conf} {year}", unit='paper')
    for idx, rec in enumerate(pbar, start=1):
        if limit and idx > limit:
            break
        # fill (silent)
        rec2 = fill_abstract_for_record(rec)
        after = rec2.get('abstract') or ''
        if after and len(after) > 40:
            # write to output file if requested
            out_path = os.environ.get('ADD_ABSTRACTS_OUT')
            if out_path:
                try:
                    with open(out_path, 'a', encoding='utf-8') as ofh:
                        ofh.write(json.dumps(rec2, ensure_ascii=False) + '\n')
                except Exception:
                    pass
        # no per-record prints; progress shown by tqdm only

    # summary
    print("--- Run summary ---")
    print(f"Candidates processed: {stats.get('candidates', 0)}")
    print(f"Recovered abstracts: {stats.get('recovered', 0)}")
    print(f"Semantic Scholar calls: {stats.get('semantic_calls', 0)}")
    print(f"OpenAlex calls: {stats.get('openalex_calls', 0)}")
    print(f"arXiv calls: {stats.get('arxiv_calls', 0)}")
    print(f"OpenReview calls: {stats.get('openreview_calls', 0)}")
    print(f"HTML attempts: {stats.get('html_calls', 0)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add/fill abstracts for DBLP backup records (lightweight, OpenAlex/arXiv/OpenReview only).')
    parser.add_argument('--backup', default='archive/dblp_conferences_papers_backup.jsonl')
    parser.add_argument('--out', help='Optional JSONL file to append enriched records to')
    parser.add_argument('--conf', required=True)
    parser.add_argument('--year', type=int, help='Single year to process (ignored if --years is provided)')
    parser.add_argument('--years', help='Comma-separated years or ranges, e.g. "2020,2022-2024". When used, --backup must be a directory containing per-conference files named <conf>_<year>.jsonl')
    parser.add_argument('--limit', type=int, default=0, help='Optional: process only the first N matching records (for quick tests)')
    args = parser.parse_args()
    if args.out:
        os.environ['ADD_ABSTRACTS_OUT'] = args.out
    # helper to parse years string
    def expand_years(s: str):
        years = set()
        for part in s.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                years.update(range(int(a), int(b) + 1))
            elif part:
                years.add(int(part))
        return sorted(years)

    if args.years:
        # require backup to be a directory
        if not os.path.isdir(args.backup):
            print('--years requires --backup to be a directory containing per-conference files like <conf>_<year>.jsonl')
        else:
            years = expand_years(args.years)
            for y in years:
                path = os.path.join(args.backup, f"{args.conf.lower()}_{y}.jsonl")
                # set per-run out file if user didn't supply one
                if not args.out:
                    os.environ['ADD_ABSTRACTS_OUT'] = f"data/enriched_{args.conf.lower()}_{y}.jsonl"
                probe_conf_year(path, args.conf, y, limit=args.limit)
    else:
        if not args.year:
            print('Either --year or --years must be provided')
        else:
            probe_conf_year(args.backup, args.conf, args.year, limit=args.limit)

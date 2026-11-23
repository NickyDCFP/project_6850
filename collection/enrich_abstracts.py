"""Fill missing abstracts in data/updated/collected/, from the sources that cover each venue.

  python -m collection.enrich_abstracts [--stages pages s2-ids openalex-doi s2-title] [--venues neurips iclr]

Stages, in order (each only touches papers still missing an abstract):
  pages         official proceedings pages linked from DBLP: papers.nips.cc, proceedings.iclr.cc and
                PMLR (ICML)
  s2-ids        Semantic Scholar batch lookup by DOI (AAAI, AIES, FAccT, SaTML) and ACL Anthology id
  openalex-doi  OpenAlex by DOI, 50 per call (Semantic Scholar withholds some publishers' abstracts)
  s2-title      Semantic Scholar title match, one call per paper, for whatever is left

Every result is appended to data/updated/cache/abstracts.jsonl (id, abstract, source) so runs can
be interrupted and resumed; the collected files are then rewritten with the abstracts filled in and
`abstract_source` set.
"""
import argparse
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from glob import glob

from bs4 import BeautifulSoup

from collection.apis import SESSION, get_json, openalex_abstract, s2_batch, s2_match, OPENALEX, _openalex_params
from collection.common import CACHE_DIR, COLLECTED_DIR, append_jsonl, has_abstract, norm_title, read_jsonl, write_jsonl

CACHE = os.path.join(CACHE_DIR, 'abstracts.jsonl')
PAGE_HOSTS = ('papers.nips.cc', 'proceedings.neurips.cc', 'proceedings.iclr.cc', 'proceedings.mlr.press')
STAGES = ['pages', 's2-ids', 'openalex-doi', 's2-title']


def load_cache():
    return {r['id']: r for r in read_jsonl(CACHE)} if os.path.exists(CACHE) else {}


def page_abstract(url):
    try:
        r = SESSION.get(url, timeout=60)
    except Exception:
        return ''
    if r.status_code != 200:
        return ''
    soup = BeautifulSoup(r.text, 'html.parser')
    node = soup.select_one('p.paper-abstract') or soup.select_one('#abstract')
    if node is None:  # older NeurIPS/ICLR page layout: <h4>Abstract</h4><p>...</p>
        h = soup.find(['h4', 'h3', 'h2'], string=re.compile(r'^\s*Abstract\s*$'))
        node = h.find_next('p') if h else None
    return re.sub(r'\s+', ' ', node.get_text(' ')).strip() if node else ''


def stage_pages(todo):
    jobs = []
    for r in todo:
        url = next((e for e in r.get('ee', []) if any(h in e for h in PAGE_HOSTS)), None)
        if url:
            jobs.append((r['id'], url.replace('http://', 'https://')))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (pid, url), abstract in zip(jobs, pool.map(lambda j: page_abstract(j[1]), jobs)):
            yield pid, abstract, 'proceedings_page'


def stage_s2_ids(todo):
    keyed = [(r['id'], f"DOI:{r['doi']}" if r.get('doi') else f"ACL:{r['paper_id']}" if r.get('paper_id') else None)
             for r in todo]
    keyed = [(pid, k) for pid, k in keyed if k]
    for i in range(0, len(keyed), 500):
        chunk = keyed[i:i + 500]
        for (pid, _), hit in zip(chunk, s2_batch([k for _, k in chunk], 'abstract')):
            yield pid, (hit or {}).get('abstract') or '', 'semantic_scholar'


def stage_openalex_doi(todo):
    with_doi = [r for r in todo if r.get('doi')]
    for i in range(0, len(with_doi), 50):
        chunk = with_doi[i:i + 50]
        page = get_json(f'{OPENALEX}/works', _openalex_params({
            'filter': 'doi:' + '|'.join(r['doi'] for r in chunk), 'per-page': 50,
            'select': 'doi,abstract_inverted_index'}))
        found = {(w.get('doi') or '').lower().split('doi.org/')[-1]: openalex_abstract(w) for w in page['results']}
        for r in chunk:
            yield r['id'], found.get(r['doi'].lower(), ''), 'openalex'


def stage_s2_title(todo):
    for r in todo:
        hit = s2_match(r['title'], 'title,abstract')
        ok = hit and norm_title(hit.get('title')) == norm_title(r['title'])
        yield r['id'], (hit.get('abstract') or '') if ok else '', 'semantic_scholar_title'


RUNNERS = {'pages': stage_pages, 's2-ids': stage_s2_ids, 'openalex-doi': stage_openalex_doi, 's2-title': stage_s2_title}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', nargs='+', choices=STAGES, default=STAGES)
    ap.add_argument('--venues', nargs='+', help='venue tokens to process (default: all collected files)')
    args = ap.parse_args()

    files = sorted(glob(os.path.join(COLLECTED_DIR, '*.jsonl')))
    if args.venues:
        files = [f for f in files if os.path.basename(f).rsplit('_', 1)[0] in args.venues]
    data = {f: read_jsonl(f) for f in files}
    cache = load_cache()

    for stage in args.stages:
        tried = {pid for pid, c in cache.items() if stage in c.get('tried', [])}
        todo = [r for rows in data.values() for r in rows
                if not has_abstract(r) and not has_abstract(cache.get(r['id'], {})) and r['id'] not in tried]
        print(f'{stage}: {len(todo)} papers still missing an abstract', flush=True)
        got, buf = 0, []
        for pid, abstract, source in RUNNERS[stage](todo):
            entry = cache.setdefault(pid, {'id': pid, 'abstract': '', 'source': '', 'tried': []})
            entry['tried'] = sorted(set(entry['tried']) | {stage})
            if len(abstract.strip()) > 50 and not has_abstract(entry):
                entry['abstract'], entry['source'] = abstract.strip(), source
                got += 1
            buf.append(entry)
            if len(buf) >= 200:
                append_jsonl(CACHE, buf)
                buf = []
        append_jsonl(CACHE, buf)
        print(f'{stage}: filled {got}', flush=True)

    missing = Counter()
    for path, rows in data.items():
        for r in rows:
            c = cache.get(r['id'])
            if not has_abstract(r) and c and has_abstract(c):
                r['abstract'], r['abstract_source'] = c['abstract'], c['source']
            if not has_abstract(r):
                missing[os.path.basename(path)[:-6]] += 1
        write_jsonl(path, rows)
    for name, rows in sorted((os.path.basename(p)[:-6], rows) for p, rows in data.items()):
        print(f'{name:>13}: {len(rows):>6} papers, {missing[name]:>5} still without abstract')


if __name__ == '__main__':
    main()

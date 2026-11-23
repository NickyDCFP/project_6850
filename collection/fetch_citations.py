"""Citation data for the corpus: the shipped 2020-2025 papers plus the newly filtered ones.

  python -m collection.fetch_citations            # resolve new papers, then rebuild the outputs
  python -m collection.fetch_citations --refresh  # re-fetch counts and references for every paper

Each paper is resolved to Semantic Scholar, first by id in 200-paper batches (ACL Anthology id, DOI,
arXiv id from DBLP's links), then by exact title match (one request per paper, mostly NeurIPS, ICLR and
ICML, which have no DOI). Papers Semantic Scholar does not know are looked up on OpenAlex by title.
Results are cached in data/updated/cache/citations.jsonl, so reruns only fetch papers not yet resolved.
The cache is gitignored; if it is missing it is rebuilt from citations/papers.jsonl without any requests.

Outputs, in data/updated/citations/:
  papers.jsonl     one row per corpus paper: corpus_id, origin (shipped/new), venue, year, title,
                   external ids, citation_count, influential_citation_count, reference_count, and
                   references (each with Semantic Scholar paperId and DOI/arXiv/ACL/DBLP ids)
  edges.csv        citing_id,cited_id: references from one corpus paper to another (corpus ids)
  summary.txt      coverage per venue-year
Citation counts are a snapshot as of the fetch date (`fetched` on each row).
"""
import argparse
import csv
import datetime
import os
import re
from collections import Counter, defaultdict
from glob import glob

from collection.apis import openalex_search_title, s2_batch, s2_match
from collection.common import (CACHE_DIR, CITATIONS_DIR, COLLECTED_DIR, FILTERED_DIR, REPO_ROOT, append_jsonl,
                               norm_title, read_jsonl, write_jsonl)

CACHE = os.path.join(CACHE_DIR, 'citations.jsonl')
SHIPPED_DIR = os.path.join(REPO_ROOT, 'data', 'final_filtered')
FIELDS = ('title,year,externalIds,citationCount,influentialCitationCount,referenceCount,'
          'references.paperId,references.externalIds,references.title,references.year')
ARXIV = re.compile(r'arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})')


def load_corpus():
    """(corpus_id, origin, venue, year, record) for every shipped and new filtered paper."""
    papers = {}
    for path in sorted(glob(os.path.join(SHIPPED_DIR, '*.jsonl'))):
        stem = os.path.basename(path)[:-6].split('_')
        venue = stem[1] if stem[0] == 'filtered' else stem[0]
        for r in read_jsonl(path):
            cid = r.get('id') or r.get('paper_id')
            papers[cid] = ('shipped', venue, int(r['year']), r)
    for path in sorted(glob(os.path.join(FILTERED_DIR, '*.jsonl'))):
        venue, year = os.path.basename(path)[:-6].rsplit('_', 1)
        for r in read_jsonl(path):
            papers.setdefault(r['id'], ('new', venue, int(year), r))
    return papers


def collected_index():
    """DBLP key / ACL id -> collected record, for the DOI and links the shipped files lack."""
    return {r['id']: r for path in glob(os.path.join(COLLECTED_DIR, '*.jsonl')) for r in read_jsonl(path)}


def lookup_keys(cid, rec, extra):
    keys = []
    if rec.get('paper_id') or (extra or {}).get('paper_id'):
        keys.append(f"ACL:{rec.get('paper_id') or extra['paper_id']}")
    doi = rec.get('doi') or (extra or {}).get('doi')
    if doi:
        keys.append(f'DOI:{doi}')
    for link in (extra or {}).get('ee', []) + [rec.get('ee') or '']:
        m = ARXIV.search(link if isinstance(link, str) else '')
        if m:
            keys.append(f'ARXIV:{m.group(1)}')
    return keys


def title_ok(a, b):
    a, b = norm_title(a), norm_title(b)
    return bool(a) and (a == b or (min(len(a), len(b)) > 30 and (a in b or b in a)))


def from_s2(hit, how):
    refs = []
    for ref in hit.get('references') or []:
        if ref and ref.get('paperId'):
            ids = ref.get('externalIds') or {}
            refs.append({'paperId': ref['paperId'], 'doi': ids.get('DOI'), 'arxiv': ids.get('ArXiv'),
                         'acl': ids.get('ACL'), 'dblp': ids.get('DBLP'), 'title': ref.get('title'),
                         'year': ref.get('year')})
    ids = hit.get('externalIds') or {}
    return {'resolved_by': how, 's2_paper_id': hit['paperId'], 'doi': ids.get('DOI'), 'arxiv': ids.get('ArXiv'),
            'dblp': ids.get('DBLP'), 'acl': ids.get('ACL'), 'citation_count': hit.get('citationCount'),
            'influential_citation_count': hit.get('influentialCitationCount'),
            'reference_count': hit.get('referenceCount'), 'references': refs}


def from_openalex(work):
    return {'resolved_by': 'openalex_title', 'openalex_id': work['id'],
            'doi': (work.get('doi') or '').split('doi.org/')[-1] or None, 'citation_count': work.get('cited_by_count'),
            'counts_by_year': work.get('counts_by_year'), 'reference_count': len(work.get('referenced_works') or []),
            'references': [{'openalex_id': w} for w in work.get('referenced_works') or []]}


def tolerant(fn, *args, default=None):
    """Call an API helper; on repeated throttling give up on this item only (it stays uncached, so the
    next run retries it) instead of ending the whole run."""
    try:
        return fn(*args)
    except RuntimeError as e:
        print(f'  skipped for now: {e}', flush=True)
        return default


def resolve(todo, papers, extra, cache):
    today = datetime.date.today().isoformat()

    def save(cid, row):
        row.update(corpus_id=cid, fetched=today)
        cache[cid] = row
        append_jsonl(CACHE, [row])

    # 1. batch by id, trying each paper's ids in turn (Semantic Scholar knows many ACL papers only by DOI)
    keys = {cid: lookup_keys(cid, papers[cid][3], extra.get(cid)) for cid in todo}
    print(f'by id: {sum(1 for k in keys.values() if k)} of {len(todo)} papers have an ACL id, DOI or arXiv id',
          flush=True)
    for round_ in range(3):
        keyed = [(cid, ks[round_]) for cid, ks in keys.items() if len(ks) > round_ and cid not in cache]
        if not keyed:
            break
        found = 0
        for i in range(0, len(keyed), 200):
            chunk = keyed[i:i + 200]
            hits = tolerant(s2_batch, [k for _, k in chunk], FIELDS, default=[None] * len(chunk))
            for (cid, key), hit in zip(chunk, hits):
                if hit and hit.get('paperId'):
                    save(cid, from_s2(hit, 's2_' + key.split(':')[0].lower()))
                    found += 1
        print(f'  id round {round_ + 1}: {found} of {len(keyed)} resolved', flush=True)

    # 2. exact title match, then a batch fetch of the matched ids for references
    rest = [cid for cid in todo if cid not in cache]
    print(f'by title: {len(rest)} papers', flush=True)
    matched, throttled = [], set()  # throttled: no answer from Semantic Scholar yet; retried next run
    for n, cid in enumerate(rest, 1):
        title = papers[cid][3]['title']
        hit = tolerant(s2_match, title, 'title,year', default=False)
        if hit is False:
            throttled.add(cid)
        elif hit and title_ok(hit.get('title'), title):
            matched.append((cid, hit['paperId']))
        if n % 200 == 0:
            print(f'  {n}/{len(rest)} titles, {len(matched)} matched', flush=True)
        if len(matched) >= 200 or (n == len(rest) and matched):
            fulls = tolerant(s2_batch, [pid for _, pid in matched], FIELDS)
            for (mcid, _), full in zip(matched, fulls or [None] * len(matched)):
                if full:
                    save(mcid, from_s2(full, 's2_title'))
                elif fulls is None:
                    throttled.add(mcid)
            matched = []

    # 3. OpenAlex title search for what Semantic Scholar does not have
    rest = [cid for cid in todo if cid not in cache and cid not in throttled]
    print(f'OpenAlex: {len(rest)} papers ({len(throttled)} left for the next run: Semantic Scholar throttled)', flush=True)
    for cid in rest:
        title = papers[cid][3]['title']
        works = tolerant(openalex_search_title, title)
        if works is None:
            continue
        hit = next((w for w in works if title_ok(w.get('title'), title)), None)
        save(cid, from_openalex(hit) if hit else {'resolved_by': None})


PAPERS_OUT = os.path.join(CITATIONS_DIR, 'papers.jsonl')


def seed_cache_from_output():
    """Recreate the (gitignored) cache from the committed papers.jsonl, which holds every cached field."""
    rows = [{k: v for k, v in r.items() if k not in ('origin', 'venue', 'year', 'title')} for r in read_jsonl(PAPERS_OUT)]
    write_jsonl(CACHE, rows)
    print(f'cache missing: rebuilt {CACHE} from {PAPERS_OUT} ({len(rows)} papers, no requests)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--refresh', action='store_true', help='ignore the cache and re-fetch every paper')
    args = ap.parse_args()

    papers = load_corpus()
    extra = collected_index()
    if not args.refresh and not os.path.exists(CACHE) and os.path.exists(PAPERS_OUT):
        seed_cache_from_output()
    cache = {} if args.refresh or not os.path.exists(CACHE) else {r['corpus_id']: r for r in read_jsonl(CACHE)}
    todo = [cid for cid in papers if cid not in cache]
    print(f'{len(papers)} corpus papers, {len(todo)} to resolve', flush=True)
    if todo:
        resolve(todo, papers, extra, cache)

    rows, by_s2 = [], {}
    for cid, (origin, venue, year, rec) in papers.items():
        c = cache.get(cid, {})
        rows.append({'corpus_id': cid, 'origin': origin, 'venue': venue, 'year': year, 'title': rec['title'],
                     **{k: v for k, v in c.items() if k != 'corpus_id'}})
        if c.get('s2_paper_id'):
            by_s2[c['s2_paper_id']] = cid
    write_jsonl(PAPERS_OUT, rows)

    edges = sorted({(r['corpus_id'], by_s2[ref['paperId']]) for r in rows for ref in r.get('references') or []
                    if ref.get('paperId') in by_s2 and by_s2[ref['paperId']] != r['corpus_id']})
    with open(os.path.join(CITATIONS_DIR, 'edges.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['citing_id', 'cited_id'])
        w.writerows(edges)

    cov = defaultdict(Counter)
    for r in rows:
        k = (r['venue'], r['year'])
        cov[k]['papers'] += 1
        cov[k]['resolved'] += bool(r.get('resolved_by'))
        cov[k]['with_refs'] += bool(r.get('references'))
    lines = [f'{"venue":>8} {"year":>4} {"papers":>6} {"resolved":>8} {"with refs":>9}']
    lines += [f'{v:>8} {y:>4} {c["papers"]:>6} {c["resolved"]:>8} {c["with_refs"]:>9}' for (v, y), c in sorted(cov.items())]
    tot = Counter()
    for c in cov.values():
        tot.update(c)
    lines.append(f'total: {tot["papers"]} papers, {tot["resolved"]} resolved, {tot["with_refs"]} with references; '
                 f'{len(edges)} citation edges within the corpus')
    with open(os.path.join(CITATIONS_DIR, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

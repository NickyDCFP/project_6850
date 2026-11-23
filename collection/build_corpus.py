"""Merge the shipped 2020-2025 corpus and the newly filtered papers into one file with uniform fields.

  python -m collection.build_corpus

Reads data/final_filtered/ (shipped, never modified), data/updated/filtered/ (new) and
data/updated/citations/papers.jsonl, and writes data/updated/corpus.jsonl, one row per paper, sorted by
venue, year and id. Every row has the same keys:

  corpus_id, origin ('shipped' | 'new'), venue (file token: aaai, acl, aies, fat, iclr, icml, neurips,
  satml), venue_name (the record's own venue string), year (int), type, title,
  authors (as in the source), authors_first_last ("First Last" for every venue; ACL gives "Last, First"),
  abstract, doi, url, ee,
  safety_score, ethics_score, keyword_category, keyword_matches   (raw, not the loader's relabelling),
  filter_result, filter_category, filter_confidence, filter_reasoning, filter_model,
  acl_category (shipped ACL records only: their stored `category` field),
  is_proceedings_record (true for the 14 shipped DBLP `proceedings` rows: whole volumes, no authors),
  s2_paper_id, citation_count, influential_citation_count, reference_count, has_references,
  s2_shared_with (other corpus_ids resolved to the same Semantic Scholar paper, e.g. workshop/Findings copies)

Reference lists stay in data/updated/citations/papers.jsonl and citation edges in edges.csv.
Labels (safety/ethics/both) are not assigned here; analysis code should apply the same rule as
analyze_aggregated.load_combined_data, including the zero-keyword AIES/FAccT/SaTML venue default.
"""
import argparse
import os
from collections import Counter, defaultdict
from glob import glob

from collection.common import CITATIONS_DIR, FILTERED_DIR, OUT_DIR, REPO_ROOT, read_jsonl, write_jsonl
from collection.collect_acl import first_last

SHIPPED_DIR = os.path.join(REPO_ROOT, 'data', 'final_filtered')
OUT = os.path.join(OUT_DIR, 'corpus.jsonl')
CITATION_FIELDS = ('s2_paper_id', 'citation_count', 'influential_citation_count', 'reference_count')


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def row(cid, origin, venue, rec, cit):
    year = int(rec['year'])
    authors = as_list(rec.get('authors'))
    is_acl = venue == 'acl'
    first_last_names = rec.get('authors_first_last') or ([first_last(a) for a in authors] if is_acl else authors)
    if origin == 'shipped' and is_acl:
        filter_result = 'pass'          # shipped ACL file holds only accepted papers; it has no filter_result
    else:
        filter_result = rec.get('filter_result')
    return {
        'corpus_id': cid, 'origin': origin, 'venue': venue, 'venue_name': rec.get('venue'), 'year': year,
        'type': rec.get('type') or ('inproceedings' if is_acl else None), 'title': rec.get('title'),
        'authors': authors, 'authors_first_last': first_last_names,
        'abstract': rec.get('abstract'), 'doi': rec.get('doi') or cit.get('doi'), 'url': rec.get('url'),
        'ee': as_list(rec.get('ee')),
        'safety_score': rec.get('safety_score'), 'ethics_score': rec.get('ethics_score'),
        'keyword_category': rec.get('keyword_category'), 'keyword_matches': rec.get('keyword_matches'),
        'filter_result': filter_result, 'filter_category': rec.get('filter_category'),
        'filter_confidence': rec.get('filter_confidence'), 'filter_reasoning': rec.get('filter_reasoning'),
        'filter_model': rec.get('filter_model') or ('gemini-2.5-flash' if origin == 'shipped' else None),
        'acl_category': rec.get('category') if origin == 'shipped' and is_acl else None,
        'is_proceedings_record': rec.get('type') == 'proceedings',
        **{k: cit.get(k) for k in CITATION_FIELDS},
        'has_references': bool(cit.get('references')),
        's2_shared_with': [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    cits = {r['corpus_id']: r for r in read_jsonl(os.path.join(CITATIONS_DIR, 'papers.jsonl'))}
    rows = {}
    for path in sorted(glob(os.path.join(SHIPPED_DIR, '*.jsonl'))):
        stem = os.path.basename(path)[:-6].split('_')
        venue = stem[1] if stem[0] == 'filtered' else stem[0]
        for rec in read_jsonl(path):
            cid = rec.get('id') or rec.get('paper_id')
            assert cid not in rows, f'duplicate shipped id {cid}'
            rows[cid] = row(cid, 'shipped', venue, rec, cits.get(cid, {}))
    n_shipped = len(rows)
    for path in sorted(glob(os.path.join(FILTERED_DIR, '*.jsonl'))):
        venue, year = os.path.basename(path)[:-6].rsplit('_', 1)
        for rec in read_jsonl(path):
            assert rec['id'] not in rows, f"new paper {rec['id']} already in the corpus"
            r = row(rec['id'], 'new', venue, rec, cits.get(rec['id'], {}))
            assert r['year'] == int(year), (rec['id'], rec['year'], year)
            rows[rec['id']] = r

    missing = set(rows) - set(cits)
    extra = set(cits) - set(rows)
    assert not missing and not extra, f'citation rows out of sync: {len(missing)} missing, {len(extra)} extra'

    by_s2 = defaultdict(list)
    for cid, r in rows.items():
        if r['s2_paper_id']:
            by_s2[r['s2_paper_id']].append(cid)
    for cids in by_s2.values():
        for cid in cids:
            rows[cid]['s2_shared_with'] = sorted(c for c in cids if c != cid)

    out = sorted(rows.values(), key=lambda r: (r['venue'], r['year'], r['corpus_id']))
    assert len({tuple(r) for r in out}) == 1, 'rows do not share one key set'
    write_jsonl(args.out, out)

    print(f'{len(out)} papers ({n_shipped} shipped, {len(out) - n_shipped} new) -> {args.out}')
    print(f"  pass filter: {sum(r['filter_result'] == 'pass' for r in out)}; "
          f"proceedings records: {sum(r['is_proceedings_record'] for r in out)}; "
          f"no authors: {sum(not r['authors'] for r in out)}; "
          f"S2-resolved: {sum(bool(r['s2_paper_id']) for r in out)}; "
          f"with references: {sum(r['has_references'] for r in out)}; "
          f"sharing an S2 id: {sum(bool(r['s2_shared_with']) for r in out)}")
    counts = Counter((r['year'], r['origin']) for r in out)
    for y in sorted({y for y, _ in counts}):
        print(f"  {y}: {counts[y, 'shipped']:5d} shipped {counts[y, 'new']:5d} new")


if __name__ == '__main__':
    main()

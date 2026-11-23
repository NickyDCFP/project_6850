"""Extract ACL Anthology papers, 2020 to the latest year present, from anthology+abstracts.bib.gz.

  python -m collection.collect_acl [--dump data/raw/anthology+abstracts.bib.gz] [--last-year 2026]

Uses the shipped parser (src/acl_anthology_scraper_v2.py) and writes data/updated/collected/acl_<year>.jsonl
with its fields, every venue in the Anthology (as for the shipped 52,604-paper ACL corpus). Two fields
are added: `id` (= `paper_id`) and `authors_first_last`, since the Anthology gives names as "Last, First"
while DBLP gives "First Last"; `authors` itself is left as in the shipped data.

By default only the papers listed in data/updated/collected_manifest.tsv are kept (the corpus's pinned
paper set), so a newer dump does not add papers. --all-papers keeps everything and updates the manifest.
"""
import argparse
import gzip
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict

from collection.common import (COLLECTED_DIR, FIRST_YEAR, RAW_DIR, apply_manifest, load_manifest, report_manifest,
                               update_manifest, write_jsonl)
from acl_anthology_scraper_v2 import ACLAnthologyParserV2  # noqa: E402 (src/ is put on sys.path by common)
from keywords import keywords  # noqa: E402


def first_last(name):
    last, sep, first = name.partition(', ')
    return f'{first} {last}' if sep else name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', default=os.path.join(RAW_DIR, 'anthology+abstracts.bib.gz'))
    ap.add_argument('--last-year', type=int, default=2026)
    ap.add_argument('--all-papers', action='store_true', help='keep papers not in the manifest and update it')
    args = ap.parse_args()

    bib = args.dump[:-3] if args.dump.endswith('.gz') else args.dump
    if bib != args.dump and not os.path.exists(bib):
        with gzip.open(args.dump, 'rb') as src, open(bib, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    parser = ACLAnthologyParserV2(bib, keywords['safety'], keywords['ethics'], min_year=FIRST_YEAR,
                                  max_year=args.last_year, output_dir=RAW_DIR)
    by_year = defaultdict(list)
    for paper in parser.parse_bib_file():
        rec = asdict(paper)
        rec['id'] = rec['paper_id']
        rec['authors_first_last'] = [first_last(a) for a in rec['authors']]
        by_year[rec['year']].append(rec)
    owns = lambda stem: stem.startswith('acl_')  # noqa: E731
    manifest = None if args.all_papers else load_manifest()
    if manifest is not None:
        by_stem, dropped, missing = apply_manifest({f'acl_{y}': recs for y, recs in by_year.items()}, manifest, owns)
        by_year = {int(s.split('_')[1]): recs for s, recs in by_stem.items()}
    for year, recs in sorted(by_year.items()):
        write_jsonl(os.path.join(COLLECTED_DIR, f'acl_{year}.jsonl'), recs)
    with_abs = Counter(r['year'] for recs in by_year.values() for r in recs if r['abstract'])
    for year, recs in sorted(by_year.items()):
        print(f'acl {year}: {len(recs)} papers, {with_abs[year]} with abstracts')
    print(f'total {sum(map(len, by_year.values()))}')
    if manifest is None:
        update_manifest({f'acl_{y}': recs for y, recs in by_year.items()}, owns)
    else:
        report_manifest(dropped, missing)


if __name__ == '__main__':
    main()

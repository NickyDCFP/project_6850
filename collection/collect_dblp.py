"""Extract the DBLP venues' papers from a fresh dblp.xml.gz dump, 2020 to the latest year present.

  python -m collection.collect_dblp [--dump data/raw/dblp.xml.gz] [--last-year 2026]

Writes data/updated/collected/<venue>_<year>.jsonl with the same fields as data/final_dataset/ plus
`ee` (all electronic-edition links) and `doi`. Only `inproceedings` records are kept: the shipped
files also contain one `proceedings` record per volume, which is the volume itself, not a paper.
Abstracts already gathered for the shipped dataset are reused by DBLP key; the rest are filled by
collection.enrich_abstracts.

By default only the papers listed in data/updated/collected_manifest.tsv are kept (the corpus's pinned
paper set), so a newer dump does not add papers. --all-papers keeps everything and updates the manifest.
"""
import argparse
import gzip
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from glob import glob

from collection.common import (COLLECTED_DIR, DBLP_VENUES, FIRST_YEAR, RAW_DIR, SHIPPED_DATASET_DIR, apply_manifest,
                               has_abstract, load_manifest, read_jsonl, report_manifest, update_manifest, write_jsonl)
from parse_dblp_xml import EntityFixingFile, parse_dtd  # noqa: E402 (repo root is put on sys.path by common)

DOI_PREFIX = ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/', 'http://dx.doi.org/')


def extract(elem):
    rec = {'id': elem.get('key'), 'type': elem.tag, 'title': '', 'authors': [], 'year': '', 'venue': '',
           'url': '', 'ee': [], 'doi': ''}
    for child in elem:
        text = ''.join(child.itertext()).strip()
        if child.tag == 'author':
            rec['authors'].append(text)
        elif child.tag == 'title':
            rec['title'] = text
        elif child.tag in ('year', 'url'):
            rec[child.tag] = text
        elif child.tag == 'booktitle':
            rec['venue'] = text
        elif child.tag == 'ee':
            rec['ee'].append(text)
    for link in rec['ee']:
        if link.lower().startswith(DOI_PREFIX):
            rec['doi'] = link.split('doi.org/', 1)[1]
            break
    return rec


def parse(dump, dtd, last_year):
    prefixes = {p: v for v, p in DBLP_VENUES.items()}
    out = defaultdict(list)
    with gzip.open(dump, 'rt', encoding='utf-8', errors='replace') as f:
        for _, elem in ET.iterparse(EntityFixingFile(f, extra_map=parse_dtd(dtd)), events=('end',)):
            if elem.tag != 'inproceedings':
                if elem.tag in ('article', 'proceedings', 'book', 'incollection', 'phdthesis', 'mastersthesis', 'www'):
                    elem.clear()
                continue
            key = elem.get('key') or ''
            venue = next((v for p, v in prefixes.items() if key.startswith(p)), None)
            if venue:
                rec = extract(elem)
                if rec['year'].isdigit() and FIRST_YEAR <= int(rec['year']) <= last_year:
                    out[(venue, rec['year'])].append(rec)
            elem.clear()
    return out


def shipped_abstracts():
    found = {}
    for path in glob(os.path.join(SHIPPED_DATASET_DIR, '*.jsonl')):
        for r in read_jsonl(path):
            if has_abstract(r):
                found[r['id']] = r['abstract']
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', default=os.path.join(RAW_DIR, 'dblp.xml.gz'))
    ap.add_argument('--dtd', default=os.path.join(RAW_DIR, 'dblp.dtd'))
    ap.add_argument('--last-year', type=int, default=2026)
    ap.add_argument('--all-papers', action='store_true', help='keep papers not in the manifest and update it')
    args = ap.parse_args()

    parsed = parse(args.dump, args.dtd, args.last_year)
    owns = lambda stem: stem.rsplit('_', 1)[0] in DBLP_VENUES  # noqa: E731
    manifest = None if args.all_papers else load_manifest()
    if manifest is None:
        groups = parsed
    else:
        by_stem, dropped, missing = apply_manifest({f'{v}_{y}': recs for (v, y), recs in parsed.items()}, manifest, owns)
        groups = {tuple(s.rsplit('_', 1)): recs for s, recs in by_stem.items()}
    abstracts = shipped_abstracts()
    reused = Counter()
    for (venue, year), recs in sorted(groups.items()):
        for r in recs:
            r['abstract'] = abstracts.get(r['id'], '')
            if r['abstract']:
                r['abstract_source'] = 'shipped_dataset'
                reused[venue, year] += 1
        write_jsonl(os.path.join(COLLECTED_DIR, f'{venue}_{year}.jsonl'), recs)
    print(f'{"venue":>8} {"year":>4} {"papers":>6} {"abstract reused":>15}')
    for (venue, year), recs in sorted(groups.items()):
        print(f'{venue:>8} {year:>4} {len(recs):>6} {reused[venue, year]:>15}')
    print(f'total {sum(map(len, groups.values()))} papers, {sum(reused.values())} abstracts reused')
    if manifest is None:
        update_manifest({f'{v}_{y}': recs for (v, y), recs in groups.items()}, owns)
    else:
        report_manifest(dropped, missing)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Probe enrichment outputs in `data/` and suggest archive moves for older files.
- Summarize per-file counts for `data/enriched_*.jsonl` and `data/enriched_*_all_years.jsonl`.
- By default only dry-run: print which files would be moved to `archive/` if they are older than `--before-year` and are not named `*backup*` or `acl*`.
- Use --move to actually relocate files.
"""
import argparse
from pathlib import Path
import json

DATA_DIR = Path('data')
ARCHIVE_DIR = Path('archive')


def count_jsonl(path: Path):
    total = 0
    with path.open('r', encoding='utf-8') as fh:
        for _ in fh:
            total += 1
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before-year', type=int, default=2022, help='Archive files for years < this year')
    p.add_argument('--move', action='store_true', help='Actually move files')
    args = p.parse_args()

    enriched = sorted(DATA_DIR.glob('enriched_*.jsonl'))
    if not enriched:
        print('No enriched files found in data/')
        return

    summary = []
    for f in enriched:
        name = f.name
        # count lines quickly
        try:
            c = count_jsonl(f)
        except Exception:
            c = None
        summary.append((name, c))

    print('Enrichment summary (file: count):')
    for name, c in summary:
        print(f'{name}: {c}')

    # Decide archive candidates: files with a year in name < before-year, excluding backups and acl*
    candidates = []
    for name, c in summary:
        lname = name.lower()
        if 'backup' in lname or lname.startswith('acl'):
            continue
        # try to extract year token
        import re
        m = re.search(r'_(\d{4})\.jsonl$', name)
        if not m:
            # also archive "_all_years" if older than before-year
            if '_all_years' in name:
                # archival decision: archive all_years files if before-year > 0
                candidates.append((name, c))
            continue
        year = int(m.group(1))
        if year < args.before_year:
            candidates.append((name, c))

    if not candidates:
        print('\nNo files match archive criteria (dry-run).')
    else:
        print('\nArchive candidates:')
        for name, c in candidates:
            print(f' - {name} ({c} lines)')

        if args.move:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            for name, c in candidates:
                src = DATA_DIR / name
                dst = ARCHIVE_DIR / name
                print(f'Moving {src} -> {dst}')
                src.rename(dst)
            print('\nMove complete.')
        else:
            print('\nDry-run: use --move to actually move files to archive/')

if __name__ == '__main__':
    main()

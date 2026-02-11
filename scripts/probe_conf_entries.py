#!/usr/bin/env python3
"""
Probe `conf_entries/` JSONL files for abstracts shorter than 500 characters.
Writes a small report to stdout: per-file counts and up to 3 sampled records per conference (evenly distributed).
"""
import json
from pathlib import Path
import random

CONF_DIR = Path('conf_entries')
OUT_LIMIT_PER_CONF = 8
THRESH = 1000

def sample_evenly(items, k):
    if not items:
        return []
    n = len(items)
    if n <= k:
        return items
    step = n / k
    res = []
    for i in range(k):
        idx = int(i * step)
        res.append(items[idx])
    return res


def main():
    files = sorted(CONF_DIR.glob('*.jsonl'))
    if not files:
        print('No files in', CONF_DIR)
        return
    total_files = 0
    total_short = 0
    report = {}
    for f in files:
        total_files += 1
        short_recs = []
        total = 0
        with f.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                total += 1
                abstract = (r.get('abstract') or '')
                if abstract and len(abstract) < THRESH:
                    short_recs.append({'id': r.get('id'), 'title': r.get('title') or r.get('paper_title'), 'abstract_len': len(abstract), 'sample_abstract': abstract[:300]})
        total_short += len(short_recs)
        samples = sample_evenly(short_recs, OUT_LIMIT_PER_CONF)
        report[f.name] = {'total': total, 'short_count': len(short_recs), 'samples': samples}

    # print concise report
    print(f'Files scanned: {total_files}')
    print(f'Total short abstracts (<{THRESH} chars): {total_short}')
    print('\nPer-file summary:')
    for fn, info in sorted(report.items()):
        print(f"{fn}: total={info['total']}, short={info['short_count']}")
    print('\nSampled short abstracts (up to 3 per file):')
    for fn, info in sorted(report.items()):
        if not info['samples']:
            continue
        print('\n---', fn)
        for s in info['samples']:
            print(f"id={s['id']} len={s['abstract_len']} title={s['title']}")
            print(s['sample_abstract'])
            print('---')

if __name__ == '__main__':
    main()

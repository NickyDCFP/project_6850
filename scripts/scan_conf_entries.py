#!/usr/bin/env python3
"""
Scan `conf_entries/` and report per-file and aggregated counts of abstracts.
Reports:
 - total records
 - abstracts present
 - abstracts >=40 chars (usable)
 - abstracts <40 chars (duds)
 - abstracts exactly 195 chars (known fake)
 - missing abstracts
Also prints small samples of dud/missing IDs per file.
"""
import json
from pathlib import Path
from collections import defaultdict

CONF_DIR = Path('conf_entries')
DUD_THRESHOLD = 40
FAKE_LEN = 195


def scan():
    files = sorted(CONF_DIR.glob('*.jsonl'))
    if not files:
        print('No files in', CONF_DIR)
        return
    agg = defaultdict(lambda: {'total':0,'with_abs':0,'usable':0,'dud':0,'fake195':0,'missing':0,'samples':[]})
    total_files = 0
    for f in files:
        total_files += 1
        conf, year_ext = f.stem.split('_',1)
        year = year_ext
        key = f"{conf}_{year}"
        with f.open('r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                agg[key]['total'] += 1
                a = r.get('abstract')
                if a:
                    agg[key]['with_abs'] += 1
                    la = len(str(a))
                    if la >= DUD_THRESHOLD:
                        agg[key]['usable'] += 1
                    else:
                        agg[key]['dud'] += 1
                        if len(agg[key]['samples']) < 3:
                            agg[key]['samples'].append({'id': r.get('id'), 'len': la, 'title': r.get('title')})
                    if la == FAKE_LEN:
                        agg[key]['fake195'] += 1
                else:
                    agg[key]['missing'] += 1
                    if len(agg[key]['samples']) < 3:
                        agg[key]['samples'].append({'id': r.get('id'), 'len': 0, 'title': r.get('title')})

    # print summary
    total_recs = sum(v['total'] for v in agg.values())
    print(f'Files scanned: {len(agg)}; total records: {total_recs}')
    print('\nPer-file breakdown:')
    for k in sorted(agg.keys()):
        v = agg[k]
        print(f"{k}: total={v['total']} with_abs={v['with_abs']} usable={v['usable']} dud={v['dud']} fake195={v['fake195']} missing={v['missing']}")
    print('\nSample dud/missing per file:')
    for k in sorted(agg.keys()):
        v = agg[k]
        if v['samples']:
            print('\n---', k)
            for s in v['samples']:
                print(s)

if __name__ == '__main__':
    scan()

#!/usr/bin/env python3
"""
Deduplicate `data/enriched_*.jsonl` by canonical paper id and recompute coverage vs `conf_entries/`.
- Writes deduped files to `data/deduped_enriched_<rest>.jsonl` (does not overwrite originals).
- Prints before/after counts and corrected recovery table.
"""
import json
from pathlib import Path
import re
from collections import defaultdict, Counter

DATA = Path('data')
CONF = Path('conf_entries')
OUT_DIR = DATA

ENRICHED = sorted(DATA.glob('enriched_*.jsonl'))
CONF_FILES = {p.stem: p for p in sorted(CONF.glob('*.jsonl'))}

id_fields = ['paper_id','id','key']


def canonical_id(rec):
    for f in id_fields:
        v = rec.get(f)
        if v:
            return str(v)
    # try doi/title fallback
    doi = (rec.get('doi') or '').strip()
    if doi:
        return doi.lower()
    title = (rec.get('title') or rec.get('paper_title') or '').strip()
    if title:
        return 'title:' + re.sub(r'\s+',' ', title).lower()
    return None


def dedupe_file(path: Path, out_path: Path):
    seen = set()
    total = 0
    written = 0
    with path.open('r', encoding='utf-8') as inf, out_path.open('w', encoding='utf-8') as outf:
        for line in inf:
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cid = canonical_id(rec)
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            json.dump(rec, outf, ensure_ascii=False)
            outf.write('\n')
            written += 1
    return total, written


def count_file_lines(path: Path):
    c = 0
    with path.open('r', encoding='utf-8') as fh:
        for _ in fh:
            c += 1
    return c


def main():
    if not ENRICHED:
        print('No enriched files found in data/')
        return
    # dedupe each file
    deduped_map = {}
    print('Deduping enriched files into data/deduped_enriched_*.jsonl (won\'t overwrite originals)')
    for ef in ENRICHED:
        out = DATA / ('deduped_' + ef.name)
        total, written = dedupe_file(ef, out)
        deduped_map[ef.name] = (total, written, out)
        print(f'{ef.name}: lines={total} -> deduped={written} (wrote {out})')

    # recompute coverage against conf_entries using deduped files
    print('\nRecomputing coverage using deduped outputs...')
    # build conf totals
    conf_totals = {p.stem: count_file_lines(p) for p in CONF.glob('*.jsonl')}

    # accumulate enriched counts from deduped files that match per-year pattern
    pattern = re.compile(r'^deduped_enriched_([a-z0-9]+)_(\d{4})\.jsonl$')
    results = defaultdict(lambda: {'conf_total':0,'enriched_unique':0})
    for name, (_, written, out) in deduped_map.items():
        m = pattern.match(out.name)
        if m:
            key = f"{m.group(1)}_{m.group(2)}"
            results[key]['enriched_unique'] = count_file_lines(out)
    # attach conf totals
    for k in list(results.keys()):
        results[k]['conf_total'] = conf_totals.get(k, 0)

    # print table
    print('conf_year | conf_total | enriched_unique | recovered_pct')
    total_conf = 0
    total_enr = 0
    for k in sorted(results.keys()):
        ct = results[k]['conf_total']
        eu = results[k]['enriched_unique']
        pct = (eu/ct*100) if ct else 0.0
        print(f"{k} | {ct} | {eu} | {pct:.1f}%")
        total_conf += ct
        total_enr += eu
    print(f'\nGlobal unique recovery: total conf_entries={total_conf}, total unique enriched={total_enr}, recovery={(total_enr/total_conf*100) if total_conf else 0:.1f}%')

if __name__ == '__main__':
    main()

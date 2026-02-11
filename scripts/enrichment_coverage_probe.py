#!/usr/bin/env python3
"""
Compare `conf_entries/*.jsonl` totals against `data/enriched_*.jsonl` outputs.
Produces per conf_year:
 - total entries in conf_entries
 - enriched outputs lines
 - percent recovered
 - breakdown of enriched abstracts by length (>=40 usable, <40 dud, ==195 fake)
 - breakdown of `enriched_from` sources (if present)
Also reports enriched files that don't map to any conf_entries file (e.g., *_all_years or debug files).
"""
import json
from pathlib import Path
import re
from collections import defaultdict, Counter

CONF_DIR = Path('conf_entries')
DATA_DIR = Path('data')

CONF_FILES = sorted(CONF_DIR.glob('*.jsonl'))
ENRICHED_FILES = sorted(DATA_DIR.glob('enriched_*.jsonl'))

def count_lines(path):
    c = 0
    with path.open('r', encoding='utf-8') as fh:
        for _ in fh:
            c += 1
    return c

# Load conf_entries totals
conf_totals = {}
for f in CONF_FILES:
    name = f.stem  # e.g., aaai_2020
    conf_totals[name] = count_lines(f)

# Prepare data structures for results
results = {}
for k in conf_totals:
    results[k] = {
        'conf_total': conf_totals[k],
        'enriched_count': 0,
        'usable': 0,
        'dud': 0,
        'fake195': 0,
        'sources': Counter(),
    }

unmatched_enriched = []

pattern_year = re.compile(r'^enriched_([a-z0-9]+)_(\d{4})\.jsonl$')
pattern_named = re.compile(r'^enriched_([a-z0-9_]+)\.jsonl$')

# Process enriched files
for ef in ENRICHED_FILES:
    name = ef.name
    m = pattern_year.match(name)
    if m:
        conf = f"{m.group(1)}_{m.group(2)}"
        if conf not in results:
            # present in data but not in conf_entries
            results.setdefault(conf, {'conf_total': 0, 'enriched_count': 0, 'usable':0, 'dud':0, 'fake195':0, 'sources': Counter()})
        target = results[conf]
        # iterate records
        with ef.open('r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                target['enriched_count'] += 1
                a = r.get('abstract') or ''
                la = len(a)
                if la >= 40:
                    target['usable'] += 1
                else:
                    target['dud'] += 1
                if la == 195:
                    target['fake195'] += 1
                # sources
                if 'enriched_from' in r:
                    if isinstance(r['enriched_from'], list):
                        for s in r['enriched_from']:
                            target['sources'][s] += 1
                    else:
                        target['sources'][str(r['enriched_from'])] += 1
    else:
        # unmatched (e.g., all_years or debug files)
        unmatched_enriched.append((name, count_lines(ef)))

# Build report
lines = []
lines.append('Per conf/year coverage:')
summary = []
for k in sorted(results.keys()):
    v = results[k]
    total = v['conf_total']
    enriched = v['enriched_count']
    pct = (enriched / total * 100) if total else (100.0 if enriched else 0.0)
    summary.append((k, total, enriched, pct, v['usable'], v['dud'], v['fake195'], dict(v['sources'])))

# Print concise table-like output
print('conf_year | total_in_conf_entries | enriched_count | recovered_pct | usable | dud | fake195 | top_sources')
for row in sorted(summary):
    k, total, enriched, pct, usable, dud, fake195, sources = row
    top_sources = ','.join([f'{s}:{c}' for s,c in Counter(sources).most_common(3)]) if sources else ''
    print(f"{k} | {total} | {enriched} | {pct:.1f}% | {usable} | {dud} | {fake195} | {top_sources}")

if unmatched_enriched:
    print('\nEnriched files that did not map to a single conf_year (name: lines):')
    for name, c in unmatched_enriched:
        print(f' - {name}: {c}')

# Global totals
total_conf = sum(v['conf_total'] for v in results.values())
total_enr = sum(v['enriched_count'] for v in results.values())
print(f'\nGlobal: total conf_entries records = {total_conf}, total enriched records = {total_enr}, overall recovery = { (total_enr/total_conf*100) if total_conf else 0:.1f}%')

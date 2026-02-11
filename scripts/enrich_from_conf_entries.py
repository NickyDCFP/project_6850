#!/usr/bin/env python3
"""Enrich per-conference files in conf_entries using the monolithic backup as a cache.

For each record in conf_entries/<conf>_<year>.jsonl that lacks an abstract, first look up the
backup for a cached abstract and copy it if present. Only if no cached abstract is found,
call add_abstracts.fill_abstract_for_record() to attempt external enrichment (Semantic Scholar, OpenAlex, etc.).

Appends recovered records to data/enriched_<conf>_<year>.jsonl.
"""
import os
import sys
import json
import pathlib
from collections import defaultdict

# ensure repo root on sys.path
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import add_abstracts as aa


def build_backup_cache(backup_path, confs_set=None):
    """Return a map of key->abstract from the backup for quick lookup.
    If confs_set is provided, only cache records whose venue/key/url match the tokens in confs_set.
    """
    cache = {}
    if not os.path.exists(backup_path):
        return cache
    with open(backup_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get('paper_id') or obj.get('key') or obj.get('id') or ''
            if not k:
                continue
            # optional conf filtering
            if confs_set:
                v = str(obj.get('venue') or obj.get('booktitle') or obj.get('venue_raw') or '')
                url = str(obj.get('url') or obj.get('ee') or '')
                matched = any((t in v.lower() or t in k.lower() or t in url.lower()) for t in confs_set)
                if not matched:
                    continue
            abstract = obj.get('abstract') or ''
            a_str = str(abstract)
            # ignore very short abstracts and known 195-char fake abstracts
            if a_str and len(a_str) >= 40 and len(a_str) != 195:
                cache[str(k)] = rejoin_whitespace(a_str)
    return cache


def rejoin_whitespace(s: str) -> str:
    return ' '.join(str(s).split())


def process_conf_file(conf_file, backup_cache):
    # infer conf and year from filename
    name = os.path.basename(conf_file)
    parts = name.split('_')
    if len(parts) < 2:
        return
    conf = parts[0].lower()
    year_part = parts[1].split('.')[0]
    try:
        year = int(year_part)
    except Exception:
        return

    out_path = f"data/enriched_{conf}_{year}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    processed = 0
    recovered = 0
    with open(conf_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            processed += 1
            # already has a decent abstract?
            if obj.get('abstract') and len(str(obj.get('abstract') or '')) >= 40:
                continue
            k = obj.get('paper_id') or obj.get('key') or obj.get('id') or ''
            # try backup cache first
            if k and k in backup_cache:
                obj['abstract'] = backup_cache[k]
                obj.setdefault('enriched_from', []).append('backup_cache')
                # write out
                with open(out_path, 'a', encoding='utf-8') as ofh:
                    ofh.write(json.dumps(obj, ensure_ascii=False) + '\n')
                recovered += 1
                continue

            # else call external enrichers
            rec2 = aa.fill_abstract_for_record(obj)
            after = rec2.get('abstract') or ''
            if after and len(after) > 40:
                with open(out_path, 'a', encoding='utf-8') as ofh:
                    ofh.write(json.dumps(rec2, ensure_ascii=False) + '\n')
                recovered += 1

    print(f"Processed {conf} {year}: scanned {processed}, recovered {recovered} (wrote to {out_path})")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--conf_dir', default='conf_entries')
    p.add_argument('--backup', default='data/dblp_conferences_papers_backup.jsonl')
    p.add_argument('--confs', default='aaai,neurips,aies,icml,iclr,satml')
    args = p.parse_args()

    confs = [c.strip().lower() for c in args.confs.split(',') if c.strip()]
    confs_set = set(confs)
    # alias: some backups use 'nips' rather than 'neurips'
    if 'neurips' in confs_set:
        confs_set.add('nips')

    print('Building backup cache (this may take a minute)...')
    # build cache only for relevant confs to save memory
    cache = build_backup_cache(args.backup, confs_set=confs_set)
    print(f'Backup cache contains {len(cache)} cached abstracts for target conferences')

    # process per-conf files in conf_dir
    for conf in confs:
        for year in range(2020, 2026):
            conf_file = os.path.join(args.conf_dir, f"{conf}_{year}.jsonl")
            if os.path.exists(conf_file):
                process_conf_file(conf_file, cache)


if __name__ == '__main__':
    main()

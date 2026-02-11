#!/usr/bin/env python3
"""Regenerate per-conference files in `conf_entries/` from the monolithic backup.

Backs up existing `conf_entries/` to `conf_entries_backup_<ts>/` first. Then streams the backup and
writes records into `conf_entries/<conf>_<year>.jsonl`. Filters out suspicious abstracts (exactly 195 chars)
and trims whitespace.
"""
import os
import sys
import json
import time
import pathlib
from collections import defaultdict


def sanitize_abstract(a):
    if not a:
        return ''
    s = ' '.join(str(a).split())
    # ignore known fake/sponsor footer of exactly 195 chars
    if len(s) == 195:
        return ''
    return s


def backup_conf_entries(conf_dir='conf_entries'):
    ts = time.strftime('%Y%m%d_%H%M%S')
    dest = f'{conf_dir}_backup_{ts}'
    if not os.path.exists(conf_dir):
        return dest
    os.makedirs(dest, exist_ok=True)
    for fn in os.listdir(conf_dir):
        src = os.path.join(conf_dir, fn)
        dst = os.path.join(dest, fn)
        try:
            with open(src, 'rb') as inf, open(dst, 'wb') as outf:
                outf.write(inf.read())
        except Exception:
            continue
    return dest


def regenerate(backup_path='data/dblp_conferences_papers_backup.jsonl', conf_dir='conf_entries'):
    if not os.path.exists(backup_path):
        print('Backup not found:', backup_path)
        return
    print('Backing up existing conf_entries/ ...')
    bak = backup_conf_entries(conf_dir)
    print('Backup saved to', bak)

    # ensure conf_dir exists and is empty
    os.makedirs(conf_dir, exist_ok=True)
    # map of file handles per conf-year
    handles = {}
    counts = defaultdict(int)

    with open(backup_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            # derive conf token and year
            v = str(o.get('venue') or o.get('booktitle') or o.get('venue_raw') or '')
            key = str(o.get('paper_id') or o.get('key') or o.get('id') or '')
            url = str(o.get('url') or o.get('ee') or '')
            # pick a token using heuristics
            token = None
            for t in ['aaai','neurips','nips','aies','fat','iclr','icml','satml']:
                if t in v.lower() or t in key.lower() or t in url.lower():
                    token = t
                    break
            if not token:
                continue
            # year
            y = o.get('year')
            try:
                year = int(y)
            except Exception:
                continue
            # sanitize abstract
            a = sanitize_abstract(o.get('abstract') or '')
            if a:
                o['abstract'] = a
            else:
                if 'abstract' in o:
                    del o['abstract']
            # write to conf file
            fn = f"{token}_{year}.jsonl"
            path = os.path.join(conf_dir, fn)
            if path not in handles:
                try:
                    handles[path] = open(path, 'a', encoding='utf-8')
                except Exception:
                    continue
            try:
                handles[path].write(json.dumps(o, ensure_ascii=False) + '\n')
                counts[path] += 1
            except Exception:
                continue

    for h in handles.values():
        try:
            h.close()
        except Exception:
            pass

    print('Regenerated files:')
    for p,c in counts.items():
        print(p, c)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--backup', default='data/dblp_conferences_papers_backup.jsonl')
    p.add_argument('--conf_dir', default='conf_entries')
    args = p.parse_args()
    regenerate(args.backup, args.conf_dir)

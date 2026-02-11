#!/usr/bin/env python3
"""
Combine strict filtering and global-first deduplication into one script.

Reads JSONL files from `extracted_by_conf_year/` and writes deduplicated
outputs to `deduped_by_conf_year/`. Uses the same strict path-segment
matching logic as the parser's strict mode (checks id/url path segments
and known aliases).

Usage: python3 scripts/strict_and_dedupe.py --input-dir extracted_by_conf_year --out-dir deduped_by_conf_year
"""
import argparse
import json
from pathlib import Path
import re
from typing import Iterable


def path_tokens_from_id(pid):
    # DBLP ids like 'conf/fat/facct2025.html#EbrahimiVN25' -> tokens ['conf','fat','facct2025.html#EbrahimiVN25']
    if not pid:
        return []
    parts = pid.split('/')
    return [p.lower() for p in parts if p]


def matches_strict(target_tokens, record_id, record_url):
    # Check id and url path tokens for target tokens or aliases
    rid = (record_id or '').lower()
    rurl = (record_url or '').lower()
    tokens = path_tokens_from_id(rid) + path_tokens_from_id(rurl)
    for t in target_tokens:
        if t in tokens:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--backup', default='data/dblp_conferences_papers_backup.jsonl', help='Path to backup JSONL file or a directory (deduped_by_conf_year/) to use for enrichment')
    p.add_argument('--limit', type=int, default=0, help='Optional: limit number of records processed per input file (useful for dry-run). 0 means no limit')
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Build a list of input files; filenames are expected like 'fat_2025.jsonl'
    files = sorted(in_dir.glob('*.jsonl'))
    if not files:
        # fallback: maybe the caller wants to run against existing deduped outputs
        fallback = Path('deduped_by_conf_year')
        if fallback.exists():
            print(f'No input files in {in_dir}; falling back to {fallback} as input')
            files = sorted(fallback.glob('*.jsonl'))
            in_dir = fallback
        else:
            print('No input files in', in_dir)
            return

    # We'll keep first-seen wins dedupe across all files based on id
    seen = set()
    total_in = 0
    total_out = 0

    # Load enrichment indices directly from a backup file so this script is self-contained.

    backup_path = Path(args.backup)
    by_doi = {}
    by_ee = {}
    by_url = {}
    by_paper_id = {}
    by_title = {}

    def norm_title(t):
        if not t:
            return ''
        tt = re.sub(r"[^a-z0-9]+", ' ', t.lower())
        return re.sub(r"\s+", ' ', tt).strip()

    def iter_backup_records(path: Path) -> Iterable[dict]:
        """Yield json records from a file or all .jsonl files in a directory."""
        if path.is_file():
            with path.open('r', encoding='utf-8') as bf:
                for line in bf:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        elif path.is_dir():
            for p in sorted(path.glob('*.jsonl')):
                with p.open('r', encoding='utf-8') as bf:
                    for line in bf:
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue

    def norm_url(u: str) -> str:
        if not u:
            return ''
        u = u.strip()
        # lower + strip trailing slash
        u = u.lower().rstrip('/')
        return u

    if not backup_path.exists():
        # If the explicit backup path wasn't found, try deduped_by_conf_year as a sensible fallback
        alt = Path('deduped_by_conf_year')
        if alt.exists():
            print('Specified backup not found at', backup_path, '; using', alt, 'for enrichment instead')
            backup_path = alt

    if backup_path.exists():
        print('Indexing backup for enrichment from', backup_path)
        for rec in iter_backup_records(backup_path):
            doi = (rec.get('doi') or '').lower()
            ee = (rec.get('ee') or '')
            url = norm_url(rec.get('url') or '')
            pid = rec.get('paper_id') or rec.get('id') or ''
            title = norm_title(rec.get('title') or rec.get('paper_title') or '')
            if doi:
                by_doi[doi] = rec
            if ee:
                by_ee[ee] = rec
            if url:
                by_url[url] = rec
            if pid:
                by_paper_id[pid] = rec
            if title:
                by_title[title] = rec
        print('Indexed backup: doi', len(by_doi), 'ee', len(by_ee), 'url', len(by_url), 'paper_id', len(by_paper_id), 'titles', len(by_title))
    else:
        print('No backup file found at', backup_path, '; enrichment will be skipped.')

    # naive alias map (same as before)
    alias_map = {
        'neurips': ['nips','neurips'],
        'nips': ['nips','neurips'],
        'icml': ['icml'],
        'iclr': ['iclr'],
        'aaai': ['aaai'],
        'fat': ['fat','facct'],
        'aies': ['aies'],
        'satml': ['satml'],
    }

    # For each extracted file, stream and write deduped records that match strict tokens
    limit = int(args.limit or 0)
    for f in files:
        conf, year = f.stem.split('_',1)
        target_tokens = alias_map.get(conf, [conf])
        out_file = out_dir / f.name
        wrote = 0
        with f.open('r', encoding='utf-8') as fh, out_file.open('w', encoding='utf-8') as oh:
            for line in fh:
                total_in += 1
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rid = r.get('id')
                rurl = r.get('url')
                if not matches_strict(target_tokens, rid, rurl):
                    # Strict filter: skip if doesn't match
                    continue
                if rid in seen:
                    continue
                seen.add(rid)
                # Attempt to find enrichment from backup indices (by id, doi, ee, url, title)
                enriched_rec = None
                if rid and rid in by_paper_id:
                    enriched_rec = by_paper_id[rid]
                if not enriched_rec:
                    doi = (r.get('doi') or '').lower()
                    if doi and doi in by_doi:
                        enriched_rec = by_doi[doi]
                if not enriched_rec:
                    ee = (r.get('ee') or '')
                    if ee and ee in by_ee:
                        enriched_rec = by_ee[ee]
                if not enriched_rec:
                    urlv = norm_url(r.get('url') or '')
                    if urlv and urlv in by_url:
                        enriched_rec = by_url[urlv]
                if not enriched_rec:
                    t = norm_title(r.get('title') or '')
                    if t and t in by_title:
                        enriched_rec = by_title[t]
                if enriched_rec:
                    if enriched_rec.get('abstract') and not r.get('abstract'):
                        if len(str(enriched_rec.get('abstract'))) > 250:
                            r['abstract'] = enriched_rec.get('abstract')
                    if enriched_rec.get('doi') and not r.get('doi'):
                        r['doi'] = enriched_rec.get('doi')
                    if enriched_rec.get('ee') and not r.get('ee'):
                        r['ee'] = enriched_rec.get('ee')
                    if enriched_rec.get('url') and not r.get('url'):
                        r['url'] = enriched_rec.get('url')
                json.dump(r, oh)
                oh.write('\n')
                wrote += 1
                total_out += 1
        print(f'Wrote {wrote} records to {out_file}')

    print('Total processed', total_in, 'deduped out', total_out)


if __name__ == '__main__':
    main()

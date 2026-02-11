#!/usr/bin/env python3
"""
Stage 1: Scrape abstracts from OpenReview and ArXiv to build a local database.
Stage 2: Use the local database to enrich the original source entries.

This script implements a more robust, targeted enrichment strategy by scraping
HTML directly from paper landing pages, bypassing unreliable public APIs.
"""
import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from collections import defaultdict
import requests
from requests.exceptions import RequestException

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tqdm import tqdm

# --- Utility Functions ---

def is_robust(text):
    if not text: return False
    s = str(text).strip()
    return len(s) >= 40 and len(s) != 195

def canonical_id(rec):
    for f in ['paper_id', 'id', 'key']:
        if v := rec.get(f): return str(v)
    if doi := (rec.get('doi') or '').strip(): return doi.lower()
    if title := (rec.get('title') or '').strip(): return 'title:' + ' '.join(title.split()).lower()
    return None

def request_with_retries(url, timeout=15, max_retries=3, backoff_factor=0.5):
    """Make a GET request with simple retries and backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response
        except RequestException as e:
            if attempt == max_retries - 1:
                # print(f"Error fetching {url}: {e}")
                return None
            wait = backoff_factor * (2 ** attempt)
            time.sleep(wait)
    return None

def scrape_abstract_from_html(url):
    """Scrapes an abstract from the HTML of a given URL."""
    response = request_with_retries(url)
    if not response:
        return None
    
    html = response.text
    
    # Pattern 1: OpenGraph meta tag (common on OpenReview, etc.)
    m = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']', html, re.IGNORECASE)
    if m:
        return re.sub(r'\s+', ' ', m.group(1)).strip()

    # Pattern 2: ArXiv specific meta tag
    m = re.search(r'<meta\s+name=["\']citation_abstract["\']\s+content=["\'](.*?)["\']', html, re.IGNORECASE)
    if m:
        return re.sub(r'\s+', ' ', m.group(1)).strip()

    # Pattern 3: ArXiv abstract block
    m = re.search(r'<blockquote class="abstract mathjax">[\s\S]*?<span class="descriptor">Abstract:</span>([\s\S]*?)</blockquote>', html, re.IGNORECASE)
    if m:
        return re.sub(r'<.*?>|\s+', ' ', m.group(1)).strip()

    # Generic patterns as a last resort
    m = re.search(r'Abstract\s*[:\n\r\>]+\s*<p[^>]*>([^<]{80,2000})', html, re.I)
    if m:
        return re.sub(r'<[^>]+>', ' ', m.group(1)).strip()

    return None

# --- Stage 1: Build Local Database ---

def build_local_db(missing_records, db_path, log_file):
    """Scrape abstracts for missing records and build a local database."""
    # Ensure directories exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    if Path(db_path).exists():
        with open(db_path, 'r', encoding='utf-8') as f:
            try:
                local_db = json.load(f)
            except json.JSONDecodeError:
                local_db = {} # Start fresh if DB is corrupt
    else:
        local_db = {}

    # Filter for records with promising URLs that we haven't already scraped
    records_to_scrape = [
        rec for rec in missing_records 
        if (cid := canonical_id(rec)) and cid not in local_db and 
           isinstance(rec.get('ee'), str) and ('openreview.net' in rec['ee'] or 'arxiv.org' in rec['ee'])
    ]
    
    if not records_to_scrape:
        print("No new records with OpenReview/ArXiv links to scrape.")
        return local_db

    print(f"Found {len(records_to_scrape)} new records to scrape from OpenReview/ArXiv.")
    
    pbar = tqdm(records_to_scrape, desc="Scraping Abstracts")
    with open(log_file, 'a', encoding='utf-8') as log_f:
        for rec in pbar:
            cid = canonical_id(rec)
            url = rec.get('ee')
            
            log_f.write(f"Scraping {cid} from {url}\n")
            abstract = scrape_abstract_from_html(url)
            
            if abstract and is_robust(abstract):
                log_f.write(f"  [SUCCESS] Found abstract of length {len(abstract)}\n")
                local_db[cid] = abstract
                # Save DB incrementally
                with open(db_path, 'w', encoding='utf-8') as f:
                    json.dump(local_db, f, indent=2)
            else:
                log_f.write("  [FAILURE] No robust abstract found.\n")

    print(f"Scraping complete. Local database now contains {len(local_db)} abstracts.")
    return local_db

# --- Stage 2: Enrich Using Local Database ---

def enrich_from_local_db(conf_dir, output_dir, local_db):
    """Enrich all source records using the created local database."""
    print(f"Enriching records and writing to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    source_files = sorted(Path(conf_dir).glob('*.jsonl'))
    for source_file in tqdm(source_files, desc="Enriching from Local DB"):
        output_file = Path(output_dir) / source_file.name
        
        enriched_records = []
        with source_file.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cid = canonical_id(rec)
                    if cid in local_db:
                        rec['abstract'] = local_db[cid]
                        rec.setdefault('enriched_from', []).append('local_scrape')
                    enriched_records.append(rec)
                except json.JSONDecodeError:
                    continue
        
        with output_file.open('w', encoding='utf-8') as f:
            for rec in enriched_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Targeted enrichment via direct scraping.")
    parser.add_argument('--conf_dir', default='conf_entries')
    parser.add_argument('--repaired_dir', default='data/enriched_repaired')
    parser.add_argument('--output_dir', default='data/final_enriched')
    parser.add_argument('--db_path', default='data/local_abstract_db.json')
    parser.add_argument('--log_file', default='data/compare/scraping_log.txt')
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of records to process for testing.")
    args = parser.parse_args()

    # --- Identify Missing Records ---
    print("Identifying records that still need abstracts...")
    repaired_abstracts = set()
    if Path(args.repaired_dir).exists():
        for repaired_file in list(Path(args.repaired_dir).glob('enriched_*.jsonl')):
            with repaired_file.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if is_robust(rec.get('abstract')) and (cid := canonical_id(rec)):
                            repaired_abstracts.add(cid)
                    except json.JSONDecodeError:
                        continue
    
    missing_records = []
    source_files = sorted(Path(args.conf_dir).glob('*.jsonl'))
    for source_file in source_files:
        with source_file.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if (cid := canonical_id(rec)) and cid not in repaired_abstracts:
                        missing_records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    if args.limit:
        # For testing, find the first N records that are actually scrappable
        scrappable_records = [
            rec for rec in missing_records 
            if isinstance(rec.get('ee'), str) and ('openreview.net' in rec['ee'] or 'arxiv.org' in rec['ee'])
        ]
        missing_records = scrappable_records[:args.limit]
        print(f"Applying test limit: found and processing {len(missing_records)} scrappable records.")
    
    print(f"Found {len(missing_records)} total records to process for this run.")

    # --- Stage 1 ---
    local_db = build_local_db(missing_records, args.db_path, args.log_file)

    # --- Stage 2 ---
    enrich_from_local_db(args.conf_dir, args.output_dir, local_db)

    print("\n--- Process Complete ---")
    print(f"Final enriched files are in: {args.output_dir}")
    print(f"Local abstract database is at: {args.db_path}")
    print(f"Detailed scraping log is at: {args.log_file}")

if __name__ == '__main__':
    main()

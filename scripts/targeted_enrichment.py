#!/usr/bin/env python3
"""
Targeted, high-yield enrichment script.

This script is designed to be more intelligent and robust than previous attempts.
It focuses on the most promising enrichment strategies and provides detailed logging.

Key Features:
-   Finds all records in `conf_entries` that are still missing a robust abstract in `enriched_repaired`.
-   Prioritizes valid URLs from the `ee` field for enrichment.
-   Logs every success, failure, and API response to a detailed log file.
-   Saves enriched records incrementally to a new `data/targeted_enrichment_output` directory.
-   Produces a summary report and a file of records that could not be enriched.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv()

import add_abstracts as aa
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

# --- Main Logic ---

def get_missing_records(conf_dir, repaired_dir):
    """Yields records from conf_dir that are missing a robust abstract in repaired_dir."""
    print("Identifying records that require enrichment...")
    
    # First, build a set of all canonical IDs that already have a robust abstract.
    repaired_abstracts = set()
    repaired_path = Path(repaired_dir)
    if repaired_path.exists():
        for repaired_file in tqdm(list(repaired_path.glob('enriched_*.jsonl')), desc="Scanning repaired files"):
            with repaired_file.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if is_robust(rec.get('abstract')):
                            if cid := canonical_id(rec):
                                repaired_abstracts.add(cid)
                    except json.JSONDecodeError:
                        continue

    print(f"Found {len(repaired_abstracts)} records with existing robust abstracts.")

    # Now, iterate through source files and yield records that are not in our completed set.
    source_files = sorted(Path(conf_dir).glob('*.jsonl'))
    for source_file in tqdm(source_files, desc="Scanning source files"):
        with source_file.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if (cid := canonical_id(rec)) and cid not in repaired_abstracts:
                        yield rec
                except json.JSONDecodeError:
                    continue

def main():
    parser = argparse.ArgumentParser(description="Run a targeted, high-yield enrichment process.")
    parser.add_argument('--conf_dir', default='conf_entries')
    parser.add_argument('--repaired_dir', default='data/enriched_repaired')
    parser.add_argument('--output_dir', default='data/targeted_enrichment_output')
    parser.add_argument('--log_file', default='data/compare/targeted_enrichment_log.txt')
    parser.add_argument('--failed_file', default='data/compare/targeted_enrichment_failed.jsonl')
    parser.add_argument('--limit', type=int, default=0, help="Process a limited number of records for testing.")
    args = parser.parse_args()

    # Setup output directories and log file
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.log_file)).mkdir(parents=True, exist_ok=True)
    log_f = open(args.log_file, 'w', encoding='utf-8')
    failed_f = open(args.failed_file, 'w', encoding='utf-8')

    # Get records to process
    missing_records = list(get_missing_records(args.conf_dir, args.repaired_dir))
    if args.limit > 0:
        missing_records = missing_records[:args.limit]
        
    print(f"\nFound {len(missing_records)} records to process.")
    if not missing_records:
        print("No missing records found. Exiting.")
        return

    # --- Enrichment Loop ---
    successful_enrichments = 0
    pbar = tqdm(missing_records, desc="Enriching records")

    for rec in pbar:
        cid = canonical_id(rec)
        log_f.write(f"--- PROCESSING: {cid} | Title: {rec.get('title', 'N/A')} ---\n")
        
        enriched_rec = aa.fill_abstract_for_record(rec.copy())
        
        if is_robust(enriched_rec.get('abstract')):
            successful_enrichments += 1
            source = enriched_rec.get('enriched_from', ['unknown'])[0]
            log_f.write(f"  [SUCCESS] Found abstract via: {source}\n")
            
            # Save the successfully enriched record
            conf_name = (cid.split('/')[1] if '/' in cid else 'unknown_conf')
            year = rec.get('year', 'unknown_year')
            output_filename = Path(args.output_dir) / f"{conf_name}_{year}.jsonl"
            with output_filename.open('a', encoding='utf-8') as out_f:
                out_f.write(json.dumps(enriched_rec, ensure_ascii=False) + '\n')
        else:
            log_f.write("  [FAILURE] No abstract recovered.\n")
            failed_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            
        pbar.set_postfix({"success": successful_enrichments, "failed": len(missing_records) - successful_enrichments})
        log_f.flush()

    # --- Final Report ---
    print("\n--- Targeted Enrichment Complete ---")
    print(f"  Processed records: {len(missing_records)}")
    print(f"  Successfully enriched: {successful_enrichments}")
    print(f"  Failed to enrich: {len(missing_records) - successful_enrichments}")
    print(f"  Detailed log saved to: {args.log_file}")
    print(f"  Failed records saved to: {args.failed_file}")
    print(f"  Enriched output saved in: {args.output_dir}")

    log_f.close()
    failed_f.close()

if __name__ == '__main__':
    main()

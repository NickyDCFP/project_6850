#!/usr/bin/env python3
"""
Final Fallback Enrichment Script

This script performs a final "mop-up" pass on the dataset. It takes the
partially enriched files and, for any records still missing an abstract,
applies the general-purpose enrichment logic from the `add_abstracts` module.

This is a best-effort attempt to fill in the remaining gaps using public APIs
and generic HTML scraping.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure repo root is on path and load environment variables
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

load_dotenv(repo_root / '.env')

from tqdm import tqdm
from add_abstracts import fill_abstract_for_record, is_robust

def main():
    parser = argparse.ArgumentParser(description="Final fallback enrichment pass.")
    parser.add_argument(
        '--data_dir', 
        default='data/final_enriched',
        help="Directory containing the partially enriched files to process."
    )
    parser.add_argument(
        '--log_file',
        default='data/compare/final_fallback_log.txt',
        help="Log file for this enrichment pass."
    )
    parser.add_argument(
        '--limit_per_file',
        type=int,
        default=None,
        help="For testing, limit the number of records to process per file."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found. Run previous enrichment script first.")
        return

    # Ensure log directory exists
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    source_files = sorted(data_dir.glob('*.jsonl'))
    
    print(f"Starting final enrichment pass on {len(source_files)} files in '{data_dir}'...")

    with open(args.log_file, 'w', encoding='utf-8') as log_f:
        for file_path in tqdm(source_files, desc="Processing files"):
            records_to_process = []
            all_records = []
            
            # First, read all records and identify those needing enrichment
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        all_records.append(rec)
                        if not is_robust(rec.get('abstract')):
                            records_to_process.append(rec)
                    except json.JSONDecodeError:
                        continue
            
            if not records_to_process:
                continue

            if args.limit_per_file:
                records_to_process = records_to_process[:args.limit_per_file]

            log_f.write(f"\n--- Processing {file_path.name}: Found {len(records_to_process)} records to attempt ---\n")

            # Create a map of original records to enriched versions
            enriched_map = {}
            pbar = tqdm(records_to_process, desc=f"Enriching {file_path.name}", leave=False)
            for rec in pbar:
                enriched_rec, status, source = fill_abstract_for_record(rec)
                enriched_map[id(rec)] = enriched_rec
                log_f.write(f"Attempted '{enriched_rec.get('title', 'N/A')[:50]}...': {status} via {source}\n")

            # Write the final combined list back to the file, overwriting it
            with file_path.open('w', encoding='utf-8') as f:
                for rec in all_records:
                    # If this record was processed, use the enriched version
                    if id(rec) in enriched_map:
                        f.write(json.dumps(enriched_map[id(rec)], ensure_ascii=False) + '\n')
                    # Otherwise, write the original record back
                    else:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("\n--- Final Fallback Process Complete ---")
    print(f"All files in '{data_dir}' have been updated.")
    print(f"Detailed log is at: {args.log_file}")

if __name__ == '__main__':
    main()

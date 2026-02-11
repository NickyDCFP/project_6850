#!/usr/bin/env python3
"""
Diagnose enrichment process by comparing conf_entries with enriched_repaired.

1.  Compares record counts between source and repaired directories.
2.  Identifies records from source that are missing from the repaired output.
3.  For a given conference, runs a verbose diagnostic on a sample of missing records
    to trace why enrichment is failing.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Ensure repo root is on path to allow importing add_abstracts
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import add_abstracts as aa

def is_robust(text):
    """Check if a string is a robust abstract."""
    if not text:
        return False
    s = str(text).strip()
    return len(s) >= 40 and len(s) != 195

def canonical_id(rec):
    """Get a canonical ID for a record."""
    for f in ['paper_id', 'id', 'key']:
        if v := rec.get(f):
            return str(v)
    if doi := (rec.get('doi') or '').strip():
        return doi.lower()
    if title := (rec.get('title') or '').strip():
        return 'title:' + ' '.join(title.split()).lower()
    return None

def compare_counts(conf_dir, repaired_dir):
    """Compare record counts and robust abstract coverage."""
    print("\n--- Enrichment Coverage Report ---")
    print(f"{'Conference':<25} {'Source Records':<15} {'Repaired Records':<18} {'With Abstract':<15} {'Missing':<10}")
    print("-" * 85)

    conf_files = sorted(Path(conf_dir).glob('*.jsonl'))
    summary = {}

    for conf_path in conf_files:
        conf_name = conf_path.stem
        repaired_path = Path(repaired_dir) / f"enriched_{conf_name}.jsonl"

        source_records = []
        with conf_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    source_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        source_count = len(source_records)
        source_ids = {canonical_id(rec) for rec in source_records if canonical_id(rec)}

        repaired_count = 0
        abstract_count = 0
        repaired_ids = set()

        if repaired_path.exists():
            with repaired_path.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        repaired_count += 1
                        if cid := canonical_id(rec):
                            repaired_ids.add(cid)
                        if is_robust(rec.get('abstract')):
                            abstract_count += 1
                    except json.JSONDecodeError:
                        continue
        
        missing_count = source_count - abstract_count
        
        summary[conf_name] = {
            'source_records': source_count,
            'repaired_records': repaired_count,
            'with_abstract': abstract_count,
            'missing_abstracts': missing_count,
            'source_ids_not_in_repaired': len(source_ids - repaired_ids)
        }
        
        print(f"{conf_name:<25} {source_count:<15} {repaired_count:<18} {abstract_count:<15} {missing_count:<10}")

    return summary

def find_missing_records(conf_file, repaired_file):
    """Find records from conf_file that are missing or lack abstracts in repaired_file."""
    source_ids = {}
    with Path(conf_file).open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                if cid := canonical_id(rec):
                    source_ids[cid] = rec
            except json.JSONDecodeError:
                continue

    repaired_abstracts = {}
    if Path(repaired_file).exists():
        with Path(repaired_file).open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if cid := canonical_id(rec):
                        if is_robust(rec.get('abstract')):
                            repaired_abstracts[cid] = rec.get('abstract')
                except json.JSONDecodeError:
                    continue
    
    missing = [rec for cid, rec in source_ids.items() if cid not in repaired_abstracts]
    return missing

def diagnose_failures(conference, conf_dir, repaired_dir, sample_size=5):
    """Run a verbose diagnostic on a sample of missing records for a conference."""
    conf_file = Path(conf_dir) / f"{conference}.jsonl"
    repaired_file = Path(repaired_dir) / f"enriched_{conference}.jsonl"

    if not conf_file.exists():
        print(f"Error: Source file not found at {conf_file}")
        return

    missing_records = find_missing_records(conf_file, repaired_file)
    
    if not missing_records:
        print(f"\nNo missing records found for {conference}. Enrichment appears complete.")
        return

    print(f"\n--- Diagnosing {min(sample_size, len(missing_records))} Missing Records for {conference} ---")
    
    # Set verbose mode for add_abstracts
    os.environ['ADD_ABSTRACTS_VERBOSE'] = '1'

    for i, rec in enumerate(missing_records[:sample_size]):
        cid = canonical_id(rec)
        print(f"\n--- Sample {i+1}/{sample_size} | ID: {cid} ---")
        print(f"  Title: {rec.get('title', 'N/A')}")
        print(f"  DOI: {rec.get('doi', 'N/A')}")
        print(f"  URL: {rec.get('url', 'N/A')}")
        print(f"  EE: {rec.get('ee', 'N/A')}")
        
        print("  Running enrichment trace...")
        enriched_rec = aa.fill_abstract_for_record(rec.copy())
        
        print("  --- Trace Complete ---")
        if is_robust(enriched_rec.get('abstract')):
            print(f"  SUCCESS: Found abstract from source(s): {enriched_rec.get('enriched_from')}")
        else:
            print("  FAILURE: No abstract could be recovered.")
            # Add specific failure analysis here if needed
            if 'openreview.net' in (rec.get('ee') or ''):
                 print("  Note: OpenReview API returned 404, which may indicate the paper is not public or the ID is incorrect.")
            if 'db/conf' in (rec.get('url') or ''):
                 print("  Note: The 'url' field contains a relative path, which is causing the HTML fetch to fail. This is a common issue.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose the paper enrichment process.")
    parser.add_argument('--conf_dir', default='conf_entries', help="Directory with source conference files.")
    parser.add_argument('--repaired_dir', default='data/enriched_repaired', help="Directory with enriched files.")
    parser.add_argument('--diagnose', type=str, help="Run a detailed diagnostic for a specific conference (e.g., 'iclr_2025').")
    parser.add_argument('--sample_size', type=int, default=5, help="Number of missing records to diagnose.")
    args = parser.parse_args()

    if args.diagnose:
        diagnose_failures(args.diagnose, args.conf_dir, args.repaired_dir, args.sample_size)
    else:
        compare_counts(args.conf_dir, args.repaired_dir)

if __name__ == '__main__':
    main()

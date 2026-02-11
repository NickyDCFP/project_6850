#!/usr/bin/env python3
"""
Summarize the cache coverage for the AIES conference papers.

This script performs the following steps:
1.  Identifies the total number of AIES papers from the `data/final_dataset/` directory.
2.  Scans the two specified cache source files:
    - `data/intermediate_data/filter_logs/aies_filter_detailed.jsonl`
    - `data/filtered_ethics_safety_papers.jsonl`
3.  Builds a set of unique AIES papers (based on a title-year ID) found within these cache files.
4.  Compares the set of total AIES papers against the cached set.
5.  Reports the total number of papers, the number of cached papers, and the percentage of coverage.

Usage:
  python3 scripts/summarize_aies_cache_coverage.py
"""
import json
from pathlib import Path
from typing import Dict, Any

# --- Paths ---
repo_root = Path(__file__).resolve().parents[1]
FINAL_DATA_DIR = repo_root / 'data' / 'final_dataset'
CACHE_FILE_1 = repo_root / 'data' / 'intermediate_data' / 'filter_logs' / 'aies_filter_detailed.jsonl'
CACHE_FILE_2 = repo_root / 'data' / 'filtered_ethics_safety_papers.jsonl'

# --- Helper Function ---

def norm_text(s: str) -> str:
    """Lowercase and handle None values."""
    return (s or '').lower()

def get_canonical_id(rec: Dict[str, Any]) -> str:
    """Creates a stable ID for a paper based on title and year."""
    title = norm_text(rec.get('title', ''))
    year = rec.get('year', '')
    return f"{title}_{year}"

# --- Main Logic ---

def main():
    # 1. Get all AIES papers from the final dataset
    total_aies_papers = []
    aies_files = list(FINAL_DATA_DIR.glob('aies_*.jsonl'))
    if not aies_files:
        print("No AIES files found in data/final_dataset/")
        return

    for file_path in aies_files:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    total_aies_papers.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    total_aies_ids = {get_canonical_id(rec) for rec in total_aies_papers}
    print(f"Found a total of {len(total_aies_ids)} unique AIES papers in the final dataset.")

    # 2. Build a set of cached AIES paper IDs from the two source files
    cached_aies_ids = set()
    cache_sources = [CACHE_FILE_1, CACHE_FILE_2]

    for source_file in cache_sources:
        if not source_file.exists():
            print(f"Warning: Cache source file not found: {source_file}")
            continue
        
        count = 0
        with source_file.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # Check if the paper is from AIES
                    venue = norm_text(rec.get('venue', ''))
                    if 'aies' in venue:
                        cached_aies_ids.add(get_canonical_id(rec))
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"- Found {count} AIES papers in {source_file.name}")

    print(f"\nFound a total of {len(cached_aies_ids)} unique AIES papers in the cache sources.")

    # 3. Calculate and report coverage
    covered_papers = total_aies_ids.intersection(cached_aies_ids)
    coverage_count = len(covered_papers)
    
    if not total_aies_ids:
        coverage_percentage = 0
    else:
        coverage_percentage = (coverage_count / len(total_aies_ids)) * 100

    print("\n--- AIES Cache Coverage Summary ---")
    print(f"Total unique AIES papers to be filtered: {len(total_aies_ids)}")
    print(f"Unique AIES papers already in cache:   {coverage_count}")
    print(f"Coverage Percentage:                     {coverage_percentage:.2f}%")
    print("------------------------------------")


if __name__ == '__main__':
    main()

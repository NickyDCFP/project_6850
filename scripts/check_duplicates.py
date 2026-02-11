#!/usr/bin/env python3
"""
Checks for duplicate records within each .jsonl file in a given directory.

A record is considered a duplicate if another record in the same file
shares the same 'key' or the same case-insensitive 'title' and 'year'.
"""
import os
import json
from pathlib import Path
import argparse
from collections import defaultdict

def check_for_duplicates(directory: Path):
    """
    Iterates through .jsonl files in a directory and reports duplicates.
    """
    print(f"Checking for duplicates in: {directory}\n")
    target_files = sorted(list(directory.glob('*.jsonl')))
    
    if not target_files:
        print("No .jsonl files found to check.")
        return

    total_duplicates_found = 0

    for file_path in target_files:
        seen_keys = set()
        seen_title_years = set()
        duplicates = defaultdict(int)
        line_count = 0

        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Check for duplicate key
                key = record.get('key')
                if key:
                    if key in seen_keys:
                        duplicates[f"key: {key}"] += 1
                    else:
                        seen_keys.add(key)

                # Check for duplicate title/year
                title = record.get('title', '').strip().lower()
                year = record.get('year')
                if title and year:
                    title_year = (title, year)
                    if title_year in seen_title_years:
                        duplicates[f"title/year: '{title[:50]}...' ({year})"] += 1
                    else:
                        seen_title_years.add(title_year)
        
        if duplicates:
            print(f"--- Duplicates found in: {file_path.name} ---")
            for dup_key, count in duplicates.items():
                # count is n-1 duplicates, so add 1 for total occurrences
                print(f"  - {dup_key}: Found {count + 1} times")
            total_duplicates_found += sum(duplicates.values())
            print("-" * (len(file_path.name) + 28))
        else:
            print(f"OK: No duplicates found in {file_path.name} ({line_count} records)")

    print("\n--- Summary ---")
    if total_duplicates_found > 0:
        print(f"Found a total of {total_duplicates_found} duplicate records across all files.")
    else:
        print("No duplicates found in any of the checked files.")

def main():
    parser = argparse.ArgumentParser(description="Check for duplicate records in .jsonl files.")
    parser.add_argument(
        '--directory', 
        default='data/final_dataset',
        help="Directory containing the .jsonl files to check."
    )
    args = parser.parse_args()
    
    check_for_duplicates(Path(args.directory))

if __name__ == '__main__':
    main()

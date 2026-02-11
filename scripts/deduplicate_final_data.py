#!/usr/bin/env python3
"""
Deduplicates records within each .jsonl file in a given directory.

Duplicates are identified by having identical case-insensitive titles.
When a duplicate is found, the record with the longest abstract is kept.
The original files are overwritten with the deduplicated content.
"""
import os
import json
from pathlib import Path
import argparse
from collections import defaultdict

def deduplicate_files(directory: Path):
    """
    Iterates through .jsonl files, removes duplicates, and overwrites the files.
    """
    print(f"Deduplicating files in: {directory}\n")
    target_files = sorted(list(directory.glob('*.jsonl')))
    
    if not target_files:
        print("No .jsonl files found to process.")
        return

    total_removed_count = 0

    for file_path in target_files:
        records_by_title = defaultdict(list)
        
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    title = record.get('title', '').strip().lower()
                    if title:
                        records_by_title[title].append(record)
                except json.JSONDecodeError:
                    continue
        
        deduplicated_records = []
        removed_for_file = 0
        
        for title, records in records_by_title.items():
            if len(records) > 1:
                # Sort by abstract length, descending. Keep the first one.
                records.sort(key=lambda r: len(r.get('abstract', '') or ''), reverse=True)
                deduplicated_records.append(records[0])
                removed_for_file += len(records) - 1
            else:
                deduplicated_records.append(records[0])
        
        if removed_for_file > 0:
            print(f"Processing {file_path.name}: Found and removed {removed_for_file} duplicate(s).")
            total_removed_count += removed_for_file
            
            # Overwrite the original file with the deduplicated content
            with file_path.open('w', encoding='utf-8') as f:
                for record in deduplicated_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        else:
            print(f"OK: No duplicates found in {file_path.name}.")

    print("\n--- Deduplication Summary ---")
    if total_removed_count > 0:
        print(f"Removed a total of {total_removed_count} duplicate records across all files.")
    else:
        print("No duplicates needed to be removed.")

def main():
    parser = argparse.ArgumentParser(description="Deduplicate records in .jsonl files based on title.")
    parser.add_argument(
        '--directory', 
        default='data/final_dataset',
        help="Directory containing the .jsonl files to deduplicate."
    )
    args = parser.parse_args()
    
    deduplicate_files(Path(args.directory))

if __name__ == '__main__':
    main()

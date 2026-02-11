#!/usr/bin/env python3
"""
Consolidates all final, enriched .jsonl files from the processing directory
into a single, clean output directory.
"""
import os
import shutil
from pathlib import Path

def main():
    source_dir = Path('data/combined_final')
    dest_dir = Path('data/final_dataset')

    if not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    dest_dir.mkdir(exist_ok=True)

    files_to_copy = list(source_dir.glob('*.jsonl'))
    
    if not files_to_copy:
        print(f"No .jsonl files found in '{source_dir}'.")
        return

    print(f"Consolidating {len(files_to_copy)} files from '{source_dir}' to '{dest_dir}'...")

    for file_path in files_to_copy:
        try:
            shutil.copy(file_path, dest_dir)
        except Exception as e:
            print(f"Error copying {file_path.name}: {e}")

    print("\nConsolidation complete.")
    print(f"All final .jsonl files are now in '{dest_dir}'.")

if __name__ == '__main__':
    main()

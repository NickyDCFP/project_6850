#!/usr/bin/env python3
"""
Diagnose key consistency across different cache and data files.

This script inspects the JSON keys from various data sources to ensure they
are compatible for building a unified filter cache. It checks:
1. `data/intermediate_data/filter_logs/*_filter_detailed.jsonl`
2. `data/filtered_ethics_safety_papers.jsonl`
3. `data/filtered_deduped_*.jsonl`
4. `data/final_dataset/*.jsonl`

It reports the set of keys found in each file type and highlights any
discrepancies, particularly for fields required for caching and filtering
(e.g., title, year, filter_result, filter_confidence, etc.).

Usage:
  python3 scripts/diagnose_cache_keys.py
"""
import json
from pathlib import Path
from collections import defaultdict

# --- Paths ---
repo_root = Path(__file__).resolve().parents[1]
DATA_DIR = repo_root / 'data'
CACHE_LOG_DIR = DATA_DIR / 'intermediate_data' / 'filter_logs'
PRE_FILTERED_FILE = DATA_DIR / 'filtered_ethics_safety_papers.jsonl'
FILTERED_DEDUPED_DIR = DATA_DIR
FINAL_DATASET_DIR = DATA_DIR / 'final_dataset'

# --- Main Logic ---

def inspect_file(path: Path, key_summary: defaultdict):
    """Reads the first line of a file and records its keys."""
    try:
        with path.open('r', encoding='utf-8') as f:
            line = f.readline()
            if not line:
                return
            rec = json.loads(line)
            key_summary[path.name].update(rec.keys())
    except (json.JSONDecodeError, IOError) as e:
        print(f"Could not process {path.name}: {e}")

def main():
    file_groups = {
        "detailed_logs": list(CACHE_LOG_DIR.glob('*_filter_detailed.jsonl')),
        "pre_filtered": [PRE_FILTERED_FILE] if PRE_FILTERED_FILE.exists() else [],
        "filtered_deduped": list(FILTERED_DEDUPED_DIR.glob('filtered_deduped_*.jsonl')),
    }

    all_keys = defaultdict(set)
    
    print("--- Inspecting File Keys ---\n")

    for group_name, files in file_groups.items():
        if not files:
            print(f"No files found for group: {group_name}\n")
            continue
            
        group_keys = set()
        # Inspect the first file of each group to get a representative key set
        if files:
            try:
                with files[0].open('r', encoding='utf-8') as f:
                    line = f.readline()
                    if line:
                        rec = json.loads(line)
                        group_keys.update(rec.keys())
            except Exception as e:
                print(f"Error reading {files[0]}: {e}")

        print(f"Group: {group_name} ({len(files)} files)")
        print(f"  Example file: {files[0].name}")
        print(f"  Keys: {sorted(list(group_keys))}")
        print("-" * 20)
        
        all_keys[group_name] = group_keys

    # --- Analysis ---
    print("\n--- Key Reconciliation Analysis ---\n")
    
    id_keys = {'title', 'year'}
    result_keys = {'filter_result', 'filter_confidence', 'filter_reasoning', 'category', 'filter_category'}
    
    for group_name, keys in all_keys.items():
        if not keys:
            continue
        
        missing_id_keys = id_keys - keys
        missing_result_keys = result_keys - keys
        
        print(f"Analysis for '{group_name}':")
        if not missing_id_keys:
            print("  ✅ Contains required ID keys ('title', 'year').")
        else:
            print(f"  ❌ Missing required ID keys: {missing_id_keys}")
            
        if not missing_result_keys:
            print("  ✅ Contains all expected result keys.")
        else:
            # Check for partial matches
            if result_keys.intersection(keys):
                print(f"  ⚠️  Partially missing result keys. Missing: {missing_result_keys}.")
                print(f"     Available result-related keys: {result_keys.intersection(keys)}")
            else:
                print(f"  ❌ Missing all result keys.")
        print("")

if __name__ == '__main__':
    main()

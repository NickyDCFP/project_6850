#!/usr/bin/env python3
"""
Combines all enrichment sources into a final, consolidated dataset.

This script correctly merges abstracts from all previous stages:
1. Starts with the original records from `conf_entries`.
2. Merges abstracts from the initial API pass (`data/enriched_repaired`).
3. Merges abstracts from the targeted scraping pass (`data/local_abstract_db.json`).
4. Writes the final, consolidated files to `data/combined_final/`.
"""
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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

def load_abstracts_from_file(file_path):
    """Loads all abstracts from a jsonl file into a dictionary."""
    abstracts = {}
    if not file_path.exists():
        return abstracts
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                if is_robust(rec.get('abstract')) and (cid := canonical_id(rec)):
                    abstracts[cid] = rec['abstract']
            except json.JSONDecodeError:
                continue
    return abstracts

def main():
    conf_dir = repo_root / 'conf_entries'
    repaired_dir = repo_root / 'data/enriched_repaired'
    db_path = repo_root / 'data/local_abstract_db.json'
    output_dir = repo_root / 'data/combined_final'

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load abstracts from the targeted scraping database
    print(f"Loading abstracts from local DB: {db_path}")
    if db_path.exists():
        with db_path.open('r', encoding='utf-8') as f:
            scraped_abstracts = json.load(f)
    else:
        scraped_abstracts = {}
    print(f" -> Found {len(scraped_abstracts)} scraped abstracts.")

    source_files = sorted(conf_dir.glob('*.jsonl'))
    
    print(f"\nConsolidating abstracts for {len(source_files)} conference files...")

    for source_file in tqdm(source_files, desc="Combining sources"):
        # 2. Load abstracts from the initial 'repaired' pass for this conference
        repaired_file = repaired_dir / f"enriched_{source_file.name}"
        repaired_abstracts = load_abstracts_from_file(repaired_file)

        # 3. Start with original records and enrich them
        final_records = []
        with source_file.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cid = canonical_id(rec)
                    
                    # Enrich in order of preference: scraped -> repaired
                    if cid in scraped_abstracts:
                        rec['abstract'] = scraped_abstracts[cid]
                    elif cid in repaired_abstracts:
                        rec['abstract'] = repaired_abstracts[cid]
                    
                    final_records.append(rec)
                except json.JSONDecodeError:
                    continue
        
        # 4. Write the truly final records to the new directory
        output_file = output_dir / source_file.name
        with output_file.open('w', encoding='utf-8') as f:
            for rec in final_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("\n--- Combination Complete ---")
    print(f"Final consolidated files are in: {output_dir}")

if __name__ == '__main__':
    main()

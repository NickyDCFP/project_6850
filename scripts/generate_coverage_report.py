#!/usr/bin/env python3
"""
Generates a report on abstract coverage across different enrichment stages.

This script compares the original source files, the first-pass enriched files,
and the final consolidated files to produce a table summarizing abstract coverage
for each conference-year pair.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# --- Utility Functions ---

def is_robust(text):
    """Checks if a text string is a non-trivial abstract."""
    if not text: return False
    s = str(text).strip()
    return len(s) >= 40 and len(s) != 195

def canonical_id(rec):
    """Creates a stable, unique ID for a paper record."""
    for f in ['paper_id', 'id', 'key']:
        if v := rec.get(f): return str(v)
    if doi := (rec.get('doi') or '').strip(): return doi.lower()
    if title := (rec.get('title') or '').strip(): return 'title:' + ' '.join(title.split()).lower()
    return None

def get_ids_with_abstracts(file_path):
    """Returns a set of canonical IDs for records in a file that have a robust abstract."""
    ids = set()
    if not file_path.exists():
        return ids
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                if is_robust(rec.get('abstract')) and (cid := canonical_id(rec)):
                    ids.add(cid)
            except json.JSONDecodeError:
                continue
    return ids

def get_all_ids(file_path):
    """Returns a set of all canonical IDs from a source file."""
    ids = set()
    if not file_path.exists():
        return ids
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                if cid := canonical_id(rec):
                    ids.add(cid)
            except json.JSONDecodeError:
                continue
    return ids

def main():
    conf_dir = repo_root / 'conf_entries'
    repaired_dir = repo_root / 'data/intermediate_data/enriched_repaired'
    final_dir = repo_root / 'data/final_dataset' # CORRECTED: Point to the consolidated data

    source_files = sorted(conf_dir.glob('*.jsonl'))

    results = []
    
    print("Analyzing abstract coverage across all conferences...")

    for source_file in source_files:
        conf_name, year_ext = source_file.name.rsplit('_', 1)
        year = year_ext.split('.')[0]

        # Use the consolidated file as the final state
        final_file = final_dir / source_file.name
        repaired_file = repaired_dir / f"enriched_{source_file.name}"

        all_source_ids = get_all_ids(source_file)
        repaired_ids_with_abstract = get_ids_with_abstracts(repaired_file)
        final_ids_with_abstract = get_ids_with_abstracts(final_file)

        total_papers = len(all_source_ids)
        if total_papers == 0:
            continue

        initially_missing = total_papers - len(repaired_ids_with_abstract)
        
        # CORRECTED: Calculate recovered as the set difference
        recovered_ids = final_ids_with_abstract - repaired_ids_with_abstract
        recovered = len(recovered_ids)

        final_missing = total_papers - len(final_ids_with_abstract)
        final_coverage = (len(final_ids_with_abstract) / total_papers) * 100 if total_papers > 0 else 0

        results.append({
            "conf": conf_name.upper(),
            "year": year,
            "total": total_papers,
            "initially_missing": initially_missing,
            "recovered": recovered,
            "final_missing": final_missing,
            "coverage": f"{final_coverage:.1f}%"
        })

    # --- Print Results Table ---
    if not results:
        print("No data found to generate a report.")
        return

    # Sort results for consistent output
    results.sort(key=lambda x: (x['conf'], x['year']))

    headers = ["Conference", "Year", "Total", "Initially Missing", "Recovered", "Final Missing", "Final Coverage"]
    
    # Calculate column widths
    widths = {h: len(h) for h in headers}
    for r in results:
        widths["Conference"] = max(widths["Conference"], len(r['conf']))
        widths["Year"] = max(widths["Year"], len(r['year']))
        widths["Total"] = max(widths["Total"], len(str(r['total'])))
        widths["Initially Missing"] = max(widths["Initially Missing"], len(str(r['initially_missing'])))
        widths["Recovered"] = max(widths["Recovered"], len(str(r['recovered'])))
        widths["Final Missing"] = max(widths["Final Missing"], len(str(r['final_missing'])))
        widths["Final Coverage"] = max(widths["Final Coverage"], len(r['coverage']))

    # Header
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(f"\n{header_line}")
    
    # Separator
    separator_line = "-|-".join("-" * widths[h] for h in headers)
    print(separator_line)

    # Data rows
    for r in results:
        row_str = (
            f"{r['conf'].ljust(widths['Conference'])} | "
            f"{r['year'].ljust(widths['Year'])} | "
            f"{str(r['total']).rjust(widths['Total'])} | "
            f"{str(r['initially_missing']).rjust(widths['Initially Missing'])} | "
            f"{str(r['recovered']).rjust(widths['Recovered'])} | "
            f"{str(r['final_missing']).rjust(widths['Final Missing'])} | "
            f"{r['coverage'].rjust(widths['Final Coverage'])}"
        )
        print(row_str)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analyzes the URL patterns for papers that are still missing abstracts.

This script focuses on specified conferences (FAT, ICML, NeurIPS) and
identifies the most common domains found in the 'ee' and 'url' fields
of records without a robust abstract. This helps prioritize the development
of new custom scrapers.
"""
import sys
import json
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

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

def main():
    combined_dir = repo_root / 'data/combined_final'
    target_conferences = ['fat', 'icml', 'neurips']
    
    domain_counter = Counter()
    missing_count = 0
    no_link_count = 0
    malformed_link_count = 0
    
    print(f"Analyzing link patterns for missing abstracts in {', '.join(c.upper() for c in target_conferences)}...")

    all_files = sorted(combined_dir.glob('*.jsonl'))
    
    # Filter for target conference files
    target_files = [
        f for f in all_files 
        if any(f.name.startswith(conf) for conf in target_conferences)
    ]

    for file_path in target_files:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if not is_robust(rec.get('abstract')):
                        missing_count += 1
                        # Prioritize the 'ee' field as it's often the direct link
                        link = rec.get('ee') or rec.get('url')
                        if not link or not isinstance(link, str):
                            no_link_count += 1
                            continue

                        if link.startswith('http'):
                            try:
                                domain = urlparse(link).netloc
                                # Clean up common subdomains
                                if domain.startswith('www.'):
                                    domain = domain[4:]
                                domain_counter[domain] += 1
                            except Exception:
                                malformed_link_count += 1
                        else:
                            malformed_link_count += 1
                except json.JSONDecodeError:
                    continue

    print(f"\nFound {missing_count} total records missing abstracts in target conferences.")
    print("Breakdown of link sources for missing abstracts:")
    print("-" * 50)

    if not domain_counter and no_link_count == 0 and malformed_link_count == 0:
        print("No missing abstracts found in target conferences.")
        return

    top_domains = domain_counter.most_common(10)
    top_domain_total = sum(count for _, count in top_domains)
    other_domain_total = sum(domain_counter.values()) - top_domain_total

    print(f"{'Source Type':<30} | {'Count'}")
    print(f"{'-'*29}-|-------")
    
    for domain, count in top_domains:
        print(f"{domain:<30} | {count}")

    if other_domain_total > 0:
        print(f"{'Other Domains':<30} | {other_domain_total}")
        
    print(f"{'Malformed/Non-HTTP Link':<30} | {malformed_link_count}")
    print(f"{'No Link Provided':<30} | {no_link_count}")
    print("-" * 50)
    total_accounted_for = top_domain_total + other_domain_total + malformed_link_count + no_link_count
    print(f"{'Total Accounted For':<30} | {total_accounted_for}")


if __name__ == '__main__':
    main()

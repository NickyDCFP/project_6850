#!/usr/bin/env python3
"""
Comprehensive, multi-stage enrichment script to fill in missing abstracts.

This script applies a series of strategies to find missing abstracts for
papers in the target conferences (FAT, ICML, NeurIPS).

The process for each missing paper is as follows:
1.  **Direct Scraping**: If a valid URL to a known site (proceedings.mlr.press,
    arxiv.org, openreview.net) is found in the 'ee' or 'url' field, a custom
    scraper for that site is used.
2.  **DOI Resolution**: If the URL is a doi.org link, the script follows the
    redirect to the actual paper page and attempts to scrape it.
3.  **API Fallback**: If the above methods fail or no valid URL is present,
    the script falls back to searching public APIs (Semantic Scholar, OpenAlex)
    using the paper's title.

The script updates the files in `data/combined_final/` in-place.
"""
import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException

# Ensure repo root is on path and load environment variables
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

load_dotenv(repo_root / '.env')

from tqdm import tqdm
# Import the existing API-based enrichment logic
from add_abstracts import fill_abstract_for_record as api_enrich

# --- Utility Functions ---
def is_robust(text):
    if not text: return False
    s = str(text).strip()
    return len(s) >= 40 and len(s) != 195

def request_with_retries(url, timeout=15, max_retries=3, backoff_factor=0.5, allow_redirects=True):
    """Make a GET request with simple retries and backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                timeout=timeout, 
                headers={'User-Agent': 'Mozilla/5.0'},
                allow_redirects=allow_redirects
            )
            response.raise_for_status()
            return response
        except RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff_factor * (2 ** attempt))
    return None

# --- Custom Scrapers ---
def scrape_abstract_from_html(html_content):
    """
    Scrapes an abstract from raw HTML content using a series of patterns.
    """
    if not html_content:
        return None

    # Pattern 1: PMLR (proceedings.mlr.press)
    m = re.search(r'<div\s+id=["\']abstract["\']\s*class=["\']abstract["\']\s*>\s*(<p>)?(.*?)(</p>)?\s*</div>', html_content, re.DOTALL | re.IGNORECASE)
    if m and is_robust(m.group(2)):
        return re.sub(r'<.*?>|\s+', ' ', m.group(2)).strip()

    # Pattern 2: OpenGraph meta tag (common on OpenReview, etc.)
    m = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
    if m and is_robust(m.group(1)):
        return re.sub(r'\s+', ' ', m.group(1)).strip()

    # Pattern 3: ArXiv specific meta tag
    m = re.search(r'<meta\s+name=["\']citation_abstract["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE)
    if m and is_robust(m.group(1)):
        return re.sub(r'\s+', ' ', m.group(1)).strip()

    # Pattern 4: ArXiv abstract block
    m = re.search(r'<blockquote class="abstract mathjax">[\s\S]*?<span class="descriptor">Abstract:</span>([\s\S]*?)</blockquote>', html_content, re.IGNORECASE)
    if m and is_robust(m.group(1)):
        return re.sub(r'<.*?>|\s+', ' ', m.group(1)).strip()

    return None

# --- Main Enrichment Logic ---
def get_abstract(rec, log_f):
    """
    Main function to orchestrate the multi-stage enrichment for a single record.
    """
    # Stage 1: Attempt direct scraping from a valid URL
    link = rec.get('ee') or rec.get('url')
    if link and isinstance(link, str) and link.startswith('http'):
        domain = urlparse(link).netloc
        log_f.write(f"  - Found URL: {link}\n")
        
        # Handle DOI links
        if domain == 'doi.org':
            log_f.write("  - Resolving DOI...\n")
            response = request_with_retries(link, allow_redirects=True)
            if response:
                final_url = response.url
                log_f.write(f"  - DOI resolved to: {final_url}\n")
                html_content = response.text
                abstract = scrape_abstract_from_html(html_content)
                if abstract:
                    return abstract, "scrape_doi_resolve"
            else:
                log_f.write("  - DOI resolution failed.\n")
        else: # Handle direct links
            response = request_with_retries(link)
            if response:
                html_content = response.text
                abstract = scrape_abstract_from_html(html_content)
                if abstract:
                    return abstract, "scrape_direct"

    # Stage 2: Fallback to API search by title
    log_f.write("  - No valid link or scraping failed. Falling back to API search...\n")
    
    # The api_enrich function modifies the record in-place.
    # We can detect success by checking if the 'enriched_from' list grows.
    original_enriched_sources = set(rec.get('enriched_from', []))
    
    enriched_rec = api_enrich(rec) # Returns the modified record
    
    new_enriched_sources = set(enriched_rec.get('enriched_from', []))
    
    added_sources = new_enriched_sources - original_enriched_sources

    if added_sources:
        # Success if a new source was added
        new_source = added_sources.pop()
        return enriched_rec.get('abstract'), f"api_{new_source}"
    
    return None, "failure"

def main():
    parser = argparse.ArgumentParser(description="Comprehensive multi-stage enrichment.")
    parser.add_argument('--data_dir', default='data/combined_final')
    parser.add_argument('--log_file', default='data/compare/comprehensive_enrich_log.txt')
    parser.add_argument('--limit_per_file', type=int, default=None)
    parser.add_argument('--conferences', default='fat,icml,neurips', help='Comma-separated list of conference prefixes to process.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    target_conferences = [conf.strip().lower() for conf in args.conferences.split(',')]
    
    # Ensure log directory exists
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    target_files = [f for f in sorted(data_dir.glob('*.jsonl')) if any(f.name.startswith(conf) for conf in target_conferences)]
    
    print(f"Starting comprehensive enrichment on {len(target_files)} files for {', '.join(c.upper() for c in target_conferences)}...")

    with open(args.log_file, 'w', encoding='utf-8') as log_f:
        for file_path in tqdm(target_files, desc="Processing Conferences"):
            records_to_process = []
            all_records = []
            
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

            enriched_map = {}
            pbar = tqdm(records_to_process, desc=f"Enriching {file_path.name}", leave=False)
            for rec in pbar:
                title = rec.get('title', 'N/A')[:50]
                log_f.write(f"Attempting '{title}...'\n")
                
                abstract, source = get_abstract(rec, log_f)
                
                if abstract:
                    log_f.write(f"  -> SUCCESS via {source}\n")
                    rec['abstract'] = abstract
                    rec.setdefault('enriched_from', []).append(source)
                else:
                    log_f.write("  -> FAILURE on all stages\n")
                
                enriched_map[id(rec)] = rec

            # Overwrite the file with updated records
            with file_path.open('w', encoding='utf-8') as f:
                for rec in all_records:
                    if id(rec) in enriched_map:
                        f.write(json.dumps(enriched_map[id(rec)], ensure_ascii=False) + '\n')
                    else:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print("\n--- Comprehensive Enrichment Complete ---")
    print(f"All target files in '{data_dir}' have been updated.")
    print(f"Detailed log is at: {args.log_file}")

if __name__ == '__main__':
    main()

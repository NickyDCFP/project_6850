#!/usr/bin/env python3
"""
Advanced filtering script for ethics and safety papers.

This script refines the filtering process by incorporating a pre-existing cache of
filtered papers, modifying the scoring logic, and applying different filtering
strategies for different conferences.

New Logic:
1.  **Load a cache of pre-filtered papers**: It reads from `data/filtered_ethics_safety_papers.jsonl`
    and `data/intermediate_data/filter_logs/` to create an index of papers that have already been
    processed by the Gemini filter. This avoids redundant API calls.

2.  **Modified Scoring**: The `safety_score` and `ethics_score` are now raw counts of
    keyword matches, not proportions.

3.  **Two-Tiered Filtering Strategy**:
    -   **Small Conferences (`aies`, `fat`, `satml`)**: All papers are considered candidates
        and are sent directly to the Gemini filter (if not already in the cache).
    -   **Large Conferences (all others)**: Papers first undergo a keyword search. Only those
        with at least one keyword match are considered candidates for the Gemini filter.

4.  **Output**:
    -   Final accepted papers are written to `data/final_filtered/filtered_{conf}.jsonl`.
    -   Detailed logs for every paper processed by Gemini are written to
        `data/final_filtered/logs/{conf}_filter_detailed.jsonl`.

Usage:
  python3 scripts/advanced_filter.py --conf aies --years 2020-2025
  python3 scripts/advanced_filter.py --conf aaai --years 2020-2025
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set
import sys

# --- Setup Paths and Imports ---
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))

from tqdm import tqdm
from keywords import keywords as KWDS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional Gemini filter
try:
    from llm_paper_filter import GeminiEthicsSafetyPaperFilter, FilterResult
    _HAS_GEMINI = True
except Exception:
    import traceback
    tb = traceback.format_exc()
    print('Failed to import llm_paper_filter (Gemini not available). Import traceback:\n', tb)
    GeminiEthicsSafetyPaperFilter = None
    _HAS_GEMINI = False

# --- Constants ---
DATA_DIR = repo_root / 'data'
FINAL_DATA_DIR = DATA_DIR / 'final_dataset'
OUTPUT_DIR = DATA_DIR / 'final_filtered'
LOG_DIR = OUTPUT_DIR / 'logs'
CACHE_LOG_DIR = DATA_DIR / 'intermediate_data' / 'filter_logs'
PRE_FILTERED_FILE = DATA_DIR / 'filtered_ethics_safety_papers.jsonl'

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

SMALL_CONFS = {'aies', 'fat', 'satml'}

# --- Helper Functions ---

def norm_text(s: str) -> str:
    return (s or '').lower()

def get_canonical_id(rec: Dict[str, Any]) -> str:
    """Creates a stable ID for a paper based on title and year."""
    title = norm_text(rec.get('title', ''))
    year = rec.get('year', '')
    return f"{title}_{year}"

def count_keyword_matches(text: str, kw_list: List[str]) -> int:
    """Counts raw occurrences of keywords in text."""
    s = norm_text(text)
    return sum(1 for kw in kw_list if kw.lower() in s)

def score_paper(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Computes raw safety and ethics scores."""
    txt = ' '.join([norm_text(rec.get('title')), norm_text(rec.get('abstract'))])
    safety_kw = KWDS.get('safety', [])
    ethics_kw = KWDS.get('ethics', [])

    safety_matches = count_keyword_matches(txt, safety_kw)
    ethics_matches = count_keyword_matches(txt, ethics_kw)

    category = 'safety' if safety_matches > ethics_matches else 'ethics'

    rec.update({
        'safety_score': safety_matches,
        'ethics_score': ethics_matches,
        'keyword_category': category,
        'keyword_matches': {'safety': safety_matches, 'ethics': ethics_matches}
    })
    return rec

def extract_filter_result(obj) -> Dict[str, Any]:
    """Normalizes return shapes from Gemini filter."""
    res = {'filter_result': 'unknown', 'filter_confidence': None, 'filter_reasoning': None}
    if obj is None:
        return res

    d = {}
    if hasattr(obj, '__dict__'):
        d = obj.__dict__
    elif isinstance(obj, dict):
        d = obj

    if not d:
        return res

    # Decision
    if 'include' in d:
        res['filter_result'] = 'pass' if bool(d.get('include')) else 'fail'
    elif 'passed' in d:
        res['filter_result'] = 'pass' if bool(d.get('passed')) else 'fail'
    elif 'decision' in d or 'result' in d:
        val = (d.get('decision') or d.get('result') or '').lower()
        res['filter_result'] = 'pass' if 'pass' in val or 'accept' in val else 'fail'
    elif 'category' in d:
        cat = (d.get('category') or '').lower()
        res['filter_result'] = 'pass' if 'safety' in cat or 'ethic' in cat else 'fail'

    # Confidence and Reasoning
    res['filter_confidence'] = d.get('confidence') or d.get('score')
    res['filter_reasoning'] = d.get('reasoning') or d.get('explanation') or d.get('rationale')

    return res

def load_filter_cache() -> Dict[str, Dict[str, Any]]:
    """Loads all existing filter decisions into a cache."""
    cache = {}
    files_to_process = list(CACHE_LOG_DIR.glob('*_filter_detailed.jsonl'))
    if PRE_FILTERED_FILE.exists():
        files_to_process.append(PRE_FILTERED_FILE)
    
    # Also include previously filtered files
    files_to_process.extend(list(DATA_DIR.glob('filtered_deduped_*.jsonl')))

    print(f"Loading filter cache from {len(files_to_process)} files...")
    for file_path in tqdm(files_to_process, desc="Caching decisions"):
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # Special handling for pre_filtered file, which implies a 'pass'
                    if file_path == PRE_FILTERED_FILE and 'filter_result' not in rec:
                        rec['filter_result'] = 'pass'
                        
                    # Check for a valid filter decision
                    if rec.get('filter_result') in ('pass', 'fail'):
                        key = get_canonical_id(rec)
                        if key not in cache:
                            cache[key] = {
                                'filter_result': rec['filter_result'],
                                'filter_confidence': rec.get('filter_confidence'),
                                'filter_reasoning': rec.get('filter_reasoning'),
                                'filter_category': rec.get('filter_category') or rec.get('category'),
                            }
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(cache)} cached decisions.")
    return cache

# --- Main Execution ---

def run_for_conf(conf: str, years: List[int], run_gemini: bool, gemini_chunk: int, filter_cache: Dict, limit: int = 0):
    # 1. Gather input files from the deduplicated final dataset
    input_files = []
    for y in years:
        p = FINAL_DATA_DIR / f'{conf}_{y}.jsonl'
        if p.exists():
            input_files.append(p)

    if not input_files:
        print(f"No files found in data/final_dataset/ for {conf} and years {years}.")
        return

    # 2. Prepare output files
    filtered_out_path = OUTPUT_DIR / f'filtered_{conf}.jsonl'
    detailed_log_path = LOG_DIR / f'{conf}_filter_detailed.jsonl'

    # 3. Initialize Gemini Filter
    gemini_filter = None
    if run_gemini and _HAS_GEMINI and os.getenv('GEMINI_API_KEY'):
        try:
            gemini_filter = GeminiEthicsSafetyPaperFilter(api_key=os.getenv('GEMINI_API_KEY'))
            print('Initialized Gemini filter.')
        except Exception as e:
            print(f'Failed to initialize Gemini filter: {e}')
    elif run_gemini:
        print('GEMINI_API_KEY not set or llm_paper_filter not available. Skipping Gemini stage.')

    # 4. Process papers: determine candidates
    all_papers = []
    for p in input_files:
        with p.open('r', encoding='utf-8') as f:
            all_papers.extend([json.loads(line) for line in f])

    candidates = []
    is_small_conf = conf in SMALL_CONFS

    for rec in tqdm(all_papers, desc=f"Scoring papers for {conf}"):
        rec = score_paper(rec)
        # For small conferences, all papers are candidates.
        # For large conferences, only papers with keyword matches are candidates.
        if is_small_conf or max(rec['safety_score'], rec['ethics_score']) > 0:
            candidates.append(rec)

    print(f"Found {len(candidates)} candidates for filtering out of {len(all_papers)} total papers.")

    # 5. Separate cached papers from those needing Gemini filtering
    accepted_from_cache = []
    gemini_candidates = []
    all_candidates_processed = [] # Will hold all candidates with their final filter status

    for rec in tqdm(candidates, desc="Checking cache"):
        key = get_canonical_id(rec)
        cached_result = filter_cache.get(key)

        if cached_result:
            rec.update(cached_result)
            if rec.get('filter_result') == 'pass':
                accepted_from_cache.append(rec)
        else:
            gemini_candidates.append(rec)  # Needs to be filtered by Gemini

    print(f"{len(accepted_from_cache)} papers accepted from cache. {len(gemini_candidates)} candidates for Gemini.")

    # This will be our final list of all accepted papers
    all_accepted_papers = list(accepted_from_cache)
    
    # 6. Run Gemini filter on remaining candidates
    if gemini_filter and gemini_candidates:
        # Apply limit if specified
        if limit > 0:
            print(f"Applying test limit: only processing {limit} of {len(gemini_candidates)} new candidates.")
            gemini_candidates = gemini_candidates[:limit]

        print(f"Running Gemini filter on {len(gemini_candidates)} new candidates...")
        
        progress = tqdm(total=len(gemini_candidates), unit='papers', desc=f'Gemini {conf}')
        for i in range(0, len(gemini_candidates), gemini_chunk):
            batch = gemini_candidates[i:i+gemini_chunk]
            try:
                # This now returns a list of FilterResult objects, one for each paper in the batch
                results = gemini_filter.filter_batch(batch, chunk_size=len(batch))
            except Exception as e:
                print(f"Gemini filter_batch failed: {e}")
                # Create a list of error results matching the batch size
                results = [FilterResult(include=False, category='error', confidence=0.0, reasoning=str(e))] * len(batch)            

            for paper, result_obj in zip(batch, results):
                info = extract_filter_result(result_obj)
                paper.update(info)

                # --- VERBOSE PRINTING ADDED FOR DEBUGGING ---
                verdict = paper.get('filter_result', 'unknown').upper()
                print(f"  - Gemini Result: {verdict} for paper '{paper.get('title', 'Untitled')[:60]}...'")
                # -----------------------------------------
                
                # If it passed, add it to our final list
                if paper.get('filter_result') == 'pass':
                    all_accepted_papers.append(paper)
            progress.update(len(batch))
        progress.close()
        print(f"Gemini filtering complete. Total accepted: {len(all_accepted_papers)}")

    # 7. Write the detailed log file *after* all filtering is complete
    with detailed_log_path.open('w', encoding='utf-8') as detfh:
        # Combine all papers that went through the process
        processed_papers = accepted_from_cache + gemini_candidates
        for paper in processed_papers:
            detfh.write(json.dumps(paper, ensure_ascii=False) + '\n')

    # 8. Write final accepted papers, ensuring no duplicates
    final_accepted_ids = set()
    with filtered_out_path.open('w', encoding='utf-8') as f:
        for rec in all_accepted_papers:
            key = get_canonical_id(rec)
            if key not in final_accepted_ids:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                final_accepted_ids.add(key)

    print(f"Wrote {len(final_accepted_ids)} unique accepted papers to {filtered_out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced filtering for ethics and safety papers.")
    parser.add_argument('--conf', required=True, help='Conference token (e.g., aies, aaai).')
    parser.add_argument('--years', default='2020,2021,2022,2023,2024,2025', help='Comma-separated years to process.')
    parser.add_argument('--no-gemini', action='store_true', help='Skip Gemini filtering, even if API key is present.')
    parser.add_argument('--chunk', type=int, default=8, help='Batch size for Gemini API calls.')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of new papers sent to Gemini (for testing).')
    args = parser.parse_args()

    years_to_process = [int(y.strip()) for y in args.years.split(',') if y.strip()]

    # Load the cache once
    cache = load_filter_cache()

    run_for_conf(
        conf=args.conf.lower(),
        years=years_to_process,
        run_gemini=(not args.no_gemini),
        gemini_chunk=args.chunk,
        filter_cache=cache,
        limit=args.limit
    )

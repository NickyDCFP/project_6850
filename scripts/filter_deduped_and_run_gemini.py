#!/usr/bin/env python3
"""
Filter deduped enriched files by keywords and (optionally) run the Gemini filter.

Steps:
1. Read `data/deduped_enriched_<conf>_<year>.jsonl` files for the requested conference(s).
2. Apply keyword scoring using `src/keywords.py` (normalized count per category).
   - safety_score = matches_safety / len(safety_kwds)
   - ethics_score = matches_ethics / len(ethics_kwds)
   - category = 'safety' or 'ethics' depending on which score is larger (ties -> 'ethics')
   - record gets fields: safety_score, ethics_score, category
3. Keep papers that have max(score) > 0 as candidates. For those, run the Gemini filter if available
   (GeminiEthicsSafetyPaperFilter). The script is resilient if Gemini API key is not set or the filter class
   cannot be instantiated: in that case `filter_result` will be 'skipped'.
4. Write accepted papers (filter pass) to `data/filtered_deduped_<conf>.jsonl` (aggregated across years).
5. Write bookkeeping logs for all processed papers to `data/filter_logs/<conf>_filter_log.jsonl` with the
   filter decision and metadata.

Usage:
  python3 scripts/filter_deduped_and_run_gemini.py --conf aaai --years 2020-2025

Notes / assumptions:
- The Gemini filter class `GeminiEthicsSafetyPaperFilter` is assumed to expose `filter_batch(papers, chunk_size=...)`
  which returns a list of results. Each result may be a dataclass or dict; this script will attempt to
  extract `category`/`passed`/`confidence`/`reasoning` from either form.
- If the Gemini API key is not set (GEMINI_API_KEY), the Gemini-stage is skipped and candidates are
  still written to the bookkeeping log with `filter_result='skipped'`.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# load keywords
import sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))
# Ensure repo root is on sys.path so sibling modules (e.g. llm_paper_filter.py) can be imported
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from tqdm import tqdm
from keywords import keywords as KWDS

# optional Gemini filter
try:
    from llm_paper_filter import GeminiEthicsSafetyPaperFilter
    _HAS_GEMINI = True
except Exception:
    # Surface the import exception so the user can see why Gemini is unavailable
    import traceback
    tb = traceback.format_exc()
    print('Failed to import llm_paper_filter (Gemini not available). Import traceback:\n', tb)
    GeminiEthicsSafetyPaperFilter = None
    _HAS_GEMINI = False

DATA = Path('data')
LOG_DIR = DATA / 'filter_logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def norm_text(s: str) -> str:
    return (s or '').lower()


def count_keyword_matches(text: str, kw_list: List[str]) -> int:
    s = norm_text(text)
    c = 0
    for kw in kw_list:
        if kw.lower() in s:
            c += 1
    return c


def score_paper(rec: Dict[str,Any]) -> Dict[str,Any]:
    # compute safety and ethics normalized scores based on title+abstract
    txt = ' '.join([str(rec.get('title') or ''), str(rec.get('abstract') or '')])
    safety_kw = KWDS.get('safety', [])
    ethics_kw = KWDS.get('ethics', [])
    safety_matches = count_keyword_matches(txt, safety_kw)
    ethics_matches = count_keyword_matches(txt, ethics_kw)
    safety_score = safety_matches / max(1, len(safety_kw))
    ethics_score = ethics_matches / max(1, len(ethics_kw))
    if safety_score > ethics_score:
        category = 'safety'
    else:
        category = 'ethics'
    rec['safety_score'] = safety_score
    rec['ethics_score'] = ethics_score
    rec['keyword_category'] = category
    rec['keyword_matches'] = {'safety': safety_matches, 'ethics': ethics_matches}
    return rec


def extract_filter_result(obj) -> Dict[str,Any]:
    # Normalize possible return shapes from Gemini filter
    res = {'filter_result': 'unknown', 'filter_confidence': None, 'filter_reasoning': None}
    if obj is None:
        return res
    # dataclass support
    try:
        # dataclass-like with attributes
        if hasattr(obj, '__dict__'):
            d = obj.__dict__
        else:
            d = dict(obj)
    except Exception:
        try:
            d = dict(obj)
        except Exception:
            d = None
    if not d:
        return res
    # map likely keys robustly
    # 1) boolean flag names used by some implementations
    if 'include' in d:
        res['filter_result'] = 'pass' if bool(d.get('include')) else 'fail'
    elif 'passed' in d:
        res['filter_result'] = 'pass' if bool(d.get('passed')) else 'fail'
    # 2) explicit decision/result strings
    elif 'decision' in d or 'result' in d:
        val = (d.get('decision') or d.get('result') or '').lower()
        if 'include' in val or 'pass' in val or 'accept' in val:
            res['filter_result'] = 'pass'
        elif 'exclude' in val or 'fail' in val or 'reject' in val:
            res['filter_result'] = 'fail'
        else:
            res['filter_result'] = 'unknown'
    # 3) category labels (map safety/ethic-related labels to pass)
    elif 'category' in d:
        cat = (d.get('category') or '').lower()
        if 'safety' in cat or 'ethic' in cat:
            res['filter_result'] = 'pass'
        else:
            res['filter_result'] = 'fail'
    else:
        res['filter_result'] = 'unknown'

    # confidence and reasoning fields (try several common names)
    res['filter_confidence'] = d.get('confidence') or d.get('score') or d.get('confidence_score')
    res['filter_reasoning'] = d.get('reasoning') or d.get('explanation') or d.get('rationale') or d.get('notes')
    return res


def run_for_conf(conf: str, years: List[int], run_gemini: bool = True, gemini_chunk: int = 8):
    # gather deduped files for conf
    files = []
    for y in years:
        p = DATA / f'deduped_enriched_{conf}_{y}.jsonl'
        if p.exists():
            files.append(p)
    if not files:
        print('No deduped enriched files found for', conf)
        return

    # prepare outputs
    filtered_out = DATA / f'filtered_deduped_{conf}.jsonl'
    bookkeeping = LOG_DIR / f'{conf}_filter_log.jsonl'

    # instantiate Gemini filter if requested and available
    gemini_filter = None
    if run_gemini and _HAS_GEMINI and os.getenv('GEMINI_API_KEY'):
        try:
            gemini_filter = GeminiEthicsSafetyPaperFilter(api_key=os.getenv('GEMINI_API_KEY'))
            print('Initialized Gemini filter')
        except Exception as e:
            print('Failed to initialize Gemini filter:', e)
            gemini_filter = None
    else:
        if run_gemini and not _HAS_GEMINI:
            print('Gemini filter class not available; skipping Gemini stage')
        if run_gemini and not os.getenv('GEMINI_API_KEY'):
            print('GEMINI_API_KEY not set; skipping Gemini stage')

    # stream papers, apply keyword filter
    candidates = []
    total_seen = 0
    with bookkeeping.open('w', encoding='utf-8') as logfh:
        for p in files:
            with p.open('r', encoding='utf-8') as fh:
                for line in fh:
                    total_seen += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rec = score_paper(rec)
                    # candidate if any positive score
                    if max(rec['safety_score'], rec['ethics_score']) > 0:
                        candidates.append(rec)
                    # always write a bookkeeping entry for audit; mark as 'keyword_candidate' flag
                    bk = {**rec}
                    bk['keyword_candidate'] = (max(rec['safety_score'], rec['ethics_score']) > 0)
                    # placeholder for filter results
                    bk.update({'filter_result': None, 'filter_confidence': None, 'filter_reasoning': None})
                    logfh.write(json.dumps(bk, ensure_ascii=False) + '\n')

    print(f'{total_seen} total papers scanned; {len(candidates)} keyword candidates')

    # if no Gemini, we stop here and write the keywords-only positives to bookkeeping (done)
    if not gemini_filter:
        print('Gemini not available; writing keyword-positive papers to filtered output (no LLM check)')
        with filtered_out.open('w', encoding='utf-8') as fo:
            for rec in candidates:
                rec_out = {k: rec.get(k) for k in ['id','type','title','authors','year','venue','url','abstract','enriched_from']}
                rec_out.update({'safety_score': rec['safety_score'], 'ethics_score': rec['ethics_score'], 'category': rec['keyword_category'], 'filter_result': 'skipped', 'filter_confidence': None, 'filter_reasoning': None})
                fo.write(json.dumps(rec_out, ensure_ascii=False) + '\n')
        return

    # Resume support: skip candidates already processed and recorded in detailed log
    detailed_log = LOG_DIR / f'{conf}_filter_detailed.jsonl'
    processed_ids = set()
    if detailed_log.exists():
        try:
            with detailed_log.open('r', encoding='utf-8') as pf:
                for line in pf:
                    try:
                        j = json.loads(line)
                        if 'id' in j and j['id'] is not None:
                            processed_ids.add(j['id'])
                    except Exception:
                        continue
        except Exception:
            processed_ids = set()
    if processed_ids:
        orig_count = len(candidates)
        candidates = [c for c in candidates if c.get('id') not in processed_ids]
        print(f'Skipping {orig_count - len(candidates)} already-processed candidates; {len(candidates)} remain', flush=True)

    # Run Gemini filter in batches with visible progress
    accepted = []
    with detailed_log.open('a', encoding='utf-8') as detfh, filtered_out.open('a', encoding='utf-8') as fo:
        total = len(candidates)
        print(f'Starting Gemini filtering on {total} candidates (chunk={gemini_chunk})', flush=True)
        # progress bar over papers
        progress = tqdm(total=total, unit='papers', desc=f'Gemini {conf}')
        i = 0
        try:
            while i < total:
                batch = candidates[i:i+gemini_chunk]
                i += gemini_chunk
                try:
                    results = gemini_filter.filter_batch(batch, chunk_size=gemini_chunk)
                except Exception as e:
                    print('Gemini filter_batch failed on a batch:', e, flush=True)
                    results = [None] * len(batch)
                # normalize and write
                for rec, r in zip(batch, results):
                    # Some filter implementations return (paper, result) pairs; accept both shapes.
                    if isinstance(r, (list, tuple)) and len(r) == 2:
                        maybe_paper, real_res = r
                        # prefer the returned result, but accept if it's (paper, result)
                        r_obj = real_res
                    else:
                        r_obj = r
                    info = extract_filter_result(r_obj)
                    rec_out = {k: rec.get(k) for k in ['id','type','title','authors','year','venue','url','abstract','enriched_from']}
                    rec_out.update({'safety_score': rec['safety_score'], 'ethics_score': rec['ethics_score'], 'category': rec['keyword_category']})
                    rec_out.update({'filter_result': info['filter_result'], 'filter_confidence': info['filter_confidence'], 'filter_reasoning': info['filter_reasoning']})
                    detfh.write(json.dumps(rec_out, ensure_ascii=False) + '\n')
                    if info['filter_result'] in ('pass','pass_with_confidence','accept') or info['filter_result'] == 'pass':
                        fo.write(json.dumps(rec_out, ensure_ascii=False) + '\n')
                        accepted.append(rec_out)
                progress.update(len(batch))
                # small flushable status
                print(f'Processed {min(i,total)}/{total} candidates (accepted so far: {len(accepted)})', flush=True)
        except KeyboardInterrupt:
            print('\nGemini run interrupted by user. Partial results saved to logs.', flush=True)
        finally:
            progress.close()
    print(f'Gemini filtering finished/paused: {len(accepted)} accepted out of {len(candidates)} keyword candidates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, help='Conference token, e.g. aaai, neurips')
    parser.add_argument('--years', default='2020,2021,2022,2023,2024,2025')
    parser.add_argument('--no-gemini', action='store_true', help='Skip Gemini stage even if API key is present')
    parser.add_argument('--chunk', type=int, default=8, help='Gemini batch size')
    args = parser.parse_args()
    years = [int(y.strip()) for y in args.years.split(',') if y.strip()]
    run_for_conf(args.conf.lower(), years, run_gemini=(not args.no_gemini), gemini_chunk=args.chunk)

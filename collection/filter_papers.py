"""Keyword + Gemini filter for the papers added since the shipped corpus; the shipped corpus is kept as is.

  python -m collection.filter_papers --validate 160       # agreement with shipped decisions (~11 requests)
  python -m collection.filter_papers --max-calls 450      # judge new papers; rerun daily until done
  python -m collection.filter_papers --max-calls 0        # no calls: just rebuild the outputs

Which papers are new: every venue's 2026 papers, plus NeurIPS, ICML and AIES 2025 (the paper excluded
the first two as unpublished, and AIES 2025 was not yet in DBLP). Everything else from 2020-2025 is
taken verbatim from data/final_filtered/; 2020-2025 papers that lack a shipped decision (mostly ACL
volumes added to the Anthology later) are left out so that corpus stays exactly as published.

The two filter steps are the shipped ones (scripts/advanced_filter.py): papers with at least one
safety/ethics keyword in title+abstract (every paper, for AIES, FAccT and SaTML) go to the LLM in
batches from the same venue-year, with the prompt in llm_paper_filter.py. Three things differ from the
paper, all recorded on every decision:
  model       gemini-2.5-flash is no longer served to new API users; default gemini-3.1-flash-lite.
  prompt      the newer models rejected a third of shipped accepts (OOD detection, interpretability,
              privacy, fairness methods), so --prompt clarified (default) appends a scope note and nine
              shipped decisions as examples (CLARIFICATION below). On 160 held-out shipped decisions
              it agrees 88% (7 shipped accepts rejected, 12 shipped rejects accepted).
  batch size  20 papers per request (paper: 8), to fit the free tier's 500 requests/day.

Decisions are appended to data/updated/cache/filter_decisions.jsonl as they arrive, so a run that
stops (quota, Ctrl-C) loses nothing. Outputs:
  data/updated/filtered/<venue>_<year>.jsonl   new papers that passed (shipped-file fields)
  data/updated/filtered/logs/<venue>_<year>_filter_detailed.jsonl   every new candidate + decision
"""
import argparse
import ast
import json
import os
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

from tenacity import retry_if_exception, stop_after_attempt, wait_random_exponential

import collection.apis  # noqa: F401 (loads .env: GEMINI_API_KEY)
from collection.common import (CACHE_DIR, COLLECTED_DIR, FILTERED_DIR, REPO_ROOT, SMALL_VENUES, append_jsonl,
                               has_abstract, read_jsonl, write_jsonl)
from keywords import keywords  # noqa: E402 (src/ is put on sys.path by common)
from llm_paper_filter import SYSTEM_INSTRUCTION, GeminiEthicsSafetyPaperFilter  # noqa: E402

# Added to the shipped prompt for the newer models, which (validated on shipped decisions) reject
# general-ML robustness, OOD, interpretability, privacy and fairness methods that gemini-2.5-flash
# accepted. The examples are shipped decisions; they are excluded from --validate samples.
CLARIFICATION = """
**Clarification on scope (follow this strictly):**
Some technical machine-learning papers count even without any discussion of alignment, catastrophic risk or
society, but only when their MAIN contribution is one of these: out-of-distribution detection, adversarial or
certified robustness against attacks, formal verification of neural networks, interpretability or
explainability of models, privacy of training data or users (differential privacy, membership inference,
privacy attacks and defenses), or fairness and social bias (bias against people or groups) in AI systems.
Everything else that is general ML still FAILS, including: better generalization or accuracy under
distribution shift, uncertainty estimation or calibration, class imbalance or noisy labels, statistical
"bias" that is not about people (e.g. exposure bias in recommenders, estimation bias), data augmentation,
and generic security of computing systems. Words like "robust", "bias", "safe" or "trustworthy" alone do not
make a paper AI Safety or AI Ethics.

**Examples of previous decisions:**
- "Igeood: An Information Geometry Approach to Out-of-Distribution Detection" -- a new OOD detection method
  for any pre-trained network. -> include, ai_safety
- "Fundamental Limits in Formal Verification of Message-Passing Neural Networks" -- decidability of output
  reachability and adversarial robustness for GNNs. -> include, ai_safety
- "Piecewise Linear Parametrization of Policies: Towards Interpretable Deep Reinforcement Learning" -- an
  interpretable policy parametrization for RL agents. -> include, ai_safety
- "FLAME: Differentially Private Federated Learning in the Shuffle Model" -- a differentially private FL
  protocol with better privacy/utility trade-offs. -> include, ai_ethics
- "A Pairwise Fair and Community-preserving Approach to k-Center Clustering" -- clustering with pairwise
  fairness constraints for individuals. -> include, ai_ethics
- "Posterior Re-calibration for Imbalanced Datasets" -- improves accuracy on rare classes. -> exclude, none
- "Addressing Class Imbalance in Federated Learning" -- a loss for imbalanced client data. -> exclude, none
- "Conservative Data Sharing for Multi-Task Offline Reinforcement Learning" -- improves multi-task offline
  RL performance by sharing data across tasks. -> exclude, none
- "CALANet: Cheap All-Layer Aggregation for Human Activity Recognition" -- a more accurate, cheaper network
  for sensor-based activity recognition. -> exclude, none

When given several papers, return a JSON list with one object per paper, in the order given.
"""
PROMPTS = {'shipped': SYSTEM_INSTRUCTION, 'clarified': SYSTEM_INSTRUCTION + CLARIFICATION}
EXAMPLE_TITLES = {'igeood', 'fundamental limits in formal verification of message-passing', 'piecewise linear parametrization of policies',
                  'flame: differentially private federated', 'a pairwise fair and community-preserving',
                  'conservative data sharing for multi-task', 'calanet', 'posterior re-calibration for imbalanced',
                  'addressing class imbalance in federated learning'}

DECISIONS = os.path.join(CACHE_DIR, 'filter_decisions.jsonl')
SAFETY_KW = [k.lower() for k in keywords['safety']]
ETHICS_KW = [k.lower() for k in keywords['ethics']]
CHUNK = 8  # the paper's batch size; --chunk overrides it


def is_new(venue, year):
    return int(year) >= 2026 or (venue in ('neurips', 'icml', 'aies') and int(year) == 2025)


def score(rec):
    """Same scoring as scripts/advanced_filter.score_paper."""
    txt = f"{rec.get('title') or ''} {rec.get('abstract') or ''}".lower()
    s = sum(1 for k in SAFETY_KW if k in txt)
    e = sum(1 for k in ETHICS_KW if k in txt)
    rec.update(safety_score=s, ethics_score=e, keyword_category='safety' if s > e else 'ethics',
               keyword_matches={'safety': s, 'ethics': e})
    return rec


def is_candidate(venue, rec):
    return venue in SMALL_VENUES or max(rec['safety_score'], rec['ethics_score']) > 0


def as_list(authors):
    if isinstance(authors, str):
        try:
            return ast.literal_eval(authors)
        except (ValueError, SyntaxError):
            return [authors]
    return authors or []


def daily_quota(exc):
    msg = str(exc)
    return '429' in msg and ('PerDay' in msg or 'per day' in msg.lower() or 'RequestsPerDay' in msg)


class Judge:
    """Runs 8-paper batches through the shipped filter, counting every request against a budget."""

    def __init__(self, model, max_calls, prompt='clarified'):
        self.filter = GeminiEthicsSafetyPaperFilter(model=model, system_instruction=PROMPTS[prompt])
        self.model = model
        self.max_calls = max_calls
        self.requests = 0
        self.lock = threading.Lock()
        self.out_of_quota = False
        self.retry_reasons = Counter()

    def _count(self, _state):
        with self.lock:
            if self.out_of_quota or self.requests >= self.max_calls:
                raise RuntimeError('request budget reached')
            self.requests += 1

    def _log_retry(self, state):
        exc = state.outcome.exception() if state.outcome else None
        if exc is not None:
            with self.lock:
                self.retry_reasons[type(exc).__name__ + ' ' + str(exc)[:40]] += 1

    def __call__(self, batch):
        papers = [{**r, 'authors': as_list(r.get('authors'))} for r in batch]
        call = self.filter.filter_paper.retry_with(
            stop=stop_after_attempt(5), wait=wait_random_exponential(min=5, max=120), before=self._count,
            after=self._log_retry,
            retry=retry_if_exception(lambda e: not daily_quota(e) and 'budget' not in str(e)))
        try:
            return call(self.filter, papers)
        except Exception as e:  # tenacity wraps the last error
            last = getattr(getattr(e, 'last_attempt', None), 'exception', lambda: e)() or e
            if daily_quota(last):
                self.out_of_quota = True
            return None


def load_decisions():
    out = {}
    if os.path.exists(DECISIONS):
        for d in read_jsonl(DECISIONS):
            out[d['id']] = d
    return out


def run_batches(judge, batches, on_result, workers=2):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for batch in batches:
            if judge.out_of_quota or judge.requests >= judge.max_calls:
                break
            futures[pool.submit(judge, batch)] = batch
            if len(futures) >= workers * 2:  # keep the queue short so the budget check stays meaningful
                done = next(as_completed(futures))
                on_result(futures.pop(done), done.result())
        for fut in as_completed(futures):
            on_result(futures[fut], fut.result())


def decision_row(rec, res, model, prompt, chunk):
    return {'id': rec['id'], 'filter_result': 'pass' if res.include else 'fail', 'filter_category': res.category,
            'filter_confidence': res.confidence, 'filter_reasoning': res.reasoning, 'filter_model': model,
            'filter_prompt': prompt, 'filter_batch_size': chunk}


def validate(model, n, prompt, chunk):
    """Judge n shipped decisions (half pass, half fail, mixed venues) and report agreement."""
    pool = defaultdict(list)
    for path in glob(os.path.join(REPO_ROOT, 'data', 'final_filtered', 'logs', '*_filter_detailed.jsonl')):
        for r in read_jsonl(path):
            if (r.get('filter_result') in ('pass', 'fail') and has_abstract(r)
                    and not any(r['title'].lower().startswith(t) for t in EXAMPLE_TITLES)):
                r['venue_token'] = os.path.basename(path).split('_')[0]
                pool[r['filter_result']].append(r)
    rng = random.Random(0)
    sample = rng.sample(pool['pass'], n // 2) + rng.sample(pool['fail'], n // 2)
    by_venue = defaultdict(list)
    for r in sample:
        by_venue[r['venue_token']].append(r)
    batches = [rows[i:i + chunk] for rows in by_venue.values() for i in range(0, len(rows), chunk)]
    judge = Judge(model, max_calls=len(batches) + 4, prompt=prompt)
    table = Counter()
    disagreements = []

    def on_result(batch, results):
        for r, res in zip(batch, results or []):
            new = 'pass' if res.include else 'fail'
            table[r['filter_result'], new] += 1
            if new != r['filter_result']:
                disagreements.append((r['filter_result'], new, r['venue_token'], r['title'][:80], res.reasoning))

    run_batches(judge, batches, on_result)
    judged = sum(table.values())
    agree = table['pass', 'pass'] + table['fail', 'fail']
    print(f'{model}, prompt={prompt}, batch={chunk}: {judged}/{len(sample)} judged in {judge.requests} requests; '
          f'agreement with shipped decisions {agree}/{judged} ({100 * agree / max(judged, 1):.0f}%)')
    print(f'  shipped pass -> new pass {table["pass", "pass"]}, -> new fail {table["pass", "fail"]}')
    print(f'  shipped fail -> new fail {table["fail", "fail"]}, -> new pass {table["fail", "pass"]}')
    for old, new, venue, title, why in disagreements:
        print(f'  [{venue}] shipped {old}, now {new}: {title}\n      {why}')
    print('tokens:', judge.filter.usage)
    out = os.path.join(CACHE_DIR, f'validation_{model}_{prompt}_{chunk}_{n}.json')
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'model': model, 'prompt': prompt, 'chunk': chunk, 'n': n, 'requests': judge.requests,
                   'table': {f'{a}->{b}': c for (a, b), c in table.items()},
                   'disagreements': disagreements, 'usage': judge.filter.usage}, f, indent=1)
    print('saved', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='gemini-3.1-flash-lite')
    ap.add_argument('--max-calls', type=int, default=480, help='request budget for this run (free tier: 500/day)')
    ap.add_argument('--validate', type=int, metavar='N', help='only check agreement on N shipped decisions')
    ap.add_argument('--prompt', choices=sorted(PROMPTS), default='clarified')
    ap.add_argument('--chunk', type=int, default=20, help=f'papers per request (paper: {CHUNK})')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    if args.validate:
        return validate(args.model, args.validate, args.prompt, args.chunk)

    decisions = load_decisions()
    groups = {}
    for path in sorted(glob(os.path.join(COLLECTED_DIR, '*.jsonl'))):
        venue, year = os.path.basename(path)[:-6].rsplit('_', 1)
        if is_new(venue, year):
            groups[venue, year] = [score(r) for r in read_jsonl(path)]
    todo = [rows for rows in ([r for r in recs if is_candidate(v, r) and r['id'] not in decisions]
                              for (v, _), recs in sorted(groups.items())) if rows]
    batches = [rows[i:i + args.chunk] for rows in todo for i in range(0, len(rows), args.chunk)]
    print(f'{sum(map(len, todo))} new candidates without a decision -> {len(batches)} batches; '
          f'budget {args.max_calls} requests', flush=True)

    if args.max_calls and batches:
        judge = Judge(args.model, args.max_calls, args.prompt)
        done = Counter()

        def on_result(batch, results):
            if results:
                rows = [decision_row(r, res, args.model, args.prompt, args.chunk) for r, res in zip(batch, results)]
                append_jsonl(DECISIONS, rows)
                decisions.update((d['id'], d) for d in rows)
                done['papers'] += len(rows)
                done['passed'] += sum(d['filter_result'] == 'pass' for d in rows)
                if done['papers'] % 400 < args.chunk:
                    print(f'  {done["papers"]} judged ({done["passed"]} passed), {judge.requests} requests', flush=True)
            else:
                done['failed batches'] += 1

        run_batches(judge, batches, on_result, workers=args.workers)
        print('retried because of:', dict(judge.retry_reasons.most_common(5)))
        print(f'this run: {dict(done)}, {judge.requests} requests, tokens {judge.filter.usage}'
              + ('  -- stopped: daily quota reached' if judge.out_of_quota else ''))

    summary = []
    for (venue, year), recs in sorted(groups.items()):
        cands = [r for r in recs if is_candidate(venue, r)]
        judged = [{**r, **{k: v for k, v in decisions[r['id']].items() if k != 'id'}} for r in cands if r['id'] in decisions]
        passed = [r for r in judged if r['filter_result'] == 'pass']
        write_jsonl(os.path.join(FILTERED_DIR, f'{venue}_{year}.jsonl'), passed)
        write_jsonl(os.path.join(FILTERED_DIR, 'logs', f'{venue}_{year}_filter_detailed.jsonl'), judged)
        summary.append((venue, year, len(recs), len(cands), len(judged), len(passed)))
    print(f'\n{"venue":>8} {"year":>4} {"papers":>6} {"candidates":>10} {"judged":>6} {"passed":>6}')
    for row in summary:
        print('{:>8} {:>4} {:>6} {:>10} {:>6} {:>6}'.format(*row))
    left = sum(c - j for _, _, _, c, j, _ in summary)
    print(f'{left} candidates still to judge' if left else 'all new candidates judged')


if __name__ == '__main__':
    main()

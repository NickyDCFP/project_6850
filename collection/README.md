# Updating the corpus to the present, with citation data

The shipped corpus (`data/final_filtered/`, 6,442 papers, 2020-2025) is kept exactly as published.
These scripts add the papers published since (NeurIPS, ICML and AIES 2025, and every venue's 2026
papers), filter them the way the paper did, and collect citation data for the whole corpus.
Everything is written to `data/updated/`; nothing in `data/final_filtered/` is changed.

Run from the repository root, in order:

    # 0. dumps (gitignored) into data/raw/: https://dblp.org/xml/dblp.xml.gz, https://dblp.org/xml/dblp.dtd,
    #    https://aclanthology.org/anthology+abstracts.bib.gz
    python -m collection.collect_dblp        # ~6 min
    python -m collection.collect_acl         # ~1.5 min
    python -m collection.enrich_abstracts --stages pages s2-ids openalex-doi   # ~1-2 h, resumable (s2-title was not run)
    python -m collection.filter_papers --validate 160
    python -m collection.filter_papers --max-calls 450   # rerun until "all new candidates judged"
    python -m collection.fetch_citations     # ~2-3 h without a Semantic Scholar key, resumable
    python -m collection.build_corpus        # seconds

Unattended on Slurm (filter across daily quota resets, then citations, then `build_corpus`), from the
repository root: `sbatch --mail-user=<you> collection/run_pipeline.sbatch [request budget]`; progress in
`data/updated/cache/pipeline.log`.

Keys go in `.env` (gitignored): `GEMINI_API_KEY` (required for filtering), and optionally
`SEMANTIC_SCHOLAR_API_KEY` and `OPENALEX_API_KEY` (faster, higher limits).

## What each step does

| Step | Output | Notes |
|---|---|---|
| `collect_dblp` | `data/updated/collected/<venue>_<year>.jsonl` | AAAI, AIES, FAccT (`fat`), ICLR, ICML, NeurIPS, SaTML from the DBLP dump, `inproceedings` only (the shipped files also hold one `proceedings` record per volume). Adds `ee` and `doi`. Abstracts already in `data/final_dataset/` are reused by DBLP key. |
| `collect_acl` | `data/updated/collected/acl_<year>.jsonl` | Every ACL Anthology venue, via the shipped parser. Adds `id` and `authors_first_last` (the Anthology gives "Last, First"; `authors` is left as shipped). |
| `enrich_abstracts` | fills `abstract`, sets `abstract_source` | Official proceedings pages (NeurIPS, ICLR, PMLR for ICML), then Semantic Scholar by DOI/ACL id, OpenAlex by DOI, Semantic Scholar by title. |
| `filter_papers` | `data/updated/filtered/<venue>_<year>.jsonl`, `.../logs/` | Keyword step, then the LLM step, for new papers only. Decisions in `data/updated/cache/filter_decisions.jsonl`. |
| `fetch_citations` | `data/updated/citations/` | `papers.jsonl` (counts + reference lists), `edges.csv` (citations between corpus papers), `summary.txt`. |
| `build_corpus` | `data/updated/corpus.jsonl` | Shipped + new papers in one file, same keys on every row: integer `year`, `authors_first_last` for all venues (ACL included), raw keyword scores and filter fields, `is_proceedings_record` (14 shipped volume rows with no authors), Semantic Scholar id and citation counts, `s2_shared_with` (3 pairs: Findings/workshop copies and one AAAI/NeurIPS dual publication). Labels are left to the analysis loader. |

## Differences from the paper's pipeline

- **Model.** The paper used `gemini-2.5-flash`, which is no longer served to new API users. New papers
  are judged by `gemini-3.1-flash-lite`.
- **Prompt.** With the shipped prompt the newer models rejected 25-33% of the shipped accepts (OOD
  detection, interpretability, privacy and fairness methods), which the original model consistently
  accepted. `--prompt clarified` (the default) appends a scope note and nine shipped decisions as
  examples (`CLARIFICATION` in `filter_papers.py`).
- **Batch size.** 20 papers per request instead of 8, to fit the free tier's 500 requests a day.
- **Validation.** On 160 shipped decisions not used as examples (80 accepts, 80 rejects, all DBLP
  venues), the new setup agrees 88%: 7 shipped accepts rejected, 12 shipped rejects accepted, so it is
  slightly more lenient than the original run. The paper reports 91-94% agreement between the original
  filter and human annotators. Results: `data/updated/cache/validation_*.json`.
- **Scope.** 2020-2025 papers that never got a shipped decision (about 1,400, mostly ACL volumes the
  Anthology added after the paper's snapshot) are not judged, so the 2020-2025 corpus stays as published.
  ICML 2026 is on PMLR (volume 306) but not yet in DBLP, and NeurIPS 2026 is not yet published; neither
  is included.

## What is in git, and what you must rebuild

Committed: `data/updated/filtered/`, `data/updated/corpus.jsonl`, `data/updated/collected_manifest.tsv`, `data/updated/citations/` (including the
95 MB `papers.jsonl`), and in `data/updated/cache/` the filter decisions, abstract lookups, validation
results and run logs. Analyses need only these.

Not committed (gitignored, too large):

| Path | Size | How to get it back |
|---|---|---|
| `data/updated/cache/citations.jsonl` | 93 MB | Nothing to do: `python -m collection.fetch_citations` recreates it from `citations/papers.jsonl` when it is missing. No requests; identical. |
| `data/updated/collected/` | 268 MB | See below. |
| `data/raw/` | 1.4 GB | Step 0 (three downloads). |

### Rebuilding `collected/` (no API requests, no new papers)

    # download the three dumps into data/raw/ (step 0), then:
    python -m collection.collect_dblp
    python -m collection.collect_acl
    python -m collection.enrich_abstracts --cache-only

- **Same papers.** `data/updated/collected_manifest.tsv` (committed) lists every paper id in `collected/`,
  per file. The collect scripts keep only those papers, in that order, so a dump downloaded later does
  not add the papers DBLP or the Anthology indexed since. They print how many were ignored, and warn if
  a pinned paper is missing from the dump (removed or re-keyed upstream).
- **No requests.** `--cache-only` fills abstracts from the committed `cache/abstracts.jsonl` and contacts
  no API.
- **Exactness.** With the dumps below (sha256), the rebuild is identical apart from the ACL parser's
  `collected_at` timestamp (checked 2026-11-23: same ids, order and every other field in all 145,439 records). A later dump gives the same
  paper set, but DBLP/Anthology corrections to existing records (titles, DOIs, author names) come along.
- **Do not re-run `filter_papers` or `fetch_citations --refresh` after a rebuild.** Their outputs are
  committed. On a later dump, corrected titles or abstracts can change keyword matches and so create
  unjudged candidates, which `filter_papers` would send to Gemini.

To deliberately extend the corpus with newly indexed papers, run the collect scripts with `--all-papers`
(this rewrites the manifest), then the full pipeline above.

`collected/` is needed only to re-run the pipeline; `build_corpus` and the analyses do not read it.

Dumps used (downloaded 2026-11-20):

    0070082378b55f9db21293eef8a87de47181cbf60a9663b9749ba88fc5e816df  dblp.xml.gz
    2d864498f9eef9298662df11e3874171ff24cd4a21ab5974f1369c90acdb1ef6  dblp.dtd
    0b757e888d7d9145477a50cc78fd4b8dcc303af4147191ebb432d35b5be976a9  anthology+abstracts.bib.gz

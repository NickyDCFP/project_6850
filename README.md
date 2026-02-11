# Ethics vs Safety: Mapping AI Ethics and AI Safety Research

This repository contains data collection, filtering, and network analysis code for reproduction of [*Mind the Gap! Pathways Towards Unifying AI Safety and Ethics Research*](https://arxiv.org/abs/2512.10058). It includes scripts to parse sources (ACL Anthology, DBLP), enrich metadata, filter papers with LLM-assisted criteria, and analyze author/paper networks and perturbations.

You will need:
1. a Gemini API key (as the LLM-judge classifier uses Gemini).
2. A Semantic Scholar API key (to do abstract enrichment).
3. The [dblp.xml](https://dblp.org/faq/How+can+I+download+the+whole+dblp+dataset.html) dump.
4. The full [ACL Anthology](https://aclanthology.org/) with abstracts.

You can also access all of this data in `data/final_filtered` and run analyses directly, along with logs that show rejected examples.

## Overview

At a high level, the pipeline is:

1. **Collect** papers from ACL Anthology and DBLP.
2. **Enrich** abstracts/metadata when missing.
3. **Filter** to retain core AI ethics / AI safety research.
4. **Analyze** trends and networks (homophily, reachability, perturbations).
5. **Visualize** results and export web dashboards.

## Abstract Enrichment

Abstract enrichment is implemented in layers:

- **API/metadata enrichment:** `add_abstracts.py` provides `fill_abstract_for_record`, which fills missing abstracts using DOI/title lookups (OpenAlex, optional Semantic Scholar if `SEMANTIC_API_KEY` is set), plus arXiv and OpenReview APIs, and HTML fallbacks. It records provenance in `enriched_from`.
- **Comprehensive enrichment:** `scripts/comprehensive_enrich.py` runs a multi-stage pipeline: direct scraping from known sources (PMLR, arXiv, OpenReview), DOI resolution + scrape, then falls back to the API-based `add_abstracts.py` logic. It updates files in `data/combined_final/` in-place and logs decisions.
- **Consolidation:** `scripts/combine_all_sources.py` merges abstracts from `conf_entries/`, `data/enriched_repaired/`, and `data/local_abstract_db.json` into `data/combined_final/`.

### Reproducing abstract enrichment

Prerequisites: ensure `data/combined_final/` exists and set any optional keys (`SEMANTIC_API_KEY`) if you want Semantic Scholar enrichment.

    python3 scripts/comprehensive_enrich.py \
      --data_dir data/combined_final \
      --conferences fat,icml,neurips

To re-merge all enrichment sources into a fresh `data/combined_final/`:

    python3 scripts/combine_all_sources.py


## Repository Structure

Key directories and files:

- `src/`: Core parsing/scraping utilities.
- `scripts/`: One-off and experiment scripts (filtering, perturbations, plotting).
- `data/`: Raw, intermediate, and filtered datasets.

Notable entry points:

- `parse_dblp_xml.py` — Stream-parse DBLP XML into JSONL.
- `src/acl_anthology_scraper_v2.py` — Parse ACL Anthology BibTeX.
- `create_filtered_dataset.py` — Apply LLM-based filtering.
- `analyze_aggregated.py` — Combine datasets and run aggregate analysis.
- `scripts/weighted_homophily*.py` — Weighted homophily / ASP experiments.
- `scripts/random_removal_nulls.py` — Random-removal nulls.

## Quickstart

### 1) Environment

    python3 -m pip install -r requirements.txt

### 2) Parse DBLP

    python3 parse_dblp_xml.py \
      --input dblp.xml \
      --output data/dblp_conferences_papers.jsonl \
      --type inproceedings --venue neurips --year 2020

#### Details on DBLP XML parsing

1. Download and unzip the DBLP XML dump (for example `dblp.xml`).
2. Run the parser to extract inproceedings for NeurIPS 2020 (as an example):

```bash
python3 parse_dblp_xml.py --input dblp.xml --output neurips2020.jsonl --type inproceedings --venue neurips --year 2020 --limit 100
```

Notes
- The script uses iterative XML parsing (ElementTree.iterparse) so it's memory efficient.
- `--venue` does a substring match against the `booktitle` and `journal` fields.
- If you need more complex extraction (DOIs, abstracts), we can add extra fields.


### 3) Parse ACL Anthology

Use `src/acl_anthology_scraper_v2.py` to parse the bib file and build ACL datasets.

### 4) Filter to AI Ethics/Safety

    export GEMINI_API_KEY=YOUR_KEY
    python3 create_filtered_dataset.py \
      --filter --batch-size 64 --delay 2.0

### 5) Run Analysis

    python3 analyze_aggregated.py

---

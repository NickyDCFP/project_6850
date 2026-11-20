# New analyses (not in the paper)

Extensions of *Mind the Gap!* (arXiv 2512.10058) built on the helpers in `reproduction/`; see
`reproduction/README.md` for the corpus, graph and homophily definitions and their `legacy` vs. new
variants. Run from the repository root:

    python -m analysis.run_all
    python -m analysis.<script> --help

Outputs are written to `analysis/results/`.

## Homophily on cumulative time windows

`homophily_over_time.py` rebuilds the Table 2 graphs from papers published up to each end year
(2020, 2020-2021, ..., 2020-2025), or from each year alone with `--windows yearly`. Author labels and
the >= 2-papers cutoff are recomputed per window. Same `--version` / `--corpus` flags as Table 2,
plus `--levels` (default: no removal). The 2020-2025 row equals the Table 2 baseline.

| Window | Paper definition, author (global / safety / ethics) | Legacy, 3,248-paper corpus |
|---|---|---|
| 2020 | 54.6 / 63.2 / 74.5 (168 authors) | 100 / 100 / n/a (19 authors) |
| 2020-2021 | 44.9 / 63.1 / 59.6 | 73.2 / 35.3 / 68.6 |
| 2020-2022 | 37.9 / 49.9 / 53.7 | 88.9 / 62.2 / 86.5 |
| 2020-2023 | 29.8 / 48.0 / 44.4 | 85.6 / 48.0 / 83.4 |
| 2020-2024 | 25.8 / 43.3 / 38.0 | 83.6 / 70.4 / 73.2 |
| 2020-2025 | 22.9 / 40.6 / 34.9 | 83.1 / 73.5 / 68.2 |

The legacy graph keeps only its largest component, which is ~20 authors in 2020, so its early
rows are not meaningful.

# Reproducing *Mind the Gap!* (arXiv 2512.10058)

Scripts that recompute the paper's tables from `data/final_filtered/`, with notes on what does
and does not reproduce. Run from the repository root:

    pip install -r requirements.txt
    python -m reproduction.run_all          # everything (~4 min)
    python -m reproduction.<script> --help  # individual tables

Outputs are written to `reproduction/results/`.

## Old vs. new behaviour

Wherever the released code differs from what the paper describes, both versions can be selected:

| What | Old (`legacy`) | New |
|---|---|---|
| Corpus: `load_combined_data(corpus=...)` | `'legacy'`: original glob `*_filtered.jsonl`, which skips `filtered_acl_anthology.jsonl` and `filtered_neurips.jsonl` (3,248 papers) | `'full'` (default): all 8 files, 6,442 papers |
| Author graph (Table 2) | `legacy_author_graph`: authors linked if they **share a co-author**; largest component only | `coauthorship_author_graph`: direct co-authorship, authors with ≥ 2 papers |
| Homophily metric | `homophily_legacy`: edges touching a **mixed** author/paper are ignored | `homophily_paper_definition`: fraction of **all** edges that are safety–safety or ethics–ethics (Sec. 3.2.1) |

`table2_homophily.py --version legacy|paper-definition` switches all three together;
`--corpus` overrides the corpus on its own.

Changing the default of `load_combined_data` to `'full'` changes the output of every original
script that calls it. Pass `corpus='legacy'` to get the original behaviour.

## Results

| Paper result | Reproduces? | Script |
|---|---|---|
| Table 2 (homophily) | **Only with `legacy`**: baseline and 100-removed rows exact; author network within 0.1 pts at every level; paper network within 0.3 pts up to 300 removed, but off at 400 removed (73.3 / 63.2 / 50.9 vs 74.7 / 62.0 / 56.7) | `table2_homophily.py` |
| Table 7 (reachability within k hops) | **Yes, exactly**, including the k=5 shuffle null (21.5%) | `table7_reachability.py` |
| Figure 5 (weighted distances) | Yes: 80% of safety–safety pairs within 5.83 ("within 6.0") | `table7_reachability.py` |
| Table 3 (bridge concentration) | **No**: generating script not in the repo | `table3_bridges.py` (best-effort) |
| Table 1 (label split) | **No**: close, not exact | `table1_labels.py` |
| Table 6 (papers per venue after filtering) | Yes, with the `full` corpus: every venue and the 6,442 total match | `analyze_aggregated.load_combined_data` |
| Figure 2 (papers per year) | Not checked | |

### Table 2: why 83.1% and what the paper's definition gives

| Step | Author global homophily |
|---|---|
| Paper's definition: all papers, co-authorship, all edges | 22.9% |
| + ignore edges touching mixed authors | 100% (pure-safety and pure-ethics authors can never co-author) |
| + link authors who share a co-author | 82.0% (= `results_data/author_homophily_results.csv`) |
| + 3,248-paper corpus | **83.1%** (paper) |

Under the paper's definition the author network is 22.9% global / 40.6% safety / 34.9% ethics,
and the paper network 56.9 / 44.6 / 43.0. About half of the authors with ≥ 2 papers are mixed
(the paper's own Table 1 says 51%), and their edges make up 77% of the co-authorship graph.
Homophily is still above chance under label shuffling, so the clustering is real but much weaker
than "over four out of every five collaborations".

The legacy author graph also silently drops every ACL author. ACL names are "Last, First" and
DBLP names are "First Last", so the 1,173 ACL-only authors form a separate component that the
largest-component step discards.

### Table 3

No script in the repo produces Table 3 or `results_data/bridge_concentration_results.csv`. The
best-effort version gives 49.0 / 87.7 at baseline (paper 58.0 / 88.1). Other variants tried
(legacy graph or corpus, recomputed top sets, ranking by weighted degree) also don't match. If
the top sets are fixed on the baseline graph, removing at least as many top-degree authors as
the set contains forces the fraction to 0. The paper's 0.0% rows, including the nulls, match that
pattern.

### Table 1

| | Pure safety | Pure ethics | Mixed |
|---|---|---|---|
| Papers (here) | 3,109 | 2,866 | 467 |
| Papers (paper) | 2,874 | 2,959 | 609 |

662 FAccT/AIES/SaTML papers match no keyword. `load_combined_data` labels them by venue, which
contradicts the paper's statement that every paper matches a keyword. If the 93 SaTML ones are
labelled ethics (their `keyword_category`) instead of safety, pure ethics becomes exactly 2,959,
but 142 papers would still have to move from safety to mixed. `src/keywords.py` has 110 safety
and 99 ethics keywords; the paper says 114 and 102.

## Known bugs in the released analysis code (not used here, left unchanged)

- `network_asp_analysis.degree_preserving_nulls` calls `nx.double_edge_swap(..., n_swaps=...)`.
  The keyword is `nswap`, and the `TypeError` is swallowed, so the graph is never rewired.
- `network_asp_analysis.permutation_reachability_all` stores results only for the last
  permutation (indentation), so `run_permutation_diff_test` p-values are invalid.
- `unified_homophily_removal_analysis.py` labels its author-network null curve "Degree-Preserving
  Rewire Null", but plots the label-shuffle mean. Its rewiring does 1,000 swaps on a ~61k-edge graph.
- ACL/DBLP author names are never reconciled, so ~1,500 authors appear as two nodes. All results
  above, including the ones that match the paper, keep this behaviour.

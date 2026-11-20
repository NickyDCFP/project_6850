"""Table 2 homophily on time-boxed graphs: cumulative (2020, 2020-2021, ..., 2020-2025) or one year each.

  python -m analysis.homophily_over_time --version paper-definition
  python -m analysis.homophily_over_time --version legacy --levels 0 100
  python -m analysis.homophily_over_time --windows yearly

Each graph is built exactly as in table2_homophily.py (same --version / --corpus semantics), but
only from papers published up to the end year. Everything is recomputed per window: author labels
use only that window's papers, and "authors with >= 2 papers" means >= 2 papers in the window.
Early graphs are small: the legacy author graph keeps only its largest component, which in 2020 is
~20 authors, so its 2020 values (100% / nan) are not meaningful. Removal levels that would remove
most of a small early graph are still run but say little.

Also reports the Table 1 author split (pure safety / pure ethics / mixed among authors with >= 2
papers in the window, names normalized as in the networks).
"""
import argparse
import os

from reproduction.common import (E, M, S, coauthorship_author_graph, legacy_author_graph, load_papers,
                                 normalize_author, write_csv)
from reproduction.table1_labels import author_counts
from reproduction.table2_homophily import legacy, paper_definition

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', choices=['legacy', 'paper-definition'], default='paper-definition')
    ap.add_argument('--corpus', choices=['legacy', 'full'],
                    help='default: legacy for --version legacy, full for --version paper-definition')
    ap.add_argument('--windows', choices=['cumulative', 'yearly'], default='cumulative')
    ap.add_argument('--levels', type=int, nargs='+', default=[0], help='numbers of top-degree nodes removed')
    args = ap.parse_args()
    corpus = args.corpus or ('legacy' if args.version == 'legacy' else 'full')
    papers = load_papers(corpus)
    run = legacy if args.version == 'legacy' else paper_definition
    years = sorted({int(p['year']) for p in papers})

    print(f'{args.windows.capitalize()} homophily (%), version={args.version}, corpus={corpus} ({len(papers)} papers)')
    print(f'{"window":>9} {"papers":>6} {"authors":>7} {"edges":>6} {"removed":>7} | {"author: global safety ethics":>28} | {"paper: global safety ethics":>27}')
    rows, splits = [], []
    for end in years:
        start = end if args.windows == 'yearly' else years[0]
        window = [p for p in papers if start <= int(p['year']) <= end]
        label = f'{start}-{end}' if end > start else str(end)
        G = legacy_author_graph(window) if args.version == 'legacy' else coauthorship_author_graph(window)[0]
        n, m = G.number_of_nodes(), G.number_of_edges()
        c = author_counts(window, 2, normalize_author)
        split = [c[S], c[E], c[M]]
        splits.append((label, split))
        for level, a, p in run(window, levels=args.levels):
            fmt = lambda t: ' '.join(f'{x:6.1f}' for x in t)
            print(f'{label:>9} {len(window):>6} {n:>7} {m:>6} {level:>7} | {fmt(a):>28} | {fmt(p):>27}')
            rows.append([start, end, len(window), n, m, *split, level, *[round(x, 2) for x in a + p]])
    print(f'\nAuthors with >= 2 papers in the window (Table 1 split)')
    print(f'{"window":>9} {"total":>6} | {"safety":>12} {"ethics":>12} {"mixed":>12}')
    for label, split in splits:
        total = sum(split)
        print(f'{label:>9} {total:>6} | ' + ' '.join(f'{x:>5} ({100 * x / total:4.1f}%)' for x in split))
    suffix = '' if args.windows == 'cumulative' else '_yearly'
    path = write_csv(f'homophily_over_time_{args.version}_{corpus}{suffix}.csv',
                     ['start_year', 'end_year', 'papers', 'author_nodes', 'author_edges',
                      'authors_safety', 'authors_ethics', 'authors_mixed', 'removed', 'author_global', 'author_safety',
                      'author_ethics', 'paper_global', 'paper_safety', 'paper_ethics'], rows, RESULTS_DIR)
    print('wrote', path)


if __name__ == '__main__':
    main()

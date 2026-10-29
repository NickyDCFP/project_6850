"""Table 2: author- and paper-network homophily under degree-based removal of top nodes.

  python -m reproduction.table2_homophily --version legacy            # reproduces the paper's Table 2
  python -m reproduction.table2_homophily --version paper-definition  # what Sec 3.2 describes

legacy (unified_homophily_removal_analysis.py, run on the corpus the original loader saw):
  * corpus: 3,248 papers -- the original loader skipped the ACL and NeurIPS files
  * author edges: authors who share a co-author (not only co-authors), largest component only
  * homophily: edges touching a mixed author/paper are ignored; author network unweighted,
    paper network weighted by shared authors
paper-definition:
  * corpus: all 6,442 papers
  * author edges: direct co-authorship among authors with >= 2 papers; paper edges: shared author
  * homophily: fraction of all edges joining safety-safety or ethics-ethics (unweighted)
"""
import argparse

from reproduction.common import (coauthorship_author_graph, homophily_legacy, homophily_paper_definition,
                                 legacy_author_categories, legacy_author_graph, legacy_paper_graph,
                                 load_papers, paper_graph, remove_top_degree, write_csv)

LEVELS = (0, 100, 200, 300, 400)
PAPER = {  # Table 2 of the paper: (global, safety, ethics)
    'author': [(83.1, 73.5, 68.2), (85.8, 74.4, 75.9), (89.7, 80.2, 82.4), (90.5, 82.0, 83.2), (90.7, 83.5, 82.2)],
    'paper': [(71.2, 46.5, 61.6), (72.9, 49.2, 63.3), (73.5, 53.5, 61.9), (74.0, 57.6, 59.9), (74.7, 62.0, 56.7)],
}


def degree_order(G):
    return [n for n, _ in sorted(G.degree(), key=lambda t: -t[1])]


def legacy(papers):
    cats = legacy_author_categories(papers)
    author_seq = degree_order(legacy_author_graph(papers))
    paper_seq = degree_order(legacy_paper_graph(papers)[0])
    for level in LEVELS:
        G = legacy_author_graph(papers, removed=frozenset(author_seq[:level]))
        P, pcats = legacy_paper_graph(papers, removed=frozenset(paper_seq[:level]))
        yield level, homophily_legacy(G, cats), homophily_legacy(P, pcats, weighted=True)


def paper_definition(papers):
    G, cats = coauthorship_author_graph(papers)
    P, pcats = paper_graph(papers)
    for level in LEVELS:
        yield (level, homophily_paper_definition(remove_top_degree(G, level), cats),
               homophily_paper_definition(remove_top_degree(P, level), pcats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', choices=['legacy', 'paper-definition'], default='legacy')
    ap.add_argument('--corpus', choices=['legacy', 'full'],
                    help='default: legacy for --version legacy, full for --version paper-definition')
    args = ap.parse_args()
    corpus = args.corpus or ('legacy' if args.version == 'legacy' else 'full')
    papers = load_papers(corpus)
    run = legacy if args.version == 'legacy' else paper_definition

    print(f'Table 2 homophily (%), version={args.version}, corpus={corpus} ({len(papers)} papers)')
    print(f'{"removed":>7} | {"author: global safety ethics":>28} | {"paper: global safety ethics":>27} | paper Table 2 (author | paper)')
    rows = []
    for (level, a, p), pa, pp in zip(run(papers), PAPER['author'], PAPER['paper']):
        fmt = lambda t: ' '.join(f'{x:6.1f}' for x in t)
        print(f'{level:>7} | {fmt(a):>28} | {fmt(p):>27} | {fmt(pa)} | {fmt(pp)}')
        rows.append([level, *[round(x, 2) for x in a + p]])
    path = write_csv(f'table2_{args.version}_{corpus}.csv',
                     ['removed', 'author_global', 'author_safety', 'author_ethics',
                      'paper_global', 'paper_safety', 'paper_ethics'], rows)
    print('wrote', path)


if __name__ == '__main__':
    main()

"""Table 3: fraction of safety-ethics shortest paths passing through the top 1% / 5% of authors by degree.

  python -m reproduction.table3_bridges [--corpus full|legacy] [--pairs 1000]

Does NOT reproduce the paper: the script that produced Table 3 (and results_data/
bridge_concentration_results.csv) is not in the repository. This implements the procedure as
described, on the same author graph as Table 7: the top sets are fixed on the baseline graph, the
top-n authors by degree are removed, and for sampled reachable pure-safety/pure-ethics pairs we check
whether one shortest path passes through a top author.

Note: with a fixed top set, removing at least as many top-degree authors as the set contains
forces the fraction to exactly 0 (the top 1% is ~46 authors), so 0% rows say nothing about fragility.
"""
import argparse
import random

import networkx as nx
import numpy as np

from reproduction.common import E, S, coauthorship_author_graph, load_papers, write_csv

LEVELS = (0, 50, 100, 150, 200)
PAPER = [(58.0, 88.1), (13.2, 84.9), (0.0, 81.8), (0.0, 64.3), (0.0, 50.8)]


def bridge_fraction(G, cats, top, pairs, seed):
    rng = random.Random(seed)
    s_nodes = [n for n in G if cats.get(n) == S]
    e_nodes = [n for n in G if cats.get(n) == E]
    valid = through = tries = 0
    while valid < pairs and tries < 50 * pairs:
        tries += 1
        u, v = rng.choice(s_nodes), rng.choice(e_nodes)
        try:
            path = nx.shortest_path(G, u, v)
        except nx.NetworkXNoPath:
            continue
        valid += 1
        through += any(n in top for n in path[1:-1])
    return 100 * through / valid if valid else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', choices=['full', 'legacy'], default='full')
    ap.add_argument('--pairs', type=int, default=1000, help='reachable S-E pairs sampled per estimate')
    ap.add_argument('--shuffles', type=int, default=20)
    ap.add_argument('--rewires', type=int, default=5)
    args = ap.parse_args()

    G, cats = coauthorship_author_graph(load_papers(args.corpus))
    order = [n for n, _ in sorted(G.degree(), key=lambda t: -t[1])]
    n = G.number_of_nodes()
    top1, top5 = set(order[:n // 100]), set(order[:n // 20])

    print(f'Table 3 (corpus={args.corpus}): |top 1%|={len(top1)}, |top 5%|={len(top5)} of {n} authors')
    print(f'{"removed":>7} | {"top1%":>6} {"top5%":>6} | paper')
    rows = []
    for level, ref in zip(LEVELS, PAPER):
        H = G.copy()
        H.remove_nodes_from(order[:level])
        t1, t5 = bridge_fraction(H, cats, top1, args.pairs, 0), bridge_fraction(H, cats, top5, args.pairs, 0)
        print(f'{level:>7} | {t1:6.1f} {t5:6.1f} | {ref[0]:.1f} / {ref[1]:.1f}')
        rows.append([level, round(t1, 2), round(t5, 2)])

    nodes = list(G)
    labels = [cats.get(x) for x in nodes]
    shuffle = []
    for i in range(args.shuffles):
        random.Random(i).shuffle(labels)
        sc = dict(zip(nodes, labels))
        shuffle.append((bridge_fraction(G, sc, top1, args.pairs // 2, i), bridge_fraction(G, sc, top5, args.pairs // 2, i)))
    rewire = []
    for i in range(args.rewires):
        H = G.copy()
        nx.double_edge_swap(H, nswap=H.number_of_edges(), max_tries=10 * H.number_of_edges(), seed=i)
        rewire.append((bridge_fraction(H, cats, top1, args.pairs // 2, i), bridge_fraction(H, cats, top5, args.pairs // 2, i)))
    sh, rw = np.mean(shuffle, axis=0), np.mean(rewire, axis=0)
    print(f'baseline nulls: label shuffle {sh[0]:.1f} / {sh[1]:.1f} (paper 52.9 / 86.4); '
          f'degree-preserving rewire {rw[0]:.1f} / {rw[1]:.1f} (paper 38.3 / 77.0)')
    write_csv(f'table3_bridges_{args.corpus}.csv', ['removed', 'top1pct', 'top5pct'], rows)


if __name__ == '__main__':
    main()

"""Table 7 (author reachability within k hops, with label-shuffle null) and Figure 5 (weighted distances).

  python -m reproduction.table7_reachability [--perms 200]

Uses the direct co-authorship graph among authors with >= 2 papers on all 6,442 papers
(network_asp_analysis.build_author_coauthorship_network), which reproduces the paper exactly.
The released permutation helpers in network_asp_analysis are not used: permutation_reachability_all
keeps only the last permutation, and degree_preserving_nulls passes an invalid keyword to
double_edge_swap, so it never rewires.
"""
import argparse

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

from reproduction.common import E, S, coauthorship_author_graph, load_papers, write_csv

KS = (3, 5, 7)
PAPER = {3: (4.1, 2.3, 1.8), 5: (23.6, 15.0, 16.9), 7: (40.0, 29.5, 33.8)}  # SS, EE, SE (%)


def reachability(D, labels, ks=KS):
    """Percent of SS / EE / SE pairs within k hops, from an all-pairs hop-distance matrix."""
    s, e = np.where(labels == S)[0], np.where(labels == E)[0]
    out = {}
    for k in ks:
        R = (D <= k) & (D > 0)
        ss = R[np.ix_(s, s)].sum() / (len(s) * (len(s) - 1))
        ee = R[np.ix_(e, e)].sum() / (len(e) * (len(e) - 1))
        se = R[np.ix_(s, e)].sum() / (len(s) * len(e))
        out[k] = (100 * ss, 100 * ee, 100 * se)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--perms', type=int, default=200, help='label-shuffle permutations')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    G, cats = coauthorship_author_graph(load_papers('full'))
    nodes = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight')
    D = shortest_path(A, unweighted=True, directed=False)
    labels = np.array([cats[n] for n in nodes])
    obs = reachability(D, labels)

    rng = np.random.default_rng(args.seed)
    null = np.array([[reachability(D, rng.permutation(labels))[k][2] for k in KS] for _ in range(args.perms)])

    print(f'Table 7: {len(nodes)} authors ({(labels == S).sum()} pure safety, {(labels == E).sum()} pure ethics), '
          f'{args.perms} label shuffles')
    print(f'{"k":>2} | {"SS":>5} {"EE":>5} {"within":>6} {"SE":>5} {"delta":>6} | {"SE null":>7} {"p(SE<=null)":>11} | paper SS/EE/SE')
    rows = []
    for i, k in enumerate(KS):
        ss, ee, se = obs[k]
        within = (ss + ee) / 2
        p = (np.sum(null[:, i] <= se) + 1) / (args.perms + 1)
        print(f'{k:>2} | {ss:5.1f} {ee:5.1f} {within:6.1f} {se:5.1f} {se - within:+6.1f} | {null[:, i].mean():7.1f} {p:11.3f} | '
              + '/'.join(f'{x:.1f}' for x in PAPER[k]))
        rows.append([k, round(ss, 2), round(ee, 2), round(se, 2), round(se - within, 2), round(null[:, i].mean(), 2), round(p, 4)])
    write_csv('table7_reachability.csv', ['k', 'SS', 'EE', 'SE', 'delta', 'SE_null_mean', 'p_value'], rows)
    print('paper: SE null at k=5 = 21.5%')

    # Figure 5: weighted distance = 1 / (number of co-authored papers)
    A.data = 1.0 / A.data
    W = shortest_path(A, method='D', directed=False)
    s, e = np.where(labels == S)[0], np.where(labels == E)[0]
    ss = W[np.ix_(s, s)][np.triu_indices(len(s), 1)]
    ee = W[np.ix_(e, e)][np.triu_indices(len(e), 1)]
    se = W[np.ix_(s, e)].ravel()
    print('\nFigure 5: 80th percentile weighted distance among connected pairs (paper: SS within 6.0)')
    for name, d in (('SS', ss), ('EE', ee), ('SE', se)):
        d = d[np.isfinite(d)]
        print(f'  {name}: {np.percentile(d, 80):.2f}  (median {np.median(d):.2f}, mean {d.mean():.2f}, {len(d)} pairs)')


if __name__ == '__main__':
    main()

"""Run label-shuffle null model on the author graph (min_p=2).
Computes delta = SE_median_weighted_dist - mean(SS_median_weighted_dist, EE_median_weighted_dist)
for each permutation and saves null distribution CSV and histogram PNG.
"""
import os, sys, csv, random
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network
from scripts.weighted_homophily_multi_pct import sample_weighted_distances_with_hops
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)

random.seed(0)


def compute_observed_delta(papers, min_p=2):
    G, cats, info = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    ss = sample_weighted_distances_with_hops(G, cats, 'pure_safety', 'pure_safety')
    ee = sample_weighted_distances_with_hops(G, cats, 'pure_ethics', 'pure_ethics')
    se = sample_weighted_distances_with_hops(G, cats, 'pure_safety', 'pure_ethics')
    obs = {
        'ss_med': ss['median_dist'], 'ee_med': ee['median_dist'], 'se_med': se['median_dist'],
        'ss_pairs': ss['pairs_considered'], 'ee_pairs': ee['pairs_considered'], 'se_pairs': se['pairs_considered']
    }
    obs['delta'] = obs['se_med'] - ( (obs['ss_med'] + obs['ee_med']) / 2.0 )
    return obs, G, cats


def shuffle_labels_and_compute(papers, cats_orig, G, iterations=500, min_p=2):
    # cats_orig maps node->category; We need list of nodes and their original categories counts
    nodes = list(G.nodes())
    orig_cats = [cats_orig[n] for n in nodes]
    results = []
    for it in range(iterations):
        # permute category labels among nodes
        perm = orig_cats.copy()
        random.shuffle(perm)
        cats_perm = {n: perm[i] for i, n in enumerate(nodes)}
        ss = sample_weighted_distances_with_hops(G, cats_perm, 'pure_safety', 'pure_safety')
        ee = sample_weighted_distances_with_hops(G, cats_perm, 'pure_ethics', 'pure_ethics')
        se = sample_weighted_distances_with_hops(G, cats_perm, 'pure_safety', 'pure_ethics')
        if ss['median_dist'] is None or ee['median_dist'] is None or se['median_dist'] is None:
            delta = None
        else:
            delta = se['median_dist'] - ((ss['median_dist'] + ee['median_dist']) / 2.0)
        results.append({'iter': it, 'ss_med': ss['median_dist'], 'ee_med': ee['median_dist'], 'se_med': se['median_dist'], 'delta': delta})
        if (it+1) % 50 == 0:
            print('Completed', it+1, 'permutations')
    return results


def main():
    papers = load_combined_data()
    obs, G, cats = compute_observed_delta(papers, min_p=2)
    print('Observed medians: SS', obs['ss_med'], 'EE', obs['ee_med'], 'SE', obs['se_med'], 'delta', obs['delta'])
    # run permutations
    per = 500
    nulls = shuffle_labels_and_compute(papers, cats, G, iterations=per, min_p=2)
    # filter out None deltas
    deltas = [r['delta'] for r in nulls if r['delta'] is not None]
    outcsv = os.path.join(OUT, 'label_shuffle_nulls_author_minp2.csv')
    with open(outcsv,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['iter','ss_med','ee_med','se_med','delta'])
        w.writeheader()
        for r in nulls:
            w.writerow(r)
    print('Wrote null CSV', outcsv)
    # p-value (one-sided: how many null deltas >= observed delta)
    import math
    if obs['delta'] is None:
        print('Observed delta is None; abort')
        return
    count_ge = sum(1 for v in deltas if v is not None and v >= obs['delta'])
    pval = (count_ge + 1) / (len(deltas) + 1)
    mean_null = np.mean(deltas)
    std_null = np.std(deltas)
    z = (obs['delta'] - mean_null) / std_null if std_null > 0 else float('inf')
    print('Null mean delta:', mean_null, 'std:', std_null, 'observed delta:', obs['delta'], 'pval(one-sided):', pval, 'z:', z)
    # histogram
    plt.figure(figsize=(6,4))
    plt.hist(deltas, bins=30, alpha=0.7)
    plt.axvline(obs['delta'], color='red', linestyle='--', label=f'obs delta={obs["delta"]:.3f}')
    plt.xlabel('Delta (SE_med - mean(SS_med,EE_med))')
    plt.ylabel('Count')
    plt.title('Label-shuffle null distribution (author graph, min_p=2)')
    plt.legend()
    png = os.path.join(OUT, 'label_shuffle_nulls_author_minp2_hist.png')
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    print('Wrote histogram', png)

if __name__ == '__main__':
    main()

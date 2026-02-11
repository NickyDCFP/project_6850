"""Compute weighted homophily and purity measures on the author coauthorship graph.

Weights = number of coauthored papers between author pairs.
Saves a CSV to perturb_grid_out/weighted_homophily_minp{min_p}.csv and prints a short summary.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv, time, random
from statistics import mean, median
import networkx as nx
from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)
random.seed(0)


def weighted_degree(G, n):
    return sum((edata.get('weight', 1) for _,_,edata in G.edges(n, data=True)))


def global_weighted_mixing(G, categories):
    mix = {}
    total = 0.0
    for u,v,data in G.edges(data=True):
        w = float(data.get('weight', 1))
        cu = categories.get(u, 'neither')
        cv = categories.get(v, 'neither')
        key = tuple(sorted([cu, cv]))
        mix[key] = mix.get(key, 0.0) + w
        total += w
    # convert to fractions
    frac = {k: (v/total if total else 0.0) for k,v in mix.items()}
    return mix, frac, total


def avg_weighted_clustering(G):
    # networkx supports weight argument for clustering
    try:
        clust = nx.clustering(G, weight='weight')
        vals = list(clust.values())
        return mean(vals) if vals else 0.0
    except Exception:
        return None


def per_category_purity(G, categories):
    cats = set(categories.values())
    results = {}
    # Precompute edge list weights
    internal_weights = {c: 0.0 for c in cats}
    incident_weights = {c: 0.0 for c in cats}
    for u,v,data in G.edges(data=True):
        w = float(data.get('weight', 1))
        cu = categories.get(u, 'neither')
        cv = categories.get(v, 'neither')
        # if both in same category, count internal weight once
        if cu == cv:
            internal_weights[cu] = internal_weights.get(cu, 0.0) + w
        # incident weight counts per node (we will sum deg weights per node)
        incident_weights[cu] = incident_weights.get(cu, 0.0) + w
        incident_weights[cv] = incident_weights.get(cv, 0.0) + w
    for c in cats:
        inc = incident_weights.get(c, 0.0)
        internal = internal_weights.get(c, 0.0)
        frac_internal = (2.0 * internal / inc) if inc > 0 else None
        results[c] = {
            'internal_edge_weight': internal,
            'incident_weight_sum': inc,
            'fraction_internal_incident': frac_internal
        }
    return results


def sample_weighted_distances(G, categories, cat_a, cat_b, sample_target=500, max_pairs=200000):
    # Use distance = 1/weight for edges; copy subgraph largest connected component
    if not nx.is_connected(G):
        H_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(H_nodes).copy()
    else:
        H = G
    # prepare distance attribute
    for u,v,data in H.edges(data=True):
        w = data.get('weight', 1.0)
        data['dist'] = 1.0 / w if w > 0 else float('inf')

    A = [n for n,c in categories.items() if c == cat_a and n in H]
    B = [n for n,c in categories.items() if c == cat_b and n in H]
    if not A or not B:
        return {'pairs_considered':0,'reachable':0,'reachable_frac':0.0,'mean_dist':None,'median_dist':None}

    total_possible = len(A) * len(B)
    pairs = []
    if total_possible <= max_pairs:
        for a in A:
            for b in B:
                pairs.append((a,b))
    else:
        seen=set(); attempts=0
        while len(pairs) < max_pairs and attempts < max_pairs*20:
            a=random.choice(A); b=random.choice(B)
            if (a,b) in seen:
                attempts+=1; continue
            seen.add((a,b)); pairs.append((a,b))
    dists = []
    reachable = 0
    pairs_considered = 0
    for u,v in pairs:
        pairs_considered += 1
        try:
            # Dijkstra shortest path length using attribute 'dist'
            d = nx.shortest_path_length(H, u, v, weight='dist')
            dists.append(d)
            reachable += 1
        except nx.NetworkXNoPath:
            continue
        if reachable >= sample_target:
            break
    mean_d = mean(dists) if dists else None
    med_d = median(dists) if dists else None
    return {'pairs_considered':pairs_considered,'reachable':reachable,'reachable_frac': reachable / pairs_considered if pairs_considered else 0.0,'mean_dist':mean_d,'median_dist':med_d}


def run(min_p=2):
    papers = load_combined_data()
    G, cats, info = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    print('Authors (min_p=%d):' % min_p, G.number_of_nodes())

    mix_counts, mix_fracs, total_w = global_weighted_mixing(G, cats)
    weighted_homophily = sum(v for k,v in mix_counts.items() if k[0] == k[1]) / total_w if total_w else 0.0
    avg_w_clust = avg_weighted_clustering(G)
    per_cat = per_category_purity(G, cats)

    # sample weighted ASPs
    ss_stats = sample_weighted_distances(G, cats, 'pure_safety', 'pure_safety', sample_target=500)
    ee_stats = sample_weighted_distances(G, cats, 'pure_ethics', 'pure_ethics', sample_target=500)
    se_stats = sample_weighted_distances(G, cats, 'pure_safety', 'pure_ethics', sample_target=500)

    # save CSV
    outcsv = os.path.join(OUT, f'weighted_homophily_minp{min_p}.csv')
    with open(outcsv,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['total_edge_weight', total_w])
        w.writerow(['weighted_homophily_fraction', weighted_homophily])
        w.writerow(['avg_weighted_clustering', avg_w_clust])
        # write mixing fractions
        for k,v in mix_fracs.items():
            w.writerow([f'mix_frac_{k[0]}__{k[1]}', v])
        # per-category
        for c,vals in per_cat.items():
            w.writerow([f'cat_{c}_internal_edge_weight', vals['internal_edge_weight']])
            w.writerow([f'cat_{c}_incident_weight_sum', vals['incident_weight_sum']])
            w.writerow([f'cat_{c}_fraction_internal_incident', vals['fraction_internal_incident']])
        # weighted ASPs
        w.writerow(['ss_pairs_considered', ss_stats['pairs_considered']])
        w.writerow(['ss_reachable', ss_stats['reachable']])
        w.writerow(['ss_mean_dist', ss_stats['mean_dist']])
        w.writerow(['ss_median_dist', ss_stats['median_dist']])

        w.writerow(['ee_pairs_considered', ee_stats['pairs_considered']])
        w.writerow(['ee_reachable', ee_stats['reachable']])
        w.writerow(['ee_mean_dist', ee_stats['mean_dist']])
        w.writerow(['ee_median_dist', ee_stats['median_dist']])

        w.writerow(['se_pairs_considered', se_stats['pairs_considered']])
        w.writerow(['se_reachable', se_stats['reachable']])
        w.writerow(['se_mean_dist', se_stats['mean_dist']])
        w.writerow(['se_median_dist', se_stats['median_dist']])

    print('Wrote', outcsv)
    # print short summary
    print('\nSUMMARY (weighted):')
    print('  total edge weight:', total_w)
    print('  weighted homophily (fraction of weight internal same-category):', weighted_homophily)
    print('  avg weighted clustering:', avg_w_clust)
    for c,vals in per_cat.items():
        print(f"  category {c}: internal_weight={vals['internal_edge_weight']}, incident_sum={vals['incident_weight_sum']}, frac_internal_incident={vals['fraction_internal_incident']}")
    print('\nWeighted ASP samples (distance = 1/weight):')
    print('  SS mean/median:', ss_stats['mean_dist'], ss_stats['median_dist'], 'reachable%', ss_stats['reachable_frac'])
    print('  EE mean/median:', ee_stats['mean_dist'], ee_stats['median_dist'], 'reachable%', ee_stats['reachable_frac'])
    print('  SE mean/median:', se_stats['mean_dist'], se_stats['median_dist'], 'reachable%', se_stats['reachable_frac'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_p', type=int, default=2)
    args = parser.parse_args()
    start=time.time(); run(min_p=args.min_p); print('Done in', time.time()-start)

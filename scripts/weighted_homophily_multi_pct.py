"""Run weighted homophily and weighted-ASP for multiple top-% removal levels (author and paper graphs).

Removals are drawn from perturb_grid_out/greedy_attack_sequence_minp2.csv (greedy ranked authors).
Produces per-% CSVs and a summary CSV at perturb_grid_out/weighted_homophily_filtered_minp2_summary.csv
"""
import os, sys, csv, random, math
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network, build_paper_coauthorship_network

import networkx as nx
from statistics import mean, median

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)
random.seed(0)


def remove_authors_from_papers(papers, remove_set):
    new = []
    for p in papers:
        new_auths = []
        for a in p.get('authors', []):
            name = a if isinstance(a, str) else (a.get('name') if isinstance(a, dict) else a)
            if name not in remove_set:
                new_auths.append(a)
        if new_auths:
            np = dict(p); np['authors'] = new_auths; new.append(np)
    return new


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
    frac = {k: (v/total if total else 0.0) for k,v in mix.items()}
    return mix, frac, total


def avg_weighted_clustering(G):
    try:
        clust = nx.clustering(G, weight='weight')
        vals = list(clust.values())
        return mean(vals) if vals else 0.0
    except Exception:
        return None


def per_category_purity(G, categories):
    cats = set(categories.values())
    internal_weights = {c: 0.0 for c in cats}
    incident_weights = {c: 0.0 for c in cats}
    for u,v,data in G.edges(data=True):
        w = float(data.get('weight', 1))
        cu = categories.get(u, 'neither')
        cv = categories.get(v, 'neither')
        if cu == cv:
            internal_weights[cu] = internal_weights.get(cu, 0.0) + w
        incident_weights[cu] = incident_weights.get(cu, 0.0) + w
        incident_weights[cv] = incident_weights.get(cv, 0.0) + w
    results = {}
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


def sample_weighted_distances_with_hops(G, categories, cat_a, cat_b, sample_target=500, max_pairs=200000):
    # work on largest cc for connectivity
    if not nx.is_connected(G):
        H_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(H_nodes).copy()
    else:
        H = G
    # prepare distance attribute (1/weight)
    for u,v,data in H.edges(data=True):
        w = data.get('weight', 1.0)
        data['dist'] = 1.0 / w if w > 0 else float('inf')
    A = [n for n,c in categories.items() if c == cat_a and n in H]
    B = [n for n,c in categories.items() if c == cat_b and n in H]
    if not A or not B:
        return {'pairs_considered':0,'reachable':0,'reachable_frac':0.0,'mean_dist':None,'median_dist':None,'mean_hops':None,'median_hops':None}
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
    w_dists = []
    hops = []
    reachable = 0
    pairs_considered = 0
    for u,v in pairs:
        pairs_considered += 1
        try:
            d = nx.shortest_path_length(H, u, v, weight='dist')
            h = nx.shortest_path_length(H, u, v)
            w_dists.append(d)
            hops.append(h)
            reachable += 1
        except nx.NetworkXNoPath:
            continue
        if reachable >= sample_target:
            break
    mean_w = mean(w_dists) if w_dists else None
    med_w = median(w_dists) if w_dists else None
    mean_h = mean(hops) if hops else None
    med_h = median(hops) if hops else None
    return {'pairs_considered':pairs_considered,'reachable':reachable,'reachable_frac': reachable / pairs_considered if pairs_considered else 0.0,'mean_dist':mean_w,'median_dist':med_w,'mean_hops':mean_h,'median_hops':med_h}


def analyze_for_removal(papers, remove_set, min_p=2):
    filtered_papers = remove_authors_from_papers(papers, remove_set)
    # Author graph
    G_auth, cats_auth, info_auth = build_author_coauthorship_network(filtered_papers, min_papers_per_author=min_p)
    mix_counts_a, mix_fracs_a, total_w_a = global_weighted_mixing(G_auth, cats_auth)
    weighted_homophily_a = sum(v for k,v in mix_counts_a.items() if k[0] == k[1]) / total_w_a if total_w_a else 0.0
    avg_w_clust_a = avg_weighted_clustering(G_auth)
    per_cat_a = per_category_purity(G_auth, cats_auth)
    ss_a = sample_weighted_distances_with_hops(G_auth, cats_auth, 'pure_safety', 'pure_safety')
    ee_a = sample_weighted_distances_with_hops(G_auth, cats_auth, 'pure_ethics', 'pure_ethics')
    se_a = sample_weighted_distances_with_hops(G_auth, cats_auth, 'pure_safety', 'pure_ethics')

    # Paper graph (rebuild from filtered papers)
    G_paper, cats_paper, info_paper = build_paper_coauthorship_network(filtered_papers, min_papers_per_author=min_p)
    mix_counts_p, mix_fracs_p, total_w_p = global_weighted_mixing(G_paper, cats_paper)
    weighted_homophily_p = sum(v for k,v in mix_counts_p.items() if k[0] == k[1]) / total_w_p if total_w_p else 0.0
    avg_w_clust_p = avg_weighted_clustering(G_paper)
    per_cat_p = per_category_purity(G_paper, cats_paper)
    ss_p = sample_weighted_distances_with_hops(G_paper, cats_paper, 'pure_safety', 'pure_safety')
    ee_p = sample_weighted_distances_with_hops(G_paper, cats_paper, 'pure_ethics', 'pure_ethics')
    se_p = sample_weighted_distances_with_hops(G_paper, cats_paper, 'pure_safety', 'pure_ethics')

    return {
        'author':{
            'G':G_auth,'categories':cats_auth,'total_weight':total_w_a,'weighted_homophily':weighted_homophily_a,'avg_clust':avg_w_clust_a,'per_cat':per_cat_a,'ss':ss_a,'ee':ee_a,'se':se_a
        },
        'paper':{
            'G':G_paper,'categories':cats_paper,'total_weight':total_w_p,'weighted_homophily':weighted_homophily_p,'avg_clust':avg_w_clust_p,'per_cat':per_cat_p,'ss':ss_p,'ee':ee_p,'se':se_p
        },
        'filtered_papers_count': len(filtered_papers)
    }


def main():
    papers = load_combined_data()
    greedy_csv = os.path.join(OUT, 'greedy_attack_sequence_minp2.csv')
    if not os.path.exists(greedy_csv):
        print('Missing greedy sequence CSV:', greedy_csv)
        return
    removals = []
    with open(greedy_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            removals.append(row['removed_node'])
    # Build baseline author graph to determine total authors at min_p
    G_base, cats_base, info_base = build_author_coauthorship_network(papers, min_papers_per_author=2)
    total_authors = G_base.number_of_nodes()
    print('Baseline authors (min_p=2):', total_authors)

    fractions = [(0.01, 'top1pct'), (0.05, 'top5pct'), (0.10, 'top10pct')]
    summary_rows = []
    for frac, label in fractions:
        top_k = max(1, int(round(total_authors * frac)))
        to_remove = set(removals[:top_k])
        print(f'Analyzing removal {label}: removing {top_k} authors ({frac*100}%)')
        results = analyze_for_removal(papers, to_remove, min_p=2)
        # write per-label CSVs
        out_a = os.path.join(OUT, f'weighted_homophily_filtered_minp2_{label}.csv')
        with open(out_a,'w',newline='',encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['metric','value'])
            w.writerow(['removed_authors_count', top_k])
            w.writerow(['filtered_papers_count', results['filtered_papers_count']])
            w.writerow(['author_total_edge_weight', results['author']['total_weight']])
            w.writerow(['author_weighted_homophily', results['author']['weighted_homophily']])
            w.writerow(['author_avg_weighted_clustering', results['author']['avg_clust']])
            w.writerow(['author_ss_median_weighted_dist', results['author']['ss']['median_dist']])
            w.writerow(['author_ss_median_hops', results['author']['ss']['median_hops']])
            w.writerow(['author_ee_median_weighted_dist', results['author']['ee']['median_dist']])
            w.writerow(['author_ee_median_hops', results['author']['ee']['median_hops']])
            w.writerow(['author_se_median_weighted_dist', results['author']['se']['median_dist']])
            w.writerow(['author_se_median_hops', results['author']['se']['median_hops']])
        out_p = os.path.join(OUT, f'weighted_homophily_filtered_papers_minp2_{label}.csv')
        with open(out_p,'w',newline='',encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['metric','value'])
            w.writerow(['removed_authors_count', top_k])
            w.writerow(['filtered_papers_count', results['filtered_papers_count']])
            w.writerow(['paper_total_edge_weight', results['paper']['total_weight']])
            w.writerow(['paper_weighted_homophily', results['paper']['weighted_homophily']])
            w.writerow(['paper_avg_weighted_clustering', results['paper']['avg_clust']])
            w.writerow(['paper_ss_median_weighted_dist', results['paper']['ss']['median_dist']])
            w.writerow(['paper_ss_median_hops', results['paper']['ss']['median_hops']])
            w.writerow(['paper_ee_median_weighted_dist', results['paper']['ee']['median_dist']])
            w.writerow(['paper_ee_median_hops', results['paper']['ee']['median_hops']])
            w.writerow(['paper_se_median_weighted_dist', results['paper']['se']['median_dist']])
            w.writerow(['paper_se_median_hops', results['paper']['se']['median_hops']])
        summary_rows.append({
            'label':label,'frac':frac,'removed':top_k,
            'author_weighted_homophily':results['author']['weighted_homophily'],'author_ss_median_weighted_dist':results['author']['ss']['median_dist'],'author_ee_median_weighted_dist':results['author']['ee']['median_dist'],'author_se_median_weighted_dist':results['author']['se']['median_dist'],'author_ss_median_hops':results['author']['ss']['median_hops'],'author_ee_median_hops':results['author']['ee']['median_hops'],'author_se_median_hops':results['author']['se']['median_hops'],
            'paper_weighted_homophily':results['paper']['weighted_homophily'],'paper_ss_median_weighted_dist':results['paper']['ss']['median_dist'],'paper_ee_median_weighted_dist':results['paper']['ee']['median_dist'],'paper_se_median_weighted_dist':results['paper']['se']['median_dist'],'paper_ss_median_hops':results['paper']['ss']['median_hops'],'paper_ee_median_hops':results['paper']['ee']['median_hops'],'paper_se_median_hops':results['paper']['se']['median_hops']
        })
    # write summary CSV
    summary_csv = os.path.join(OUT, 'weighted_homophily_filtered_minp2_summary.csv')
    with open(summary_csv,'w',newline='',encoding='utf-8') as f:
        fieldnames = ['label','frac','removed','author_weighted_homophily','author_ss_median_weighted_dist','author_ee_median_weighted_dist','author_se_median_weighted_dist','author_ss_median_hops','author_ee_median_hops','author_se_median_hops','paper_weighted_homophily','paper_ss_median_weighted_dist','paper_ee_median_weighted_dist','paper_se_median_weighted_dist','paper_ss_median_hops','paper_ee_median_hops','paper_se_median_hops']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print('Wrote summary:', summary_csv)
    print('Per-label CSVs written to', OUT)

if __name__ == '__main__':
    main()

"""Run random-removal null experiments for top-1/5/10% greedy removals.
Saves CSV summary and per-fraction null arrays.
"""
import os, sys, csv, random, math
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network, build_paper_coauthorship_network, normalize_author
import networkx as nx
from statistics import mean, median, pstdev

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)
random.seed(1)


def remove_authors_from_papers(papers, remove_set):
    new = []
    for p in papers:
        new_auths = []
        for a in p.get('authors', []):
            name = a if isinstance(a, str) else (a.get('name') if isinstance(a, dict) else a)
            n = normalize_author(name) if name else name
            if n not in remove_set:
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


def sample_weighted_distances_with_hops(G, categories, cat_a, cat_b, sample_target=500, max_pairs=200000):
    if not nx.is_connected(G):
        H_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(H_nodes).copy()
    else:
        H = G
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


def compute_metrics_for_papers(papers, min_p=2):
    G_auth, cats_auth, info_auth = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    mix_counts_a, mix_fracs_a, total_w_a = global_weighted_mixing(G_auth, cats_auth)
    weighted_homophily_a = sum(v for k,v in mix_counts_a.items() if k[0] == k[1]) / total_w_a if total_w_a else 0.0
    ss_a = sample_weighted_distances_with_hops(G_auth, cats_auth, 'pure_safety', 'pure_safety')
    ee_a = sample_weighted_distances_with_hops(G_auth, cats_auth, 'pure_ethics', 'pure_ethics')
    se_a = sample_weighted_distances_with_hops(G_auth, cats_auth, 'pure_safety', 'pure_ethics')

    # paper graph
    G_paper, cats_paper, info_paper = build_paper_coauthorship_network(papers, min_papers_per_author=min_p)
    mix_counts_p, mix_fracs_p, total_w_p = global_weighted_mixing(G_paper, cats_paper)
    weighted_homophily_p = sum(v for k,v in mix_counts_p.items() if k[0] == k[1]) / total_w_p if total_w_p else 0.0
    ss_p = sample_weighted_distances_with_hops(G_paper, cats_paper, 'pure_safety', 'pure_safety')
    ee_p = sample_weighted_distances_with_hops(G_paper, cats_paper, 'pure_ethics', 'pure_ethics')
    se_p = sample_weighted_distances_with_hops(G_paper, cats_paper, 'pure_safety', 'pure_ethics')

    return {
        'author':{
            'weighted_homophily': weighted_homophily_a,
            'ss_median_w': ss_a['median_dist'],'ee_median_w': ee_a['median_dist'],'se_median_w': se_a['median_dist'],
            'ss_median_hops': ss_a['median_hops'],'ee_median_hops': ee_a['median_hops'],'se_median_hops': se_a['median_hops']
        },
        'paper':{
            'weighted_homophily': weighted_homophily_p,
            'ss_median_w': ss_p['median_dist'],'ee_median_w': ee_p['median_dist'],'se_median_w': se_p['median_dist'],
            'ss_median_hops': ss_p['median_hops'],'ee_median_hops': ee_p['median_hops'],'se_median_hops': se_p['median_hops']
        }
    }


def main(reps=50):
    papers = load_combined_data()
    greedy_csv = os.path.join(OUT, 'greedy_attack_sequence_minp2.csv')
    if not os.path.exists(greedy_csv):
        print('Missing greedy sequence CSV:', greedy_csv); return
    removals = []
    with open(greedy_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            removals.append(row['removed_node'])
    # baseline authors
    G_base, cats_base, info_base = build_author_coauthorship_network(papers, min_papers_per_author=2)
    total_authors = G_base.number_of_nodes()
    all_authors_list = list(G_base.nodes())
    fractions = [(0.01,'top1pct'),(0.05,'top5pct'),(0.10,'top10pct')]
    summary_rows = []
    for frac,label in fractions:
        k = max(1, int(round(total_authors * frac)))
        print(f'Running nulls for {label}: k={k}, reps={reps}')
        # observed
        to_remove = set(removals[:k])
        filtered_obs = remove_authors_from_papers(papers, to_remove)
        obs_metrics = compute_metrics_for_papers(filtered_obs, min_p=2)
        # run random reps
        null_hom_auth = []
        null_se_w_auth = []
        null_se_hops_auth = []
        null_hom_paper = []
        null_se_w_paper = []
        null_se_hops_paper = []
        for i in range(reps):
            samp = set(random.sample(all_authors_list, k))
            filtered = remove_authors_from_papers(papers, samp)
            m = compute_metrics_for_papers(filtered, min_p=2)
            null_hom_auth.append(m['author']['weighted_homophily'])
            null_se_w_auth.append(m['author']['se_median_w'] if m['author']['se_median_w'] is not None else float('nan'))
            null_se_hops_auth.append(m['author']['se_median_hops'] if m['author']['se_median_hops'] is not None else float('nan'))
            null_hom_paper.append(m['paper']['weighted_homophily'])
            null_se_w_paper.append(m['paper']['se_median_w'] if m['paper']['se_median_w'] is not None else float('nan'))
            null_se_hops_paper.append(m['paper']['se_median_hops'] if m['paper']['se_median_hops'] is not None else float('nan'))
            if (i+1) % 50 == 0:
                print('  completed', i+1, 'reps')
        # compute p-values (one-sided: homophily increase, SE increase meaning longer)
        def emp_p(obs, arr, direction='greater'):
            arr = [x for x in arr if x==x]
            if not arr:
                return None
            if direction=='greater':
                return (sum(1 for x in arr if x >= obs) + 1) / (len(arr) + 1)
            else:
                return (sum(1 for x in arr if x <= obs) + 1) / (len(arr) + 1)
        p_h_auth = emp_p(obs_metrics['author']['weighted_homophily'], null_hom_auth, 'greater')
        p_se_w_auth = emp_p(obs_metrics['author']['se_median_w'], null_se_w_auth, 'greater')
        p_se_hops_auth = emp_p(obs_metrics['author']['se_median_hops'], null_se_hops_auth, 'greater')
        p_h_paper = emp_p(obs_metrics['paper']['weighted_homophily'], null_hom_paper, 'greater')
        p_se_w_paper = emp_p(obs_metrics['paper']['se_median_w'], null_se_w_paper, 'greater')
        p_se_hops_paper = emp_p(obs_metrics['paper']['se_median_hops'], null_se_hops_paper, 'greater')
        row = {
            'label':label,'frac':frac,'k':k,
            'obs_auth_hom':obs_metrics['author']['weighted_homophily'],'null_auth_hom_mean':mean(null_hom_auth),'null_auth_hom_std':pstdev(null_hom_auth),'p_auth_hom':p_h_auth,
            'obs_auth_se_median_w':obs_metrics['author']['se_median_w'],'null_auth_se_w_mean':mean([x for x in null_se_w_auth if x==x]),'null_auth_se_w_std':pstdev([x for x in null_se_w_auth if x==x]),'p_auth_se_w':p_se_w_auth,
            'obs_auth_se_median_hops':obs_metrics['author']['se_median_hops'],'null_auth_se_hops_mean':mean([x for x in null_se_hops_auth if x==x]),'null_auth_se_hops_std':pstdev([x for x in null_se_hops_auth if x==x]),'p_auth_se_hops':p_se_hops_auth,
            'obs_paper_hom':obs_metrics['paper']['weighted_homophily'],'null_paper_hom_mean':mean(null_hom_paper),'null_paper_hom_std':pstdev(null_hom_paper),'p_paper_hom':p_h_paper,
            'obs_paper_se_median_w':obs_metrics['paper']['se_median_w'],'null_paper_se_w_mean':mean([x for x in null_se_w_paper if x==x]),'null_paper_se_w_std':pstdev([x for x in null_se_w_paper if x==x]),'p_paper_se_w':p_se_w_paper,
            'obs_paper_se_median_hops':obs_metrics['paper']['se_median_hops'],'null_paper_se_hops_mean':mean([x for x in null_se_hops_paper if x==x]),'null_paper_se_hops_std':pstdev([x for x in null_se_hops_paper if x==x]),'p_paper_se_hops':p_se_hops_paper
        }
        summary_rows.append(row)
        # save per-fraction null arrays for inspection
        def save_arr(arr, name):
            outf = os.path.join(OUT, f'{name}_{label}.csv')
            with open(outf,'w',newline='',encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['value'])
                for v in arr:
                    w.writerow([v])
        save_arr(null_hom_auth, 'null_auth_hom')
        save_arr(null_se_w_auth, 'null_auth_se_w')
        save_arr(null_se_hops_auth, 'null_auth_se_hops')
        save_arr(null_hom_paper, 'null_paper_hom')
        save_arr(null_se_w_paper, 'null_paper_se_w')
        save_arr(null_se_hops_paper, 'null_paper_se_hops')

    # write summary CSV
    out = os.path.join(OUT, f'random_removal_nulls_summary_minp2_reps{reps}.csv')
    with open(out,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)
    print('Wrote summary CSV', out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=50, help='Number of random removal repetitions')
    args = parser.parse_args()
    main(reps=args.reps)

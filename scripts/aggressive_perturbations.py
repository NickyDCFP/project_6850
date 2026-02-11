"""Aggressive perturbations to maximize S/E separation.

Run: python3 scripts/aggressive_perturbations.py
Saves: perturb_grid_out/aggressive_results.csv and plots.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time, random, csv
from collections import Counter
import networkx as nx
from statistics import mean

from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network, build_paper_coauthorship_network, reachability_by_hops

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)
random.seed(1)

def remove_authors_from_papers(papers, remove_set):
    new = []
    for p in papers:
        new_auths = []
        for a in p.get('authors', []):
            name = a.get('name') if isinstance(a, dict) else a
            if name not in remove_set:
                new_auths.append(a)
        if new_auths:
            np = dict(p); np['authors'] = new_auths; new.append(np)
    return new

def most_frequent_internal_authors(G, cats, sample_pairs=20000, sample_target=5000):
    S = [n for n,c in cats.items() if c=='pure_safety' and n in G]
    E = [n for n,c in cats.items() if c=='pure_ethics' and n in G]
    counts = Counter(); valid=0; attempts=0
    while valid < sample_target and attempts < sample_pairs:
        u = random.choice(S); v = random.choice(E)
        attempts += 1
        try:
            path = nx.shortest_path(G, u, v)
        except nx.NetworkXNoPath:
            continue
        internal = path[1:-1]
        for a in internal:
            counts[a]+=1
        valid+=1
    return counts.most_common()

def apply_rewire(G, frac):
    H = G.copy()
    m = H.number_of_edges()
    swaps = max(1, int(frac * m))
    try:
        nx.double_edge_swap(H, nswap=swaps, max_tries=swaps*10)
    except Exception:
        pass
    return H

def run_once(min_p):
    papers = load_combined_data()
    G_auth, cats_auth, info_auth = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    baseline = reachability_by_hops(G_auth, cats_auth, max_k=6)
    baseline_se = [baseline['se_frac_by_k'][k] for k in range(3,6)]
    baseline_mean = mean(baseline_se)

    N = G_auth.number_of_nodes(); M = G_auth.number_of_edges()
    deg_sorted = sorted(G_auth.degree(), key=lambda x: -x[1])
    top_by_deg = [n for n,_ in deg_sorted]

    results = []

    # large removals by percent
    for pct in [0.005, 0.01, 0.05, 0.10]:
        k = max(1, int(pct * N))
        to_remove = set(top_by_deg[:k])
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        r = reachability_by_hops(G_r, cats_r, max_k=6)
        r_mean = mean([r['se_frac_by_k'][k] for k in [3,4,5]])
        results.append({'perturb':'remove_pct_deg','param':pct,'removed':k,'se_mean_before':baseline_mean,'se_mean_after':r_mean,'delta':baseline_mean - r_mean})

    # fixed removals
    for k in [200,500,1000]:
        to_remove = set(top_by_deg[:min(k, len(top_by_deg))])
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        r = reachability_by_hops(G_r, cats_r, max_k=6)
        r_mean = mean([r['se_frac_by_k'][k] for k in [3,4,5]])
        results.append({'perturb':'remove_fixed_deg','param':k,'removed':len(to_remove),'se_mean_before':baseline_mean,'se_mean_after':r_mean,'delta':baseline_mean - r_mean})

    # remove most frequent internal authors (bridge nodes)
    freq = most_frequent_internal_authors(G_auth, cats_auth, sample_pairs=50000, sample_target=5000)
    most_common = [n for n,_ in freq]
    for k in [10,50,100]:
        to_remove = set(most_common[:k])
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        r = reachability_by_hops(G_r, cats_r, max_k=6)
        r_mean = mean([r['se_frac_by_k'][k] for k in [3,4,5]])
        results.append({'perturb':'remove_bridge_freq','param':k,'removed':len(to_remove),'se_mean_before':baseline_mean,'se_mean_after':r_mean,'delta':baseline_mean - r_mean})

    # big rewire fractions
    for frac in [0.05, 0.10, 0.20]:
        H = apply_rewire(G_auth, frac)
        r = reachability_by_hops(H, cats_auth, max_k=6)
        r_mean = mean([r['se_frac_by_k'][k] for k in [3,4,5]])
        results.append({'perturb':'rewire_frac','param':frac,'removed':0,'se_mean_before':baseline_mean,'se_mean_after':r_mean,'delta':baseline_mean - r_mean})

    # targeted bridge insertion: connect top S hubs to top E hubs by adding m edges
    # find top S and top E by degree
    S_nodes = [n for n,c in cats_auth.items() if c=='pure_safety' and n in G_auth]
    E_nodes = [n for n,c in cats_auth.items() if c=='pure_ethics' and n in G_auth]
    degS = sorted(((n,G_auth.degree(n)) for n in S_nodes), key=lambda x: -x[1])
    degE = sorted(((n,G_auth.degree(n)) for n in E_nodes), key=lambda x: -x[1])
    topS = [n for n,_ in degS[:50]]
    topE = [n for n,_ in degE[:50]]
    for addm in [10,50,200]:
        H = G_auth.copy()
        added = 0
        i=0
        while added < addm and i < len(topS)*len(topE):
            a = random.choice(topS); b = random.choice(topE)
            if not H.has_edge(a,b):
                H.add_edge(a,b); added+=1
            i+=1
        r = reachability_by_hops(H, cats_auth, max_k=6)
        r_mean = mean([r['se_frac_by_k'][k] for k in [3,4,5]])
        results.append({'perturb':'add_bridges','param':addm,'removed':0,'se_mean_before':baseline_mean,'se_mean_after':r_mean,'delta':baseline_mean - r_mean})

    # sort by delta descending
    results_sorted = sorted(results, key=lambda x: -x['delta'])
    outcsv = os.path.join(OUT, f'aggressive_results_minp{min_p}.csv')
    with open(outcsv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        w.writeheader(); w.writerows(results_sorted)
    return results_sorted

def main():
    allres = {}
    for min_p in (0,2):
        print('Running aggressive perturbations min_p=',min_p)
        res = run_once(min_p)
        allres[min_p]=res
        print('Top perturbations:')
        for r in res[:5]:
            print(r)

if __name__ == '__main__':
    start=time.time(); main(); print('Done in', time.time()-start)

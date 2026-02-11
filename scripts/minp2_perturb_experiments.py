"""Focused perturbation experiments for min_p=2.

Run: python3 scripts/minp2_perturb_experiments.py
Outputs: perturb_grid_out/minp2_experiments.csv and plots for top configs
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time, random, csv
from collections import Counter
import networkx as nx
from statistics import mean
import matplotlib.pyplot as plt

from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network, reachability_by_hops

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)
random.seed(2)

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

def top_by_betweenness(G, k=500, sample_k=500):
    bc = nx.betweenness_centrality(G, k=sample_k, seed=0)
    return [n for n,_ in sorted(bc.items(), key=lambda x: -x[1])[:k]]

def bridge_frequency_nodes(G, cats, sample_pairs=30000, sample_target=5000):
    S = [n for n,c in cats.items() if c=='pure_safety' and n in G]
    E = [n for n,c in cats.items() if c=='pure_ethics' and n in G]
    freq = Counter(); valid=0; attempts=0
    while valid < sample_target and attempts < sample_pairs:
        u = random.choice(S); v = random.choice(E)
        attempts += 1
        try:
            path = nx.shortest_path(G, u, v)
        except nx.NetworkXNoPath:
            continue
        valid += 1
        for a in path[1:-1]:
            freq[a] += 1
    return [n for n,_ in freq.most_common()]

def apply_rewire(G, frac):
    H = G.copy()
    m = H.number_of_edges()
    swaps = max(1, int(frac * m))
    try:
        nx.double_edge_swap(H, nswap=swaps, max_tries=swaps*10)
    except Exception:
        pass
    return H

def reach_k5_stats(G, cats):
    r = reachability_by_hops(G, cats, max_k=5)
    se = r['se_frac_by_k'][4]
    ss = r['ss_frac_by_k'][4]
    ee = r['ee_frac_by_k'][4]
    delta = se - mean([ss,ee])
    return {'se':se,'ss':ss,'ee':ee,'delta':delta}

def run():
    papers = load_combined_data()
    min_p = 2
    print('Building baseline graphs (min_p=2)')
    G_auth, cats_auth, info_auth = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    total = G_auth.number_of_nodes()
    print('Authors:', total)
    baseline = reach_k5_stats(G_auth, cats_auth)
    print('Baseline @k=5:', baseline)

    # prepare rankings
    deg_sorted = sorted(G_auth.degree(), key=lambda x: -x[1])
    nodes_by_deg = [n for n,_ in deg_sorted]
    nodes_by_bc = top_by_betweenness(G_auth, k=total, sample_k=500)
    bridge_nodes = bridge_frequency_nodes(G_auth, cats_auth, sample_pairs=50000, sample_target=5000)

    experiments = []

    # percentiles and fixed K removals
    for pct in (0.01, 0.05, 0.10):
        k = max(1, int(pct * total))
        to_remove = set(nodes_by_deg[:k])
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        stats = reach_k5_stats(G_r, cats_r)
        experiments.append({'name':f'remove_top_pct_deg_{int(pct*100)}%','removed':k,'stats':stats})

    for K in (50,100,200,500,1000):
        k = min(K, total)
        to_remove = set(nodes_by_deg[:k])
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        stats = reach_k5_stats(G_r, cats_r)
        experiments.append({'name':f'remove_topK_deg_{K}','removed':k,'stats':stats})

    # betweenness removals
    for K in (50,100,200,500):
        to_remove = set(nodes_by_bc[:K])
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        stats = reach_k5_stats(G_r, cats_r)
        experiments.append({'name':f'remove_topK_bc_{K}','removed':len(to_remove),'stats':stats})

    # bridge-frequency removals
    for K in (10,50,100,200):
        to_remove = set(bridge_nodes[:K])
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        stats = reach_k5_stats(G_r, cats_r)
        experiments.append({'name':f'remove_topK_bridgefreq_{K}','removed':len(to_remove),'stats':stats})

    # rewires
    for frac in (0.02,0.05,0.10,0.20):
        H = apply_rewire(G_auth, frac)
        stats = reach_k5_stats(H, cats_auth)
        experiments.append({'name':f'rewire_frac_{int(frac*100)}pc','removed':0,'stats':stats})

    # targeted additions: add m S-E edges between top S and top E
    S_nodes = [n for n,c in cats_auth.items() if c=='pure_safety' and n in G_auth]
    E_nodes = [n for n,c in cats_auth.items() if c=='pure_ethics' and n in G_auth]
    degS = sorted(((n,G_auth.degree(n)) for n in S_nodes), key=lambda x: -x[1])
    degE = sorted(((n,G_auth.degree(n)) for n in E_nodes), key=lambda x: -x[1])
    topS = [n for n,_ in degS[:100]]
    topE = [n for n,_ in degE[:100]]
    for addm in (1,5,10,50):
        H = G_auth.copy(); added=0; tries=0
        while added < addm and tries < 10000:
            a = random.choice(topS); b = random.choice(topE)
            if not H.has_edge(a,b): H.add_edge(a,b); added+=1
            tries+=1
        stats = reach_k5_stats(H, cats_auth)
        experiments.append({'name':f'add_bridges_{addm}','removed':0,'stats':stats})

    # random removals controls
    for K in (50,100,500):
        nodes = list(G_auth.nodes())
        to_remove = set(random.sample(nodes, min(K,len(nodes))))
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        stats = reach_k5_stats(G_r, cats_r)
        experiments.append({'name':f'random_remove_{K}','removed':len(to_remove),'stats':stats})

    # save CSV summarizing experiments
    outcsv = os.path.join(OUT,'minp2_experiments.csv')
    with open(outcsv,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['name','removed','se_k5','ss_k5','ee_k5','delta'])
        for e in experiments:
            s=e['stats']
            w.writerow([e['name'], e['removed'], s['se'], s['ss'], s['ee'], s['delta']])

    # rank by delta (maximize SE - avg(SS,EE)?? here we want maximize segregation meaning gap positive? we want SE small relative to avg SS/EE so delta negative large magnitude; sort by delta ascending)
    experiments_sorted = sorted(experiments, key=lambda x: x['stats']['delta'])
    top = experiments_sorted[:10]
    print('\nTop 10 perturbations that increase segregation (most negative delta):')
    for e in top:
        s=e['stats']
        print(e['name'], 'removed', e['removed'], 'SE@5', round(s['se'],6), 'SS@5', round(s['ss'],6), 'EE@5', round(s['ee'],6), 'delta', round(s['delta'],6))

    # plot best one: compare baseline vs best removals
    best = top[0]
    print('\nPlotting baseline vs', best['name'])
    # rebuild G for best
    bname = best['name']
    if bname.startswith('remove'):
        # find removed set
        parts = bname.split('_')
        if 'pct' in bname:
            # remove percent
            pct = int(parts[3].replace('%',''))/100.0
            k = max(1,int(pct * total))
            to_remove = set(nodes_by_deg[:k])
        elif parts[2]=='topK':
            K = int(parts[3])
            to_remove = set(nodes_by_deg[:min(K,total)])
        elif 'bc' in bname:
            K = int(parts[-1]); to_remove = set(nodes_by_bc[:K])
        elif 'bridgefreq' in bname:
            K = int(parts[-1]); to_remove = set(bridge_nodes[:K])
        else:
            to_remove = set()
        filtered = remove_authors_from_papers(papers, to_remove)
        G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
    elif bname.startswith('rewire'):
        frac = float(bname.split('_')[-1].replace('pc',''))/100.0
        G_r = apply_rewire(G_auth, frac); cats_r = cats_auth
    elif bname.startswith('add_bridges'):
        # regenerate from experiments list
        parts = bname.split('_'); addm=int(parts[-1])
        H = G_auth.copy(); added=0; tries=0
        while added<addm and tries<10000:
            a=random.choice(topS); b=random.choice(topE)
            if not H.has_edge(a,b): H.add_edge(a,b); added+=1
            tries+=1
        G_r = H; cats_r = cats_auth
    else:
        G_r = G_auth; cats_r = cats_auth

    # reachability curves k=1..12
    def reach_curve(G,cats,maxk=12):
        r = reachability_by_hops(G,cats,max_k=maxk)
        return r['se_frac_by_k'], r['ss_frac_by_k'], r['ee_frac_by_k']

    se_b, ss_b, ee_b = reach_curve(G_auth, cats_auth, maxk=12)
    se_r, ss_r, ee_r = reach_curve(G_r, cats_r, maxk=12)
    ks = list(range(1,13))
    fig,ax = plt.subplots(figsize=(6,4))
    ax.plot(ks, [x*100 for x in se_b], label='SE baseline')
    ax.plot(ks, [x*100 for x in se_r], label=f'SE {best["name"]}')
    ax.set_xlabel('k'); ax.set_ylabel('SE %'); ax.set_title('SE reachability baseline vs perturbation (min_p=2)')
    ax.legend(); fig.savefig(os.path.join(OUT,'minp2_best_perturb_compare.png'), dpi=150)
    print('Wrote', outcsv, 'and plot minp2_best_perturb_compare.png')

if __name__ == '__main__':
    start=time.time(); run(); print('Done in', time.time()-start)

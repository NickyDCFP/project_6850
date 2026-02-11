"""Removal sensitivity and null tests for author bridges and reachability.

Run: python3 scripts/removal_and_nulls.py
Generates CSVs and plots in perturb_grid_out/.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time, random, csv
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean

from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network, build_paper_coauthorship_network, reachability_by_hops

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)

random.seed(0)

def top_k_by_degree(G, k):
    return [n for n,_ in sorted(G.degree(), key=lambda x: -x[1])[:k]]

def approx_betweenness(G, k=None):
    # use sampling if k provided
    try:
        if k is None:
            return nx.betweenness_centrality(G)
        else:
            return nx.betweenness_centrality(G, k=k, seed=0)
    except Exception:
        return nx.betweenness_centrality(G)

def sample_SE_pairs(S_nodes, E_nodes, max_pairs):
    total = len(S_nodes)*len(E_nodes)
    if total <= max_pairs:
        for s in S_nodes:
            for e in E_nodes:
                yield (s,e)
    else:
        seen=set(); attempts=0
        while len(seen) < max_pairs and attempts < max_pairs*20:
            s=random.choice(S_nodes); e=random.choice(E_nodes)
            if s==e or (s,e) in seen:
                attempts+=1; continue
            seen.add((s,e)); yield (s,e)

def bridge_frac_authors(G, cats, top_set, sample_target=2000, max_pairs=200000):
    S=[n for n,c in cats.items() if c=='pure_safety' and n in G]
    E=[n for n,c in cats.items() if c=='pure_ethics' and n in G]
    valid=0; thru=0
    for u,v in sample_SE_pairs(S,E,max_pairs):
        try:
            path = nx.shortest_path(G,u,v)
        except nx.NetworkXNoPath:
            continue
        valid+=1
        internal = path[1:-1]
        if any(n in top_set for n in internal): thru+=1
        if valid>=sample_target: break
    return {'valid':valid,'through':thru,'frac':thru/valid if valid else None}

def compute_reachability_snapshot(G, cats, kmax=6):
    r = reachability_by_hops(G, cats, max_k=kmax)
    return r

def remove_authors_from_papers(papers, remove_set):
    new = []
    for p in papers:
        new_auths=[]
        for a in p.get('authors',[]):
            name = a.get('name') if isinstance(a, dict) else a
            if name not in remove_set:
                new_auths.append(a)
        if new_auths:
            np = dict(p); np['authors']=new_auths; new.append(np)
    return new

def label_shuffle_bridge_null(G, cats, top_set, reps=200, sample_target=1000):
    # shuffle labels across existing nodes; keep topology & top_set fixed
    nodes = list(G.nodes())
    labels = [cats.get(n,'mixed') for n in nodes]
    fracs=[]
    for i in range(reps):
        random.shuffle(labels)
        shuffled = {n:labels[idx] for idx,n in enumerate(nodes)}
        bf = bridge_frac_authors(G, shuffled, top_set, sample_target=sample_target)
        fracs.append(bf['frac'] if bf['frac'] is not None else 0)
    return fracs

def rewire_bridge_null(G, cats, top_set, reps=100, swaps_per_edge=1, sample_target=1000):
    fracs=[]
    m = G.number_of_edges()
    swaps = max(1, int(swaps_per_edge * m))
    for i in range(reps):
        H = G.copy()
        try:
            nx.double_edge_swap(H, nswap=swaps, max_tries=swaps*10)
        except Exception:
            pass
        bf = bridge_frac_authors(H, cats, top_set, sample_target=sample_target)
        fracs.append(bf['frac'] if bf['frac'] is not None else 0)
    return fracs

def run_for_minp(min_p, kmax=6):
    print('\n=== RUN min_p=',min_p)
    papers = load_combined_data()
    G_auth, cats_auth, info_auth = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    G_paper, cats_paper, info_paper = build_paper_coauthorship_network(papers, min_papers_per_author=min_p)
    print('Authors:',G_auth.number_of_nodes(),'Edges:',G_auth.number_of_edges())

    # baseline reachability
    reach_auth = compute_reachability_snapshot(G_auth, cats_auth, kmax=kmax)

    # top author lists
    N = G_auth.number_of_nodes()
    top1pct = max(1, int(0.01*N))
    top5pct = max(1, int(0.05*N))

    deg_sorted = sorted(G_auth.degree(), key=lambda x: -x[1])
    top100_deg = set([n for n,_ in deg_sorted[:100]])
    top500_deg = set([n for n,_ in deg_sorted[:500]])

    print('Computing approx betweenness (k=500 sample)...')
    bc = approx_betweenness(G_auth, k=500)
    bc_sorted = sorted(bc.items(), key=lambda x: -x[1])
    top100_bc = set([n for n,_ in bc_sorted[:100]])
    top500_bc = set([n for n,_ in bc_sorted[:500]])

    # Bridge fractions
    bf_top100_deg = bridge_frac_authors(G_auth, cats_auth, top100_deg, sample_target=2000)
    bf_top500_deg = bridge_frac_authors(G_auth, cats_auth, top500_deg, sample_target=2000)
    bf_top100_bc = bridge_frac_authors(G_auth, cats_auth, top100_bc, sample_target=2000)
    bf_top500_bc = bridge_frac_authors(G_auth, cats_auth, top500_bc, sample_target=2000)

    # Nulls (label shuffle) for top500_deg
    print('Running label-shuffle nulls (200 reps) for top500_deg')
    lab_fracs = label_shuffle_bridge_null(G_auth, cats_auth, top500_deg, reps=200, sample_target=1000)
    # Rewire nulls
    print('Running rewire nulls (100 reps) for top500_deg')
    rew_fracs = rewire_bridge_null(G_auth, cats_auth, top500_deg, reps=100, swaps_per_edge=1, sample_target=1000)

    # Removal sensitivity: remove top100/500 by degree and bc
    removals = [ ('deg',100,top100_deg), ('deg',500,top500_deg), ('bc',100,top100_bc), ('bc',500,top500_bc) ]
    removal_results = []
    for typ,k,topset in removals:
        print('Removing',typ,k)
        removed = set(topset)
        # remove these authors from papers and rebuild graphs
        filtered = remove_authors_from_papers(papers, removed)
        G_auth_nb, cats_auth_nb, info_auth_nb = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        reach_after = compute_reachability_snapshot(G_auth_nb, cats_auth_nb, kmax=kmax)
        # compute delta at k=4..6 average
        deltas = []
        for k in (4,5,6):
            before = reach_auth['se_frac_by_k'][k-1]
            after = reach_after['se_frac_by_k'][k-1] if k-1 < len(reach_after['se_frac_by_k']) else None
            if after is None: continue
            deltas.append(before - after)
        removal_results.append({'type':typ,'k':k,'delta_mean_k4_6': mean(deltas) if deltas else None,'before_counts': [reach_auth['se_counts_by_k'][i] for i in range(3,6)],'after_counts':[reach_after['se_counts_by_k'][i] for i in range(3,6)]})

    # Save CSV summary
    out_csv = os.path.join(OUT, f'removal_nulls_minp{min_p}.csv')
    with open(out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['N_authors', G_auth.number_of_nodes()])
        w.writerow(['N_edges', G_auth.number_of_edges()])
        w.writerow(['se_frac_k4_before', reach_auth['se_frac_by_k'][3]])
        w.writerow(['se_frac_k5_before', reach_auth['se_frac_by_k'][4]])
        w.writerow(['se_frac_k6_before', reach_auth['se_frac_by_k'][5]])
        w.writerow(['bridge_top100_deg_frac', bf_top100_deg['frac']])
        w.writerow(['bridge_top500_deg_frac', bf_top500_deg['frac']])
        w.writerow(['bridge_top100_bc_frac', bf_top100_bc['frac']])
        w.writerow(['bridge_top500_bc_frac', bf_top500_bc['frac']])
        w.writerow(['label_shuffle_top500_deg_mean', mean(lab_fracs)])
        w.writerow(['label_shuffle_top500_deg_std', (sum((x-mean(lab_fracs))**2 for x in lab_fracs)/len(lab_fracs))**0.5])
        w.writerow(['rewire_top500_deg_mean', mean(rew_fracs)])
        w.writerow(['rewire_top500_deg_std', (sum((x-mean(rew_fracs))**2 for x in rew_fracs)/len(rew_fracs))**0.5])
        for rr in removal_results:
            w.writerow([f"removal_{rr['type']}_{rr['k']}_delta_mean_k4_6", rr['delta_mean_k4_6']])

    # plots: reachability before/after for one removal (deg 100 and 500)
    try:
        import matplotlib.pyplot as plt
        ks = list(range(1,7))
        se_before = [reach_auth['se_frac_by_k'][i] for i in range(6)]
        fig,ax = plt.subplots()
        ax.plot(ks,se_before,label='before')
        # after for deg100
        filtered100 = remove_authors_from_papers(papers, top100_deg)
        G_a_100, cats_a_100, _ = build_author_coauthorship_network(filtered100, min_papers_per_author=min_p)
        reach_100 = compute_reachability_snapshot(G_a_100, cats_a_100, kmax=6)
        se_after100 = [reach_100['se_frac_by_k'][i] for i in range(6)]
        ax.plot(ks,se_after100,label='after remove top100_deg')
        # after for deg500
        filtered500 = remove_authors_from_papers(papers, top500_deg)
        G_a_500, cats_a_500, _ = build_author_coauthorship_network(filtered500, min_papers_per_author=min_p)
        reach_500 = compute_reachability_snapshot(G_a_500, cats_a_500, kmax=6)
        se_after500 = [reach_500['se_frac_by_k'][i] for i in range(6)]
        ax.plot(ks,se_after500,label='after remove top500_deg')
        ax.set_xlabel('k'); ax.set_ylabel('SE fraction'); ax.set_title(f'min_p={min_p} author SE reachability before/after')
        ax.legend()
        fig.savefig(os.path.join(OUT,f'reachability_removals_minp{min_p}.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print('Plotting failed',e)

    # bridge null histogram plot
    try:
        fig,ax = plt.subplots()
        ax.hist(lab_fracs, bins=30, alpha=0.6, label='label-shuffle')
        ax.hist(rew_fracs, bins=30, alpha=0.6, label='rewire')
        ax.axvline(bf_top500_deg['frac'] if bf_top500_deg['frac'] is not None else 0, color='red', label='observed top500_deg')
        ax.set_xlabel('bridge frac'); ax.set_ylabel('count'); ax.set_title(f'Nulls vs observed bridge frac (min_p={min_p})')
        ax.legend()
        fig.savefig(os.path.join(OUT,f'bridge_nulls_minp{min_p}.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print('Plotting failed',e)

    print('Wrote', out_csv)

def main():
    for min_p in (0,2):
        run_for_minp(min_p, kmax=6)

if __name__ == '__main__':
    start=time.time(); main(); print('All done in',time.time()-start)

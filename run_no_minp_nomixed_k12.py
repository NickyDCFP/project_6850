#!/usr/bin/env python3
"""Run no-min_p analysis but remove mixed nodes, compute reachability up to k=12, save plots,
and print SE - AVG(SS,EE) per k for author and paper graphs.
"""
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict]:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_author(name: str) -> str:
    if not name:
        return name
    s = name.strip()
    parts = s.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return ' '.join(parts)


def build_full_graphs(papers: List[Dict]):
    # Paper graph
    sel = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]
    P = nx.Graph()
    p_cats = {}
    for i, p in enumerate(sel):
        s = float(p.get('safety_score', 0) or 0)
        e = float(p.get('ethics_score', 0) or 0)
        if s > 0 and e == 0:
            cat = 'pure_safety'
        elif e > 0 and s == 0:
            cat = 'pure_ethics'
        elif s > 0 and e > 0:
            cat = 'mixed'
        else:
            cat = 'neither'
        P.add_node(i, safety=s, ethics=e)
        p_cats[i] = cat

    author_to_papers = defaultdict(list)
    for i, p in enumerate(sel):
        for a in p.get('authors', []):
            if a:
                author_to_papers[normalize_author(a)].append(i)

    for plist in author_to_papers.values():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            for j in range(i+1, len(plist)):
                u, v = plist[i], plist[j]
                if P.has_edge(u, v):
                    P[u][v]['weight'] += 1
                else:
                    P.add_edge(u, v, weight=1)

    # Author graph (aggregate scores)
    author_scores = defaultdict(lambda: {'safety': 0.0, 'ethics': 0.0, 'papers': set()})
    for idx, p in enumerate(sel):
        s = float(p.get('safety_score', 0) or 0)
        e = float(p.get('ethics_score', 0) or 0)
        for a in p.get('authors', []):
            if not a:
                continue
            na = normalize_author(a)
            author_scores[na]['safety'] += s
            author_scores[na]['ethics'] += e
            author_scores[na]['papers'].add(idx)

    A = nx.Graph()
    a_cats = {}
    for a, sc in author_scores.items():
        st = sc['safety']
        et = sc['ethics']
        if st > 0 and et == 0:
            cat = 'pure_safety'
        elif et > 0 and st == 0:
            cat = 'pure_ethics'
        elif st > 0 and et > 0:
            cat = 'mixed'
        else:
            cat = 'neither'
        A.add_node(a, safety=st, ethics=et)
        a_cats[a] = cat

    paper_to_authors = defaultdict(list)
    for a, sc in author_scores.items():
        for pidx in sc['papers']:
            paper_to_authors[pidx].append(a)

    for alist in paper_to_authors.values():
        if len(alist) < 2:
            continue
        for i in range(len(alist)):
            for j in range(i+1, len(alist)):
                u, v = alist[i], alist[j]
                if u == v:
                    continue
                if A.has_edge(u, v):
                    A[u][v]['weight'] += 1
                else:
                    A.add_edge(u, v, weight=1)

    return A, a_cats, P, p_cats


def remove_mixed_nodes(G: nx.Graph, categories: Dict):
    keep = [n for n in G.nodes() if categories.get(n) in ('pure_safety', 'pure_ethics')]
    H = G.subgraph(keep).copy()
    new_cat = {n: categories[n] for n in H.nodes()}
    return H, new_cat


def reachability_by_hops(G: nx.Graph, categories: Dict, max_k: int = 12):
    S_nodes = [n for n, c in categories.items() if c == 'pure_safety' and n in G]
    E_nodes = [n for n, c in categories.items() if c == 'pure_ethics' and n in G]
    Ns = len(S_nodes)
    Ne = len(E_nodes)
    total_SE = Ns * Ne
    total_SS = Ns * (Ns - 1) // 2
    total_EE = Ne * (Ne - 1) // 2

    S_index = {n: i for i, n in enumerate(sorted(S_nodes))}
    E_index = {n: i for i, n in enumerate(sorted(E_nodes))}

    se_counts = [0] * (max_k + 1)
    ss_counts = [0] * (max_k + 1)
    ee_counts = [0] * (max_k + 1)

    for s in S_nodes:
        lengths = nx.single_source_shortest_path_length(G, s, cutoff=max_k)
        for t, d in lengths.items():
            if d == 0:
                continue
            if d <= max_k:
                if t in E_index:
                    se_counts[d] += 1
                if t in S_index and S_index[t] > S_index[s]:
                    ss_counts[d] += 1

    for e in E_nodes:
        lengths = nx.single_source_shortest_path_length(G, e, cutoff=max_k)
        for t, d in lengths.items():
            if d == 0:
                continue
            if d <= max_k and t in E_index and E_index[t] > E_index[e]:
                ee_counts[d] += 1

    se_cum = []
    ss_cum = []
    ee_cum = []
    s_acc = ss_acc = ee_acc = 0
    for k in range(1, max_k + 1):
        s_acc += se_counts[k]
        ss_acc += ss_counts[k]
        ee_acc += ee_counts[k]
        se_cum.append(s_acc)
        ss_cum.append(ss_acc)
        ee_cum.append(ee_acc)

    se_frac = [c / total_SE if total_SE else 0.0 for c in se_cum]
    ss_frac = [c / total_SS if total_SS else 0.0 for c in ss_cum]
    ee_frac = [c / total_EE if total_EE else 0.0 for c in ee_cum]

    return {'se_frac_by_k': se_frac, 'ss_frac_by_k': ss_frac, 'ee_frac_by_k': ee_frac, 'Ns': Ns, 'Ne': Ne}


def plot_percentages(res, out_prefix, title_prefix=''):
    ks = list(range(1, len(res['se_frac_by_k']) + 1))
    ss = [f * 100.0 for f in res['ss_frac_by_k']]
    se = [f * 100.0 for f in res['se_frac_by_k']]
    ee = [f * 100.0 for f in res['ee_frac_by_k']]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, ss, marker='o', label='Safety–Safety')
    plt.plot(ks, se, marker='o', label='Safety–Ethics')
    plt.plot(ks, ee, marker='o', label='Ethics–Ethics')
    plt.xticks(ks)
    plt.xlabel('hops (k)')
    plt.ylabel('Reachable pairs (%)')
    plt.title(title_prefix + 'Reachability by hops (<=k)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + '.png', dpi=200)
    plt.close()


def main():
    try:
        from analyze_aggregated import load_combined_data
        papers = load_combined_data()
    except Exception:
        papers = load_jsonl('sample.jsonl')

    if not papers:
        print('No papers found; exiting')
        return

    for p in papers:
        p['authors'] = [normalize_author(a) for a in p.get('authors', []) if a]

    A, a_cats, P, p_cats = build_full_graphs(papers)

    # remove mixed nodes
    A_nomix, a_nomix_cats = remove_mixed_nodes(A, a_cats)
    P_nomix, p_nomix_cats = remove_mixed_nodes(P, p_cats)

    print(f'Author graph (no min_p, no mixed): nodes={A_nomix.number_of_nodes()}, edges={A_nomix.number_of_edges()}')
    print(f'Paper graph (no min_p, no mixed): nodes={P_nomix.number_of_nodes()}, edges={P_nomix.number_of_edges()}')

    max_k = 12
    a_res = reachability_by_hops(A_nomix, a_nomix_cats, max_k=max_k)
    p_res = reachability_by_hops(P_nomix, p_nomix_cats, max_k=max_k)

    plot_percentages(a_res, 'reachability_author_no_minp_nomixed_k12', title_prefix='Author (no_minp, no_mixed) - ')
    plot_percentages(p_res, 'reachability_paper_no_minp_nomixed_k12', title_prefix='Paper (no_minp, no_mixed) - ')

    # Print SE - AVG(SS,EE) for each k
    ks = list(range(1, max_k + 1))
    print('\nAuthor: SE - AVG(SS,EE) (% points)')
    for i, k in enumerate(ks):
        se = a_res['se_frac_by_k'][i]*100
        ss = a_res['ss_frac_by_k'][i]*100
        ee = a_res['ee_frac_by_k'][i]*100
        diff = se - ((ss + ee)/2.0)
        print(f'k={k}: {diff:.4f}')

    print('\nPaper: SE - AVG(SS,EE) (% points)')
    for i, k in enumerate(ks):
        se = p_res['se_frac_by_k'][i]*100
        ss = p_res['ss_frac_by_k'][i]*100
        ee = p_res['ee_frac_by_k'][i]*100
        diff = se - ((ss + ee)/2.0)
        print(f'k={k}: {diff:.4f}')

    print('\nPlots saved: reachability_author_no_minp_nomixed_k12.png, reachability_paper_no_minp_nomixed_k12.png')


if __name__ == '__main__':
    main()

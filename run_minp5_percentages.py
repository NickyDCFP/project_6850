#!/usr/bin/env python3
"""Run a focused analysis: min_papers_per_author=5 and print reachability percentages k=1..6.
"""
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

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

def build_author_network_with_filter(papers: List[Dict], min_papers_per_author: int):
    author_counts = defaultdict(int)
    for p in papers:
        s = float(p.get('safety_score', 0) or 0)
        e = float(p.get('ethics_score', 0) or 0)
        if s > 0 or e > 0:
            for a in p.get('authors', []):
                if a:
                    author_counts[normalize_author(a)] += 1

    qualified = {a for a, c in author_counts.items() if c >= min_papers_per_author}
    scores = defaultdict(lambda: {'safety': 0.0, 'ethics': 0.0, 'papers': set()})
    for idx, p in enumerate(papers):
        s = float(p.get('safety_score', 0) or 0)
        e = float(p.get('ethics_score', 0) or 0)
        if s > 0 or e > 0:
            for a in p.get('authors', []):
                if not a:
                    continue
                na = normalize_author(a)
                if na in qualified:
                    scores[na]['safety'] += s
                    scores[na]['ethics'] += e
                    scores[na]['papers'].add(idx)

    G = nx.Graph()
    cats = {}
    for a, sc in scores.items():
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
        G.add_node(a, safety=st, ethics=et)
        cats[a] = cat

    paper_to_authors = defaultdict(list)
    for a, sc in scores.items():
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
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G, cats

def build_paper_network_with_author_filter(papers: List[Dict], min_papers_per_author: int):
    sel = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]
    author_counts = defaultdict(int)
    for p in sel:
        for a in p.get('authors', []):
            if a:
                author_counts[normalize_author(a)] += 1

    qualified = {a for a, c in author_counts.items() if c >= min_papers_per_author}
    filtered = []
    for p in sel:
        authors = [normalize_author(a) for a in p.get('authors', []) if a]
        q = [a for a in authors if a in qualified]
        if q:
            p2 = p.copy()
            p2['authors'] = q
            filtered.append(p2)

    G = nx.Graph()
    cats = {}
    for i, p in enumerate(filtered):
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
        G.add_node(i, safety=s, ethics=e)
        cats[i] = cat

    author_to_papers = defaultdict(list)
    for i, p in enumerate(filtered):
        for a in p.get('authors', []):
            author_to_papers[a].append(i)

    for plist in author_to_papers.values():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            for j in range(i+1, len(plist)):
                u, v = plist[i], plist[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G, cats

def reachability_by_hops(G: nx.Graph, categories: Dict, max_k: int = 6):
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

    return {
        'se_frac_by_k': se_frac, 'ss_frac_by_k': ss_frac, 'ee_frac_by_k': ee_frac,
        'Ns': Ns, 'Ne': Ne,
    }

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

    thr = 5
    aG, a_cats = build_author_network_with_filter(papers, thr)
    pG, p_cats = build_paper_network_with_author_filter(papers, thr)

    a_res = reachability_by_hops(aG, a_cats, max_k=6)
    p_res = reachability_by_hops(pG, p_cats, max_k=6)

    ks = list(range(1, 7))
    print(f'Author graph (minp={thr}): Ns={a_res["Ns"]}, Ne={a_res["Ne"]}')
    print('Author S–E % by k:', [round(x*100, 4) for x in a_res['se_frac_by_k']])
    print('Author S–S % by k:', [round(x*100, 4) for x in a_res['ss_frac_by_k']])
    print('Author E–E % by k:', [round(x*100, 4) for x in a_res['ee_frac_by_k']])
    print('')
    print(f'Paper graph (minp={thr}): Ns={p_res["Ns"]}, Ne={p_res["Ne"]}')
    print('Paper S–E % by k:', [round(x*100, 4) for x in p_res['se_frac_by_k']])
    print('Paper S–S % by k:', [round(x*100, 4) for x in p_res['ss_frac_by_k']])
    print('Paper E–E % by k:', [round(x*100, 4) for x in p_res['ee_frac_by_k']])

if __name__ == '__main__':
    main()

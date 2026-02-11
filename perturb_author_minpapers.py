#!/usr/bin/env python3
"""
Sweep author-min-paper thresholds and compute reachability-by-hops for paper and author graphs.

This script is intentionally self-contained so it can be run independently of the larger
analysis script. It reuses small helper functions (normalization, graph building, reachability)
derived from `network_asp_analysis.py` but keeps the logic focused on the perturbation sweep.
"""
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
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


def build_paper_network_with_author_filter(papers: List[Dict], min_papers_per_author: int) -> Tuple[nx.Graph, Dict]:
    """Build paper graph but only keep authors who appear in >= min_papers_per_author papers.
    Papers that lose all qualified authors are removed.
    Returns graph (nodes are integer indices into filtered_papers) and categories mapping.
    """
    sel = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]

    # Count papers per normalized author
    author_counts = defaultdict(int)
    for p in sel:
        for a in p.get('authors', []):
            if a:
                author_counts[normalize_author(a)] += 1

    qualified = {a for a, c in author_counts.items() if c >= min_papers_per_author}

    # Filter papers to those that have at least one qualified author; update their author lists
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


def build_author_network_with_filter(papers: List[Dict], min_papers_per_author: int) -> Tuple[nx.Graph, Dict]:
    """Build author graph keeping only authors with >= min_papers_per_author safety/ethics papers.
    Returns graph with node names as normalized author strings and categories mapping.
    """
    author_counts = defaultdict(int)
    for p in papers:
        s = float(p.get('safety_score', 0) or 0)
        e = float(p.get('ethics_score', 0) or 0)
        if s > 0 or e > 0:
            for a in p.get('authors', []):
                if a:
                    author_counts[normalize_author(a)] += 1

    qualified = {a for a, c in author_counts.items() if c >= min_papers_per_author}

    # aggregate per-author scores
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

    # coauthorship edges among qualified authors
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
        'Ns': Ns, 'Ne': Ne,
        'total_SE': total_SE, 'total_SS': total_SS, 'total_EE': total_EE,
        'se_counts_by_k': se_cum, 'ss_counts_by_k': ss_cum, 'ee_counts_by_k': ee_cum,
        'se_frac_by_k': se_frac, 'ss_frac_by_k': ss_frac, 'ee_frac_by_k': ee_frac,
    }


def plot_reachability(res, out_path_prefix: str, title_prefix: str = '', counts: bool = False):
    ks = list(range(1, len(res['se_frac_by_k']) + 1))
    if counts:
        ss = res.get('ss_counts_by_k', [])
        se = res.get('se_counts_by_k', [])
        ee = res.get('ee_counts_by_k', [])
    else:
        ss = [f * 100.0 for f in res['ss_frac_by_k']]
        se = [f * 100.0 for f in res['se_frac_by_k']]
        ee = [f * 100.0 for f in res['ee_frac_by_k']]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, ss, marker='o', label='Safety–Safety')
    plt.plot(ks, se, marker='o', label='Safety–Ethics')
    plt.plot(ks, ee, marker='o', label='Ethics–Ethics')
    plt.xticks(ks)
    plt.xlabel('hops (k)')
    plt.ylabel('Reachable pairs (count)' if counts else 'Reachable pairs (%)')
    plt.title(title_prefix + ('Reachability by hops (counts)' if counts else 'Reachability by hops (<=k)'))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_path_prefix + ("_counts.png" if counts else ".png")
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    print('Loading data via analyze_aggregated.load_combined_data if available, else sample.jsonl')
    # try to import the loader from analyze_aggregated if present
    try:
        from analyze_aggregated import load_combined_data
        papers = load_combined_data()
    except Exception:
        print('Could not import load_combined_data, falling back to sample.jsonl')
        papers = load_jsonl('sample.jsonl')

    if not papers:
        print('No papers found; exiting')
        return

    # normalize authors in-place
    for p in papers:
        p['authors'] = [normalize_author(a) for a in p.get('authors', []) if a]

    out_rows = []
    thresholds = list(range(3, 11))
    for thr in thresholds:
        print(f'\n=== Threshold: authors with >= {thr} papers ===')
        # build author graph and paper graph under this threshold
        aG, a_cats = build_author_network_with_filter(papers, min_papers_per_author=thr)
        pG, p_cats = build_paper_network_with_author_filter(papers, min_papers_per_author=thr)

        print(f'Author graph nodes: {aG.number_of_nodes()}, edges: {aG.number_of_edges()}')
        print(f'Paper graph nodes: {pG.number_of_nodes()}, edges: {pG.number_of_edges()}')

        a_res = reachability_by_hops(aG, a_cats, max_k=6)
        p_res = reachability_by_hops(pG, p_cats, max_k=6)

        # save plots
        plot_reachability(a_res, f'reachability_author_minp_{thr}', title_prefix=f'Author (minp={thr}) - ')
        plot_reachability(a_res, f'reachability_author_minp_{thr}', title_prefix=f'Author (minp={thr}) - ', counts=True)
        plot_reachability(p_res, f'reachability_paper_minp_{thr}', title_prefix=f'Paper (minp={thr}) - ')
        plot_reachability(p_res, f'reachability_paper_minp_{thr}', title_prefix=f'Paper (minp={thr}) - ', counts=True)

        # add summary row (k=6 values)
        row = {
            'threshold': thr,
            'author_nodes': aG.number_of_nodes(), 'author_edges': aG.number_of_edges(),
            'paper_nodes': pG.number_of_nodes(), 'paper_edges': pG.number_of_edges(),
            'author_SE_k6_pct': a_res['se_frac_by_k'][-1] * 100.0,
            'author_SS_k6_pct': a_res['ss_frac_by_k'][-1] * 100.0,
            'author_EE_k6_pct': a_res['ee_frac_by_k'][-1] * 100.0,
            'paper_SE_k6_pct': p_res['se_frac_by_k'][-1] * 100.0,
            'paper_SS_k6_pct': p_res['ss_frac_by_k'][-1] * 100.0,
            'paper_EE_k6_pct': p_res['ee_frac_by_k'][-1] * 100.0,
            'paper_SE_k6_count': p_res['se_counts_by_k'][-1],
            'paper_SS_k6_count': p_res['ss_counts_by_k'][-1],
            'paper_EE_k6_count': p_res['ee_counts_by_k'][-1],
        }
        out_rows.append(row)

    # write CSV summary
    csv_path = 'perturb_author_minpapers_summary.csv'
    keys = list(out_rows[0].keys()) if out_rows else []
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f'Finished sweep; summary written to {csv_path}. PNGs saved for each threshold (percent and counts).')


if __name__ == '__main__':
    main()

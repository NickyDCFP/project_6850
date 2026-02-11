#!/usr/bin/env python3
"""Rewire graphs to increase within-cluster connectivity (S-S, E-E) while avoiding new S-E edges.

Runs experiments for minp thresholds and reports reachability up to k=7 before and after rewiring.
"""
import csv
import json
import os
import random
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


def build_paper_graph(papers: List[Dict], min_papers_per_author: int = None):
    # Determine qualified authors
    author_counts = defaultdict(int)
    for p in papers:
        for a in p.get('authors', []):
            if a:
                author_counts[normalize_author(a)] += 1

    if min_papers_per_author is None:
        qualified = set(author_counts.keys())
    else:
        qualified = {a for a, c in author_counts.items() if c >= min_papers_per_author}

    sel = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]
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


def reachability_by_hops(G: nx.Graph, categories: Dict, max_k: int = 7):
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
        'se_counts_by_k': se_cum, 'ss_counts_by_k': ss_cum, 'ee_counts_by_k': ee_cum,
        'Ns': Ns, 'Ne': Ne,
    }


def constrained_rewire_increase_within(G: nx.Graph, categories: Dict, target_swaps_fraction: float = 0.05, max_attempts_per_swap: int = 50, seed: int = None):
    """Attempt degree-preserving swaps that avoid creating S-E edges and try to increase same-category edges.

    This picks random edge pairs and proposes swaps; accepts swaps that do not create S-E edges and that
    increase the sum of same-category edges or decrease S-E edge count.
    """
    if seed is not None:
        random.seed(seed)

    m = G.number_of_edges()
    if m == 0:
        return G.copy()
    target_swaps = max(1, int(m * target_swaps_fraction))
    edges = list(G.edges())

    def edge_category(u, v):
        cu = categories.get(u, 'neither')
        cv = categories.get(v, 'neither')
        if cu == 'pure_safety' and cv == 'pure_safety':
            return 'SS'
        if cu == 'pure_ethics' and cv == 'pure_ethics':
            return 'EE'
        if (cu == 'pure_safety' and cv == 'pure_ethics') or (cu == 'pure_ethics' and cv == 'pure_safety'):
            return 'SE'
        return 'OTHER'

    H = G.copy()
    successful = 0
    attempts = 0
    max_attempts = target_swaps * max_attempts_per_swap
    edge_list = list(H.edges())

    while successful < target_swaps and attempts < max_attempts:
        attempts += 1
        e1 = random.choice(edge_list)
        e2 = random.choice(edge_list)
        if e1 == e2:
            continue
        a, b = e1
        c, d = e2

        # avoid overlapping nodes
        if len({a, b, c, d}) < 4:
            continue

        # two possible swaps: (a,c)+(b,d) or (a,d)+(b,c)
        propose = [ (a, c, b, d), (a, d, b, c) ]
        accepted = False
        for (u1, v1, u2, v2) in propose:
            if H.has_edge(u1, v1) or H.has_edge(u2, v2):
                continue
            # avoid self loops
            if u1 == v1 or u2 == v2:
                continue

            # categories of proposed edges
            cat1 = edge_category(u1, v1)
            cat2 = edge_category(u2, v2)

            # reject if any proposed edge is SE (we avoid creating S-E edges)
            if cat1 == 'SE' or cat2 == 'SE':
                continue

            # current categories
            cur1 = edge_category(a, b)
            cur2 = edge_category(c, d)

            # compute score: number of same-category edges (SS or EE) after swap minus before
            def same_score(cat):
                return 1 if cat in ('SS', 'EE') else 0

            before = same_score(cur1) + same_score(cur2)
            after = same_score(cat1) + same_score(cat2)

            # also compute SE count before/after
            se_before = (1 if cur1 == 'SE' else 0) + (1 if cur2 == 'SE' else 0)
            se_after = (1 if cat1 == 'SE' else 0) + (1 if cat2 == 'SE' else 0)

            # accept if after >= before and se_after <= se_before, and prefer cases where after>before
            if (after > before and se_after <= se_before) or (after == before and se_after < se_before):
                # perform swap
                H.remove_edge(a, b)
                H.remove_edge(c, d)
                H.add_edge(u1, v1)
                H.add_edge(u2, v2)
                # update edge_list
                edge_list.remove((a, b))
                edge_list.remove((c, d))
                edge_list.append((u1, v1))
                edge_list.append((u2, v2))
                successful += 1
                accepted = True
                break
        # end propose loop
    # end while
    return H


def plot_before_after(before_res, after_res, out_prefix, title_prefix=''):
    ks = list(range(1, len(before_res['se_frac_by_k']) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(ks, [x*100 for x in before_res['ss_frac_by_k']], marker='o', color='C0', linestyle='--', label='SS before')
    plt.plot(ks, [x*100 for x in before_res['se_frac_by_k']], marker='o', color='C1', linestyle='--', label='SE before')
    plt.plot(ks, [x*100 for x in before_res['ee_frac_by_k']], marker='o', color='C2', linestyle='--', label='EE before')

    plt.plot(ks, [x*100 for x in after_res['ss_frac_by_k']], marker='o', color='C0', label='SS after')
    plt.plot(ks, [x*100 for x in after_res['se_frac_by_k']], marker='o', color='C1', label='SE after')
    plt.plot(ks, [x*100 for x in after_res['ee_frac_by_k']], marker='o', color='C2', label='EE after')

    plt.xticks(ks)
    plt.xlabel('hops (k)')
    plt.ylabel('Reachable pairs (%)')
    plt.title(title_prefix + 'Reachability before vs after (<=k)')
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

    thresholds = [None, 2, 3]  # None -> no min_p
    rows = []
    for thr in thresholds:
        label = 'no_minp' if thr is None else f'minp_{thr}'
        print(f'\n=== Running {label} ===')
        G, cats = build_paper_graph(papers, min_papers_per_author=thr)
        print(f'Paper nodes={G.number_of_nodes()}, edges={G.number_of_edges()}')

        before = reachability_by_hops(G, cats, max_k=7)

        # rewire
        H = constrained_rewire_increase_within(G, cats, target_swaps_fraction=0.05, max_attempts_per_swap=50, seed=42)
        after = reachability_by_hops(H, cats, max_k=7)

        # save comparison plot
        out_prefix = f'reachability_paper_rewire_{label}_k7'
        plot_before_after(before, after, out_prefix, title_prefix=f'Paper ({label}) - ')

        # compute SE - AVG(SS,EE) before and after
        for when, res in [('before', before), ('after', after)]:
            ks = list(range(1, 8))
            diffs = []
            for i in range(len(ks)):
                se = res['se_frac_by_k'][i] * 100
                ss = res['ss_frac_by_k'][i] * 100
                ee = res['ee_frac_by_k'][i] * 100
                diffs.append(se - ((ss + ee) / 2.0))
            rows.append({'config': label, 'when': when, 'diffs': diffs, 'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()})

        print(f'Completed {label}; plot saved to {out_prefix}.png')

    # write CSV summary (k1..k7 columns)
    csv_path = 'rewire_paper_summary_k7.csv'
    fieldnames = ['config', 'when', 'nodes', 'edges'] + [f'k{i}' for i in range(1, 8)]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {'config': r['config'], 'when': r['when'], 'nodes': r['nodes'], 'edges': r['edges']}
            for i, v in enumerate(r['diffs'], start=1):
                row[f'k{i}'] = v
            writer.writerow(row)

    print('\nWrote summary to', csv_path)


if __name__ == '__main__':
    main()

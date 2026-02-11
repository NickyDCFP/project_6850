#!/usr/bin/env python3
"""Run Louvain (or fallback greedy modularity) community detection and report S/E homophily per community.

Produces CSV summary and a couple of PNGs per run.
"""
import argparse
import csv
import math
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import community as community_louvain  # python-louvain
    HAVE_LOUVAIN = True
except Exception:
    HAVE_LOUVAIN = False

from network_asp_analysis import (
    load_jsonl, normalize_author, build_paper_coauthorship_network,
    build_author_coauthorship_network
)


def run_louvain_or_fallback(G: nx.Graph):
    """Return partition dict node->community_id and number of communities.
    Tries python-louvain first; falls back to greedy_modularity_communities.
    """
    if HAVE_LOUVAIN:
        part = community_louvain.best_partition(G)
        # community ids might be non-contiguous; keep as-is
        return part
    else:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            coms = list(greedy_modularity_communities(G))
            part = {}
            for i, c in enumerate(coms):
                for n in c:
                    part[n] = i
            return part
        except Exception:
            # fallback: single community
            return {n: 0 for n in G.nodes()}


def analyze_partition(G: nx.Graph, categories: dict, partition: dict):
    """Compute per-community statistics: size, label composition, purity, internal homophily."""
    comm_to_nodes = defaultdict(list)
    for n, cid in partition.items():
        comm_to_nodes[cid].append(n)

    rows = []
    for cid, nodes in sorted(comm_to_nodes.items(), key=lambda x: -len(x[1])):
        labels = [categories.get(n, 'neither') for n in nodes]
        c = Counter(labels)
        size = len(nodes)
        # purity: fraction of nodes in dominant label
        dominant_label, dominant_count = c.most_common(1)[0]
        purity = dominant_count / size if size else 0.0

        # internal edges homophily: among edges entirely within community, fraction that are same-label
        internal_edges = 0
        same_label_internal = 0
        for u, v in G.subgraph(nodes).edges():
            internal_edges += 1
            if categories.get(u) == categories.get(v):
                same_label_internal += 1

        internal_homophily = same_label_internal / internal_edges if internal_edges else float('nan')

        rows.append({
            'community': cid,
            'size': size,
            'dominant_label': dominant_label,
            'dominant_count': dominant_count,
            'purity': purity,
            'internal_edges': internal_edges,
            'same_label_internal': same_label_internal,
            'internal_homophily': internal_homophily,
            'count_pure_safety': c.get('pure_safety', 0),
            'count_pure_ethics': c.get('pure_ethics', 0),
            'count_mixed': c.get('mixed', 0),
            'count_neither': c.get('neither', 0),
        })

    return rows


def plot_community_composition(rows, out_png):
    # stacked bar of top communities by size
    top = rows[:20]
    labels = [f"c{int(r['community'])}" for r in top]
    sizes = [r['size'] for r in top]
    safety = [r['count_pure_safety'] for r in top]
    ethics = [r['count_pure_ethics'] for r in top]
    mixed = [r['count_mixed'] for r in top]
    neither = [r['count_neither'] for r in top]

    ind = np.arange(len(top))
    plt.figure(figsize=(10, 5))
    p1 = plt.bar(ind, safety, label='pure_safety')
    p2 = plt.bar(ind, ethics, bottom=safety, label='pure_ethics')
    bottom_se = [s + e for s, e in zip(safety, ethics)]
    p3 = plt.bar(ind, mixed, bottom=bottom_se, label='mixed')
    bottom_all = [b + m for b, m in zip(bottom_se, mixed)]
    p4 = plt.bar(ind, neither, bottom=bottom_all, label='neither')
    plt.xticks(ind, labels, rotation=45, ha='right')
    plt.ylabel('nodes')
    plt.title('Top communities composition (by nodes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_purity_scatter(rows, out_png):
    x = [r['size'] for r in rows]
    y = [r['purity'] for r in rows]
    colors = []
    for r in rows:
        if r['dominant_label'] == 'pure_safety':
            colors.append('tab:blue')
        elif r['dominant_label'] == 'pure_ethics':
            colors.append('tab:orange')
        else:
            colors.append('tab:gray')

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, c=colors, alpha=0.8)
    plt.xscale('log')
    plt.xlabel('community size (log)')
    plt.ylabel('purity (dominant label fraction)')
    plt.title('Community purity vs size')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', choices=['paper', 'author'], default='paper')
    parser.add_argument('--minp', type=str, default='2', help='min papers per author (use 0 for no filter)')
    parser.add_argument('--top_n', type=int, default=0, help='optional: build top-N paper network (only for paper entity)')
    parser.add_argument('--out_prefix', type=str, default='louvain', help='output file prefix')
    args = parser.parse_args()

    # parse minp
    try:
        minp = int(args.minp)
    except Exception:
        minp = 2

    print(f"Loading combined data and building {args.entity} graph (minp={minp})...")
    from analyze_aggregated import load_combined_data
    papers = load_combined_data()

    # normalize authors similar to network_asp_analysis
    all_papers = []
    for p in papers:
        authors = []
        for a in p.get('authors', []):
            a = (a or '').strip()
            if not a:
                continue
            authors.append(normalize_author(a))
        p['authors'] = authors
        all_papers.append(p)

    if args.entity == 'paper':
        if args.top_n and args.top_n > 0:
            from network_asp_analysis import build_top_n_paper_network
            G, categories = build_top_n_paper_network(all_papers, top_n=args.top_n)
        else:
            G, categories, info = build_paper_coauthorship_network(all_papers, min_papers_per_author=minp)
    else:
        G, categories, info = build_author_coauthorship_network(all_papers, min_papers_per_author=minp)

    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print('Empty graph, exiting')
        return

    print('Detecting communities (Louvain or fallback) ...')
    partition = run_louvain_or_fallback(G)
    comm_rows = analyze_partition(G, categories, partition)

    # sort rows by size desc
    comm_rows.sort(key=lambda r: -r['size'])

    out_csv = f"{args.out_prefix}_{args.entity}_minp{minp}.csv"
    print(f"Writing CSV: {out_csv}")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comm_rows[0].keys()))
        writer.writeheader()
        for r in comm_rows:
            writer.writerow(r)

    # plots
    out_comp = f"{args.out_prefix}_{args.entity}_minp{minp}_topcomps.png"
    out_scatter = f"{args.out_prefix}_{args.entity}_minp{minp}_purity_scatter.png"
    plot_community_composition(comm_rows, out_comp)
    plot_purity_scatter(comm_rows, out_scatter)

    print('Done. Outputs:', out_csv, out_comp, out_scatter)


if __name__ == '__main__':
    main()

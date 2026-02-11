#!/usr/bin/env python3
"""
Analyze min cut and bridge connectivity in paper co-authorship networks.
"""
import sys
import random
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analyze_aggregated import load_combined_data

def load_paper_categories(papers):
    """Load paper categories based on safety/ethics scores."""
    categories = {}
    for i, p in enumerate(papers):
        s = p.get('safety_score', 0)
        e = p.get('ethics_score', 0)
        if s > 0 and e == 0:
            cat = 'pure_safety'
        elif e > 0 and s == 0:
            cat = 'pure_ethics'
        elif s > 0 and e > 0:
            cat = 'mixed'
        else:
            cat = 'neither'
        categories[i] = cat
    return categories

def load_paper_graph(papers, min_authors_per_paper=2):
    """Load the paper co-authorship graph."""
    G = nx.Graph()
    paper_categories = {}
    for i, p in enumerate(papers):
        authors = p.get('authors', [])
        if len(authors) < min_authors_per_paper:
            continue
            
        s = p.get('safety_score', 0)
        e = p.get('ethics_score', 0)
        if s > 0 and e == 0:
            cat = 'pure_safety'
        elif e > 0 and s == 0:
            cat = 'pure_ethics'
        elif s > 0 and e > 0:
            cat = 'mixed'
        else:
            cat = 'neither'
        G.add_node(i, category=cat)
        paper_categories[i] = cat
    
    # Add edges between papers with shared authors
    author_to_papers = defaultdict(list)
    for i, p in enumerate(papers):
        if i not in G.nodes():
            continue
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
    
    # Use largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        paper_categories = {n: paper_categories[n] for n in G.nodes()}
    
    return G, paper_categories

def bridge_fraction_by_papers(G_paper, paper_cats, top_papers, sample_target=1000):
    """Compute bridge fraction for papers."""
    S = [n for n,c in paper_cats.items() if c == 'pure_safety']
    E = [n for n,c in paper_cats.items() if c == 'pure_ethics']
    if not S or not E:
        return None
    valid = 0; thru = 0
    pairs = []
    total_possible = len(S)*len(E)
    if total_possible <= 200000:
        for s in S:
            for e in E:
                pairs.append((s,e))
    else:
        seen=set(); attempts=0
        while len(pairs) < 200000 and attempts < 200000*20:
            a=random.choice(S); b=random.choice(E)
            if (a,b) in seen: attempts+=1; continue
            seen.add((a,b)); pairs.append((a,b))
    for u,v in pairs:
        try:
            path = nx.shortest_path(G_paper, u, v)
        except nx.NetworkXNoPath:
            continue
        valid += 1
        internal = path[1:-1]
        if any(p in top_papers for p in internal):
            thru += 1
        if valid >= sample_target:
            break
    return thru / valid if valid else None

def compute_paper_nulls(G, cats, top_papers, n_shuffles=50, sample_target=1000):
    """Compute null distributions for paper bridge fraction."""
    observed = bridge_fraction_by_papers(G, cats, top_papers, sample_target=sample_target)
    if observed is None:
        return {'shuffle_fracs': []}
    
    shuffle_fracs = []
    for _ in range(n_shuffles):
        shuffled_cats = {n: random.choice(['pure_safety', 'pure_ethics']) for n in cats}
        null_result = bridge_fraction_by_papers(G, shuffled_cats, top_papers, sample_target=sample_target)
        if null_result is not None:
            shuffle_fracs.append(null_result)
    
    return {'shuffle_fracs': shuffle_fracs, 'observed': observed}

def main():
    print("Starting paper min cut and bridge analysis...")
    papers = load_combined_data()
    paper_categories = load_paper_categories(papers)
    G_paper, paper_cats = load_paper_graph(papers, min_authors_per_paper=2)
    
    print(f"Paper graph: {G_paper.number_of_nodes()} nodes, {G_paper.number_of_edges()} edges")
    safety_papers = [n for n,c in paper_cats.items() if c == 'pure_safety']
    ethics_papers = [n for n,c in paper_cats.items() if c == 'pure_ethics']
    print(f"Safety papers: {len(safety_papers)}, Ethics papers: {len(ethics_papers)}")
    
    # Min cut for papers
    print("Computing paper min cut...")
    sample_size = min(10, len(safety_papers), len(ethics_papers))
    safety_sample = random.sample(safety_papers, sample_size)
    ethics_sample = random.sample(ethics_papers, sample_size)
    
    cut_sizes = []
    cut_papers = []
    for s in safety_sample:
        for e in ethics_sample:
            if s != e:
                try:
                    cut = nx.minimum_node_cut(G_paper, s, e)
                    cut_sizes.append(len(cut))
                    cut_papers.extend(cut)
                except nx.NetworkXError:
                    continue
    
    if cut_sizes:
        avg_cut_size = np.mean(cut_sizes)
        unique_cut_papers = set(cut_papers)
        cut_composition = {}
        for p in unique_cut_papers:
            cat = paper_cats.get(p, 'neither')
            cut_composition[cat] = cut_composition.get(cat, 0) + 1
        
        total_papers = len(G_paper.nodes())
        percent = (len(unique_cut_papers) / total_papers) * 100
        
        print(f"Average paper min cut size: {avg_cut_size:.1f}")
        print(f"Unique papers in cuts: {len(unique_cut_papers)} ({percent:.1f}% of total)")
        print(f"Cut composition: {cut_composition}")
        
        # Plot paper min cut
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        labels = list(cut_composition.keys())
        sizes = list(cut_composition.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Paper Min Vertex Cut Composition\n({len(unique_cut_papers)} Papers, {percent:.1f}% of Total)')
        plt.tight_layout()
        fig.savefig(repo_root / 'paper_min_vertex_cut_analysis.pdf', dpi=300, bbox_inches='tight')
        print("Paper min cut figure saved.")
    
    # Bridge connectivity for papers
    print("Computing paper bridge connectivity...")
    degrees = dict(G_paper.degree())
    top_1pct_papers = set(sorted(degrees, key=degrees.get, reverse=True)[:int(0.01 * len(G_paper.nodes()))])
    top_5pct_papers = set(sorted(degrees, key=degrees.get, reverse=True)[:int(0.05 * len(G_paper.nodes()))])
    
    print(f"Top 1% papers: {len(top_1pct_papers)}")
    print(f"Top 5% papers: {len(top_5pct_papers)}")
    
    bf_1pct = bridge_fraction_by_papers(G_paper, paper_cats, top_1pct_papers, sample_target=1000)
    bf_5pct = bridge_fraction_by_papers(G_paper, paper_cats, top_5pct_papers, sample_target=1000)
    
    print(f"Paper bridge fraction top 1%: {bf_1pct:.3f}")
    print(f"Paper bridge fraction top 5%: {bf_5pct:.3f}")
    
    # Nulls
    print("Computing paper nulls...")
    nulls_1pct = compute_paper_nulls(G_paper, paper_cats, top_1pct_papers, n_shuffles=50, sample_target=1000)
    nulls_5pct = compute_paper_nulls(G_paper, paper_cats, top_5pct_papers, n_shuffles=50, sample_target=1000)
    
    # Plot
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    labels = ['Top 1%', 'Top 5%']
    observed = [bf_1pct, bf_5pct]
    shuffle_means = [np.mean(nulls_1pct['shuffle_fracs']), np.mean(nulls_5pct['shuffle_fracs'])]
    shuffle_stds = [np.std(nulls_1pct['shuffle_fracs']), np.std(nulls_5pct['shuffle_fracs'])]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, observed, width, label='Observed', color='skyblue')
    ax.bar(x + width/2, shuffle_means, width, yerr=shuffle_stds, label='Label-Shuffle Null', color='orange', alpha=0.7, capsize=5)
    
    ax.set_xlabel('Top Papers (%)')
    ax.set_ylabel('Bridge Fraction')
    ax.set_title('Paper Bridge Concentration (minp=2)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = repo_root / 'paper_bridge_connectivity.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Paper bridge plot saved to {plot_path}")

if __name__ == '__main__':
    main()
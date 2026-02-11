#!/usr/bin/env python3
"""
Unified homophily removal analysis for author and paper networks.
Creates side-by-side subplots with shared legend.
"""
import sys
import json
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analyze_aggregated import load_combined_data

def load_author_categories(papers):
    """Load author categories based on their papers."""
    author_scores = defaultdict(lambda: {'safety': 0, 'ethics': 0, 'total': 0})
    for p in papers:
        s = p.get('safety_score', 0)
        e = p.get('ethics_score', 0)
        for a in p.get('authors', []):
            author_scores[a]['safety'] += s
            author_scores[a]['ethics'] += e
            author_scores[a]['total'] += 1
    
    categories = {}
    for a, scores in author_scores.items():
        if scores['total'] >= 2:  # min_p
            if scores['safety'] > 0 and scores['ethics'] == 0:
                categories[a] = 'pure_safety'
            elif scores['ethics'] > 0 and scores['safety'] == 0:
                categories[a] = 'pure_ethics'
            else:
                categories[a] = 'mixed'
    return categories

def load_author_graph(papers, perturbation=None, removal_sequence=None):
    """Load the author coauthorship graph using old logic (transitive connections)."""
    # Build author-paper mapping
    author_papers = defaultdict(list)
    for p in papers:
        pid = p.get('paper_id') or p.get('id') or str(hash(str(p)))
        for a in p.get('authors', []):
            author_papers[a].append(pid)
    
    # Filter authors with >= min_p papers
    authors = [a for a, ps in author_papers.items() if len(ps) >= 2]
    
    # If perturbation, remove authors
    if perturbation and removal_sequence is not None:
        removed_authors = set(removal_sequence[:perturbation])
        authors = [a for a in authors if a not in removed_authors]
    
    # Build graph: authors as nodes, edges weighted by coauthored papers (old logic)
    G = nx.Graph()
    author_pairs = defaultdict(int)
    for ps in author_papers.values():
        for i in range(len(ps)):
            for j in range(i+1, len(ps)):
                p1_authors = [a for a in author_papers if ps[i] in author_papers[a]]
                p2_authors = [a for a in author_papers if ps[j] in author_papers[a]]
                for a1 in p1_authors:
                    for a2 in p2_authors:
                        if a1 in authors and a2 in authors and a1 != a2:
                            pair = tuple(sorted([a1, a2]))
                            author_pairs[pair] += 1
    
    for pair, weight in author_pairs.items():
        G.add_edge(pair[0], pair[1], weight=weight)
    
    # Use largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G

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

def load_paper_graph(papers, perturbation=None, removal_sequence=None):
    """Load the paper coauthorship graph, optionally with perturbation."""
    # If perturbation, remove papers
    if perturbation and removal_sequence is not None:
        removed_papers = set(removal_sequence[:perturbation])
    else:
        removed_papers = set()
    
    # Build paper graph using original indices
    G = nx.Graph()
    paper_categories = {}
    for i, p in enumerate(papers):
        if i in removed_papers:
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
        if i in removed_papers:
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
        # Filter categories to only include nodes in largest component
        paper_categories = {n: paper_categories[n] for n in G.nodes()}
    
    return G, paper_categories

def get_removal_sequences(papers, categories):
    """Get removal sequences for different methods."""
    # Aggregate scores
    author_agg_scores = {}
    for p in papers:
        s = p.get('safety_score', 0)
        e = p.get('ethics_score', 0)
        for a in p.get('authors', []):
            if a not in author_agg_scores:
                author_agg_scores[a] = {'safety': 0, 'ethics': 0}
            author_agg_scores[a]['safety'] += s
            author_agg_scores[a]['ethics'] += e
    
    # Build full graph to get degrees
    G_full = load_author_graph(papers, perturbation=0)
    degrees = dict(G_full.degree())
    
    # Degree-based removal: sort by degree descending
    degree_sequence = sorted(degrees.keys(), key=lambda a: degrees[a], reverse=True)
    
    return {'degree': degree_sequence}

def get_paper_removal_sequences(papers):
    """Get removal sequences for papers by degree."""
    # Build full graph to get degrees using original indices
    G, _ = load_paper_graph(papers)
    
    # Get degrees
    degrees = dict(G.degree())
    
    # Sort by degree descending
    removal_sequence = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    return {'degree': removal_sequence}

def compute_author_homophily(G, categories):
    """Compute homophily metrics excluding mixed authors."""
    ss = 0  # pure_safety to pure_safety
    ee = 0  # pure_ethics to pure_ethics
    se = 0  # pure_safety to pure_ethics
    
    for u, v in G.edges(data=False):
        cu = categories.get(u, 'none')
        cv = categories.get(v, 'none')
        weight = 1
        if cu == 'pure_safety' and cv == 'pure_safety':
            ss += weight
        elif cu == 'pure_ethics' and cv == 'pure_ethics':
            ee += weight
        elif (cu == 'pure_safety' and cv == 'pure_ethics') or (cu == 'pure_ethics' and cv == 'pure_safety'):
            se += weight
    
    total_pure_edges = ss + ee + se
    global_homophily = (ss + ee) / total_pure_edges if total_pure_edges else 0.0
    safety_homophily = ss / (ss + se) if (ss + se) else 0.0
    ethics_homophily = ee / (ee + se) if (ee + se) else 0.0
    
    return {
        'global_homophily': global_homophily,
        'safety_homophily': safety_homophily,
        'ethics_homophily': ethics_homophily
    }

def compute_paper_homophily(G, categories):
    """Compute homophily metrics excluding mixed authors, using weighted edges."""
    ss = 0.0  # pure_safety to pure_safety
    ee = 0.0  # pure_ethics to pure_ethics
    se = 0.0  # pure_safety to pure_ethics
    
    for u, v, d in G.edges(data=True):
        cu = categories.get(u, 'neither')
        cv = categories.get(v, 'neither')
        weight = d.get('weight', 1.0)
        if cu == 'pure_safety' and cv == 'pure_safety':
            ss += weight
        elif cu == 'pure_ethics' and cv == 'pure_ethics':
            ee += weight
        elif (cu == 'pure_safety' and cv == 'pure_ethics') or (cu == 'pure_ethics' and cv == 'pure_safety'):
            se += weight
    
    total_pure_weight = ss + ee + se
    global_homophily = (ss + ee) / total_pure_weight if total_pure_weight else 0.0
    safety_homophily = ss / (ss + se) if (ss + se) else 0.0
    ethics_homophily = ee / (ee + se) if (ee + se) else 0.0
    
    return {
        'global_homophily': global_homophily,
        'safety_homophily': safety_homophily,
        'ethics_homophily': ethics_homophily
    }

def compute_null_homophily(G, categories, n_shuffles=5, n_rewires=2):
    """Compute null homophily via label shuffling and graph rewiring."""
    nodes = list(G.nodes())
    orig_labels = [categories.get(n, 'neither') for n in nodes]
    
    # Label shuffle
    shuffle_hom = []
    for _ in range(n_shuffles):
        shuffled_labels = np.random.permutation(orig_labels)
        shuffled_cat = {n: lab for n, lab in zip(nodes, shuffled_labels)}
        h = compute_author_homophily(G, shuffled_cat)
        shuffle_hom.append(h)
    
    # Rewire
    rewire_hom = []
    for _ in range(n_rewires):
        H = G.copy()
        nx.double_edge_swap(H, nswap=1000, max_tries=10000)
        h = compute_author_homophily(H, categories)
        rewire_hom.append(h)
    
    return {
        'shuffle_hom': shuffle_hom,
        'rewire_hom': rewire_hom
    }

def compute_null_paper_homophily(G, categories, n_shuffles=5, n_rewires=2):
    """Compute null homophily via label shuffling and graph rewiring."""
    nodes = list(G.nodes())
    orig_labels = [categories.get(n, 'neither') for n in nodes]
    
    # Label shuffle
    shuffle_hom = []
    for _ in range(n_shuffles):
        shuffled_labels = np.random.permutation(orig_labels)
        shuffled_cat = {n: lab for n, lab in zip(nodes, shuffled_labels)}
        h = compute_paper_homophily(G, shuffled_cat)
        shuffle_hom.append(h)
    
    # Rewire
    rewire_hom = []
    for _ in range(n_rewires):
        H = G.copy()
        nx.double_edge_swap(H, nswap=1000, max_tries=10000)
        h = compute_paper_homophily(H, categories)
        rewire_hom.append(h)
    
    return {
        'shuffle_hom': shuffle_hom,
        'rewire_hom': rewire_hom
    }

def main():
    print("Starting unified homophily removal analysis...")
    papers = load_combined_data()
    author_categories = load_author_categories(papers)
    paper_categories = load_paper_categories(papers)
    
    author_removal_sequences = get_removal_sequences(papers, author_categories)
    paper_removal_sequences = get_paper_removal_sequences(papers)
    
    levels = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    
    author_results = []
    author_null_results = []
    paper_results = []
    paper_null_results = []
    
    for level in levels:
        print(f"Processing level {level}")
        
        # Author network
        author_G = load_author_graph(papers, perturbation=level, removal_sequence=author_removal_sequences['degree'])
        author_obs_stats = compute_author_homophily(author_G, author_categories)
        
        if level == 0:
            print(f"Author baseline homophily: global={author_obs_stats['global_homophily']:.3f}, safety={author_obs_stats['safety_homophily']:.3f}, ethics={author_obs_stats['ethics_homophily']:.3f}")
        
        author_null_stats = compute_null_homophily(author_G, author_categories, n_shuffles=5, n_rewires=2)
        author_avg_null_global = np.mean([h['global_homophily'] for h in author_null_stats['shuffle_hom']])
        author_avg_null_safety = np.mean([h['safety_homophily'] for h in author_null_stats['shuffle_hom']])
        author_avg_null_ethics = np.mean([h['ethics_homophily'] for h in author_null_stats['shuffle_hom']])
        
        author_row = [
            level,
            author_obs_stats['global_homophily'], author_obs_stats['safety_homophily'], author_obs_stats['ethics_homophily']
        ]
        author_results.append(author_row)
        
        author_null_row = [level, author_avg_null_global, author_avg_null_safety, author_avg_null_ethics]
        author_null_results.append(author_null_row)
        
        # Save incremental results
        pd.DataFrame(author_results, columns=['level', 'global', 'safety', 'ethics']).to_csv(repo_root / 'author_homophily_results.csv', index=False)
        pd.DataFrame(author_null_results, columns=['level', 'global_null', 'safety_null', 'ethics_null']).to_csv(repo_root / 'author_null_homophily_results.csv', index=False)
        pd.DataFrame(paper_results, columns=['level', 'global', 'safety', 'ethics']).to_csv(repo_root / 'paper_homophily_results.csv', index=False)
        pd.DataFrame(paper_null_results, columns=['level', 'global_null', 'safety_null', 'ethics_null']).to_csv(repo_root / 'paper_null_homophily_results.csv', index=False)
        
        # Paper network
        paper_G, paper_cats = load_paper_graph(papers, perturbation=level, removal_sequence=paper_removal_sequences['degree'])
        paper_obs_stats = compute_paper_homophily(paper_G, paper_cats)
        
        if level == 0:
            print(f"Paper baseline homophily: global={paper_obs_stats['global_homophily']:.3f}, safety={paper_obs_stats['safety_homophily']:.3f}, ethics={paper_obs_stats['ethics_homophily']:.3f}")
        
        paper_null_stats = compute_null_paper_homophily(paper_G, paper_cats, n_shuffles=5, n_rewires=2)
        paper_avg_null_global = np.mean([h['global_homophily'] for h in paper_null_stats['rewire_hom']])
        paper_avg_null_safety = np.mean([h['safety_homophily'] for h in paper_null_stats['rewire_hom']])
        paper_avg_null_ethics = np.mean([h['ethics_homophily'] for h in paper_null_stats['rewire_hom']])
        
        paper_row = [
            level,
            paper_obs_stats['global_homophily'], paper_obs_stats['safety_homophily'], paper_obs_stats['ethics_homophily']
        ]
        paper_results.append(paper_row)
        
        paper_null_row = [level, paper_avg_null_global, paper_avg_null_safety, paper_avg_null_ethics]
        paper_null_results.append(paper_null_row)
    
    # Plot
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Author plot
    ax1 = axes[0]
    author_df_obs = pd.DataFrame(author_results, columns=['level', 'global', 'safety', 'ethics'])
    author_df_null = pd.DataFrame(author_null_results, columns=['level', 'global_null', 'safety_null', 'ethics_null'])
    
    ax1.plot(author_df_obs['level'], author_df_obs['global'], label='Global Homophily*', color='purple', linewidth=2)
    ax1.plot(author_df_obs['level'], author_df_obs['safety'], label='Safety Homophily*', color='red', linewidth=2)
    ax1.plot(author_df_obs['level'], author_df_obs['ethics'], label='Ethics Homophily*', color='blue', linewidth=2)
    
    ax1.plot(author_df_null['level'], author_df_null['global_null'], color='purple', linewidth=2, linestyle='--', alpha=0.7)
    ax1.plot(author_df_null['level'], author_df_null['safety_null'], color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax1.plot(author_df_null['level'], author_df_null['ethics_null'], color='blue', linewidth=2, linestyle='--', alpha=0.7)
    
    # Grey dotted line for legend
    ax1.plot([], [], color='grey', linewidth=2, linestyle='--', alpha=0.7, label='Degree-Preserving Rewire Null Model')
    
    ax1.set_xlabel('Number of Authors Removed (by Degree)')
    ax1.set_ylabel('Homophily')
    ax1.set_title('Author Network')
    ax1.set_xticks(np.arange(0, 450, 50))
    ax1.set_yticks(np.arange(0.6, 0.9, 0.1))
    ax1.grid(True, alpha=0.3)
    
    # Paper plot
    ax2 = axes[1]
    paper_df_obs = pd.DataFrame(paper_results, columns=['level', 'global', 'safety', 'ethics'])
    paper_df_null = pd.DataFrame(paper_null_results, columns=['level', 'global_null', 'safety_null', 'ethics_null'])
    
    ax2.plot(paper_df_obs['level'], paper_df_obs['global'], label='Global Homophily*', color='purple', linewidth=2)
    ax2.plot(paper_df_obs['level'], paper_df_obs['safety'], label='Safety Homophily*', color='red', linewidth=2)
    ax2.plot(paper_df_obs['level'], paper_df_obs['ethics'], label='Ethics Homophily*', color='blue', linewidth=2)
    
    ax2.plot(paper_df_null['level'], paper_df_null['global_null'], color='purple', linewidth=2, linestyle='--', alpha=0.7)
    ax2.plot(paper_df_null['level'], paper_df_null['safety_null'], color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax2.plot(paper_df_null['level'], paper_df_null['ethics_null'], color='blue', linewidth=2, linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Number of Papers Removed (by Degree)')
    ax2.set_ylabel('Homophily')
    ax2.set_title('Paper Network')
    ax2.set_xticks(np.arange(0, 450, 50))
    ax2.set_yticks(np.arange(0.4, 0.85, 0.1))
    ax2.grid(True, alpha=0.3)
    
    # Shared legend - combine labels from both plots
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Use ax1 labels since they should be the same
    fig.legend(handles1, labels1, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=True)
    
    plt.tight_layout()
    plot_path = repo_root / 'unified_homophily_degree_removal.pdf'
    #plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    # Save results to CSV
    author_df_obs.to_csv(repo_root / 'author_homophily_results.csv', index=False)
    author_df_null.to_csv(repo_root / 'author_null_homophily_results.csv', index=False)
    paper_df_obs.to_csv(repo_root / 'paper_homophily_results.csv', index=False)
    paper_df_null.to_csv(repo_root / 'paper_null_homophily_results.csv', index=False)
    print("Results saved to CSV files")

if __name__ == '__main__':
    main()
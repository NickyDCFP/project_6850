#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import networkx as nx
from collections import defaultdict

# Add repo root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from network_asp_analysis import sample_weighted_distances, load_filtered_graph

def compute_weighted_asp_for_level(level, min_p=2):
    print(f"Computing weighted ASP for {level}...")
    G = load_filtered_graph(min_p=min_p, perturbation=level)
    
    # Load author categories
    author_cats = {}
    with open(repo_root / 'data/final_dataset/filtered_ethics_safety_papers.jsonl', 'r') as f:
        for line in f:
            import json
            rec = json.loads(line)
            for author in rec.get('authors', []):
                author_cats[author] = rec.get('category', 'unknown')
    
    # Filter to authors in graph
    author_cats = {a: c for a, c in author_cats.items() if a in G.nodes()}
    
    # Sample distances
    results = sample_weighted_distances(G, author_cats, n_samples=500, seed=42)
    
    return {
        'ss_mean': results['ss']['mean_dist'],
        'ss_median': results['ss']['median_dist'],
        'ee_mean': results['ee']['mean_dist'],
        'ee_median': results['ee']['median_dist'],
        'se_mean': results['se']['mean_dist'],
        'se_median': results['se']['median_dist'],
        'ss_median_hops': results['ss']['median_hops'],
        'ee_median_hops': results['ee']['median_hops'],
        'se_median_hops': results['se']['median_hops']
    }

# Levels
levels = ['baseline', 'top1pct', 'top5pct', 'top10pct']

results = {}
for level in levels:
    if level == 'baseline':
        # Load from existing
        df = pd.read_csv(repo_root / 'perturb_grid_out/weighted_homophily_minp2.csv')
        results[level] = {
            'ss_mean': df[df['metric'] == 'ss_mean_dist']['value'].iloc[0],
            'ss_median': df[df['metric'] == 'ss_median_dist']['value'].iloc[0],
            'ee_mean': df[df['metric'] == 'ee_mean_dist']['value'].iloc[0],
            'ee_median': df[df['metric'] == 'ee_median_dist']['value'].iloc[0],
            'se_mean': df[df['metric'] == 'se_mean_dist']['value'].iloc[0],
            'se_median': df[df['metric'] == 'se_median_dist']['value'].iloc[0],
            'ss_median_hops': 6,  # assumed
            'ee_median_hops': 6,
            'se_median_hops': 6
        }
    else:
        results[level] = compute_weighted_asp_for_level(level)

# Compute deltas
for level, data in results.items():
    mean_ss_ee = (data['ss_mean'] + data['ee_mean']) / 2
    median_ss_ee = (data['ss_median'] + data['ee_median']) / 2
    hops_ss_ee = (data['ss_median_hops'] + data['ee_median_hops']) / 2
    
    data['delta_mean'] = data['se_mean'] - mean_ss_ee
    data['delta_median'] = data['se_median'] - median_ss_ee
    data['delta_hops'] = data['se_median_hops'] - hops_ss_ee

# Print results
for level in levels:
    print(f"\n{level.upper()}:")
    data = results[level]
    print(f"  SS mean: {data['ss_mean']:.3f}, median: {data['ss_median']:.3f}, hops: {data['ss_median_hops']}")
    print(f"  EE mean: {data['ee_mean']:.3f}, median: {data['ee_median']:.3f}, hops: {data['ee_median_hops']}")
    print(f"  SE mean: {data['se_mean']:.3f}, median: {data['se_median']:.3f}, hops: {data['se_median_hops']}")
    print(f"  Delta mean: {data['delta_mean']:.3f}, median: {data['delta_median']:.3f}, hops: {data['delta_hops']:.1f}")

# Generate plot
import matplotlib.pyplot as plt

levels_labels = ['Baseline', '1%', '5%', '10%']
deltas_mean = [results['baseline']['delta_mean'], results['top1pct']['delta_mean'], results['top5pct']['delta_mean'], results['top10pct']['delta_mean']]
deltas_median = [results['baseline']['delta_median'], results['top1pct']['delta_median'], results['top5pct']['delta_median'], results['top10pct']['delta_median']]
deltas_hops = [results['baseline']['delta_hops'], results['top1pct']['delta_hops'], results['top5pct']['delta_hops'], results['top10pct']['delta_hops']]

plt.figure(figsize=(10, 6))
plt.plot(levels_labels, deltas_mean, marker='o', label='Mean Weighted Distance')
plt.plot(levels_labels, deltas_median, marker='s', label='Median Weighted Distance')
plt.plot(levels_labels, deltas_hops, marker='^', label='Median Hops')
plt.xlabel('Removal Level')
plt.ylabel('Delta (SE - mean(SS, EE))')
plt.title('Weighted ASP Deltas Across Perturbation Levels')
plt.legend()
plt.grid(True)
plt.savefig(repo_root / 'weighted_asp_deltas.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as weighted_asp_deltas.png")
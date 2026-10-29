#!/usr/bin/env python3
"""
Aggregate analysis combining ACL Anthology and DBLP conference datasets.
Normalizes author names (removing DBLP numeric disambiguators),
computes yearly trends, and builds colored paper and co-authorship networks.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List
from glob import glob
import matplotlib.pyplot as plt
import networkx as nx




def normalize_author(name: str) -> str:
    """Remove DBLP disambiguation numbers like 'Smith, John 0001' or 'John Smith 2'."""
    if not name:
        return name
    s = name.strip()
    # Remove trailing numeric tokens
    parts = s.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    # Remove zero-padded numeric suffixes
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 3:
        parts = parts[:-1]
    return ' '.join(parts)


def load_papers_data(filepath: str) -> List[Dict]:
    """Load papers data from JSONL file (ACL or DBLP)."""
    papers = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    return papers

load_jsonl = load_papers_data


FINAL_FILTERED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'final_filtered')


def load_combined_data(corpus: str = 'full') -> List[Dict]:
    """Load and combine ACL and DBLP data.

    corpus='full'   -> all 8 files in data/final_filtered (6,442 papers, as described in the paper).
    corpus='legacy' -> the original glob '*_filtered.jsonl', which silently skips
                       filtered_acl_anthology.jsonl and filtered_neurips.jsonl (3,248 papers).
                       The paper's Table 2 was produced on this subset.
    """
    patterns = {'full': '*.jsonl', 'legacy': '*_filtered.jsonl'}
    if corpus not in patterns:
        raise ValueError(f"corpus must be one of {sorted(patterns)}, got {corpus!r}")
    all_papers = []
    for filepath in sorted(glob(os.path.join(FINAL_FILTERED_DIR, patterns[corpus]))):
        papers = load_papers_data(filepath)
        # Files are named either '<conf>_filtered.jsonl' or 'filtered_<conf>[_anthology].jsonl'
        stem = os.path.basename(filepath).split('.')[0].split('_')
        source = stem[1] if stem[0] == 'filtered' else stem[0]
        for paper in papers:
            paper['source_file'] = source
            if paper['source_file'] in ['aies', 'fat', 'satml']:
                if paper['ethics_score'] == 0 and paper['safety_score'] == 0:
                    if paper['source_file'] == 'satml':
                        paper['safety_score'] += 1
                    else:
                        paper['ethics_score'] += 1
            paper['category'] = 'both' if paper['safety_score'] > 0 and paper['ethics_score'] > 0 else \
                                'safety' if paper['safety_score'] > 0 else \
                                'ethics' if paper['ethics_score'] > 0 else 'none'
            paper['year'] = int(paper.get('year'))
            if paper['category'] == 'none':
                print(paper)
                raise AssertionError("Paper with neither safety nor ethics score found:")
        all_papers.extend(papers)
    return all_papers


def analyze_yearly_trends(papers: List[Dict]) -> Dict:
    yearly = defaultdict(lambda: {'total': 0, 'safety': 0, 'ethics': 0, 'both': 0})
    for p in papers:
        y = p.get('year') or 0
        try:
            y = int(y)
        except Exception:
            continue
        cat = p.get('category', 'neither')
        yearly[y]['total'] += 1
        if cat == 'safety':
            yearly[y]['safety'] += 1
        elif cat == 'ethics':
            yearly[y]['ethics'] += 1
        elif cat == 'both':
            yearly[y]['both'] += 1
    return dict(yearly)


def plot_yearly_trends(yearly_stats: Dict, out_path: str):
    years = sorted(yearly_stats.keys())
    safety = [yearly_stats[y]['safety'] for y in years]
    ethics = [yearly_stats[y]['ethics'] for y in years]
    both = [yearly_stats[y]['both'] for y in years]
    total = [yearly_stats[y]['total'] for y in years]

    plt.figure(figsize=(12, 8))
    plt.bar(years, safety, color='red', alpha=0.7, label='Safety')
    plt.bar(years, ethics, bottom=safety, color='blue', alpha=0.7, label='Ethics')
    plt.bar(years, both, bottom=[s+e for s, e in zip(safety, ethics)], color='purple', alpha=0.7, label='Both')
    for y, t in zip(years, total):
        plt.text(y, t + 3, str(t), ha='center', va='bottom', fontsize=9)
    plt.title('Combined: Safety vs Ethics Papers by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_colored_paper_network(papers: List[Dict], out_path: str, top_n: int = 400, bins: int = 20):
    # Filter and rank
    scored = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]
    scored.sort(key=lambda p: max(p.get('safety_score', 0) or 0, p.get('ethics_score', 0) or 0), reverse=True)
    sel = scored[:top_n]
    if not sel:
        return

    G = nx.Graph()
    for i, p in enumerate(sel):
        G.add_node(i, title=p.get('title', ''), s=float(p.get('safety_score', 0) or 0), e=float(p.get('ethics_score', 0) or 0))

    author_to_papers = defaultdict(list)
    for i, p in enumerate(sel):
        for a in p.get('authors', []):
            if a:
                author_to_papers[a].append(i)
    for plist in author_to_papers.values():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            for j in range(i+1, len(plist)):
                u, v = plist[i], plist[j]
                G.add_edge(u, v, weight=1 if not G.has_edge(u, v) else G[u][v]['weight'] + 1)

    if G.number_of_edges() == 0:
        return

    pos = nx.spring_layout(G, k=2.5, iterations=120, seed=42)
    # Force horizontal position strictly by safety proportion using bins
    ratios = {}
    for n in G.nodes:
        s = G.nodes[n]['s']
        e = G.nodes[n]['e']
        t = s + e
        # safety proportion in [0,1], map to [-1,1]
        safety_prop = 0.5 if t == 0 else (s / t)
        ratios[n] = safety_prop
    # Bin and set x to bin centers strictly
    for n, p in pos.items():
        r = ratios.get(n, 0.5)
        bin_idx = max(0, min(bins - 1, int(r * bins)))
        center = (bin_idx + 0.5) / bins  # 0..1
        forced_x = -1.0 + 2.0 * center   # -1..1 left..right
        p[0] = forced_x
    # Jitter positions to reduce overlap
    jitter = 0.05
    for n, p in pos.items():
        p[0] += random.uniform(-jitter, jitter)
        p[1] += random.uniform(-jitter, jitter)
    colors = []
    sizes = []
    for n in G.nodes:
        s = G.nodes[n]['s']
        e = G.nodes[n]['e']
        t = s + e
        if t > 0:
            r = s / t
            b = e / t
            colors.append((r, 0.0, b, 0.9))
        else:
            colors.append((0.7, 0.7, 0.7, 0.6))
        sizes.append(100 + 200 * t)

    plt.figure(figsize=(18, 14))
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=0.8, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors='black', linewidths=0.5)
    plt.title('Combined Paper Network Colored by Safety (red) vs Ethics (blue)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_author_network_colored(papers: List[Dict], out_path: str, top_n: int = 300, bins: int = 20):
    """Build an author co-authorship network and color authors by aggregated
    safety (red) vs ethics (blue) ratio. Node size scales with total score and degree.
    """
    # Aggregate scores and papers per author
    author_scores = defaultdict(lambda: {'s': 0.0, 'e': 0.0, 'papers': set()})
    for idx, p in enumerate(papers):
        s = float(p.get('safety_score', 0) or 0)
        e = float(p.get('ethics_score', 0) or 0)
        for a in p.get('authors', []):
            if not a:
                continue
            author_scores[a]['s'] += s
            author_scores[a]['e'] += e
            author_scores[a]['papers'].add(idx)

    # Build graph (exclude authors with zero total score => 'neither')
    G = nx.Graph()
    included = set()
    for a, se in author_scores.items():
        s_val = float(se['s'])
        e_val = float(se['e'])
        if (s_val + e_val) <= 0.0:
            continue
        G.add_node(a, s=s_val, e=e_val, npapers=len(se['papers']))
        included.add(a)

    # Co-authorship edges
    paper_to_authors = defaultdict(list)
    for a, se in author_scores.items():
        if a not in included:
            continue
        for pid in se['papers']:
            paper_to_authors[pid].append(a)
    for plist in paper_to_authors.values():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                u, v = plist[i], plist[j]
                if u == v:
                    continue
                if G.has_edge(u, v):
                    G[u][v]['w'] += 1
                else:
                    G.add_edge(u, v, w=1)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return

    # Select top authors by degree for readability
    deg = dict(G.degree())
    top_nodes = [n for n, _ in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    H = G.subgraph(top_nodes).copy()
    if H.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(H, k=2.5, iterations=120, seed=42)
    # Force horizontal position strictly by safety proportion using bins
    ratios = {}
    for n in H.nodes:
        s = float(H.nodes[n].get('s', 0.0))
        e = float(H.nodes[n].get('e', 0.0))
        t = s + e
        safety_prop = 0.5 if t == 0 else (s / t)
        ratios[n] = safety_prop
    for n, p in pos.items():
        r = ratios.get(n, 0.5)
        bin_idx = max(0, min(bins - 1, int(r * bins)))
        center = (bin_idx + 0.5) / bins
        forced_x = -1.0 + 2.0 * center
        p[0] = forced_x
    # Jitter positions to reduce overlap
    jitter = 0.05
    for n, p in pos.items():
        p[0] += random.uniform(-jitter, jitter)
        p[1] += random.uniform(-jitter, jitter)
    colors = []
    sizes = []
    for n in H.nodes:
        s = float(H.nodes[n].get('s', 0.0))
        e = float(H.nodes[n].get('e', 0.0))
        t = s + e
        if t > 0:
            r = s / t
            b = e / t
            colors.append((r, 0.0, b, 0.9))
        else:
            colors.append((0.7, 0.7, 0.7, 0.6))
        sizes.append(60 + 15 * deg.get(n, 1) + 8 * t)

    plt.figure(figsize=(18, 14))
    nx.draw_networkx_edges(H, pos, alpha=0.4, width=[0.6 + 0.3 * d.get('w', 1) for _, _, d in H.edges(data=True)], edge_color='gray')
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=sizes, edgecolors='black', linewidths=0.4)
    # No labels for a cleaner view
    plt.title('Combined Author Co-authorship Network (Red=Safety, Blue=Ethics)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    combined = load_combined_data()

    yearly = analyze_yearly_trends(combined)
    plot_yearly_trends(yearly, 'combined_yearly_trends.png')
    plot_colored_paper_network(combined, 'combined_paper_network_colored.png', top_n=1000)
    plot_author_network_colored(combined, 'combined_author_network_colored.png', top_n=1000)

    print('Combined analysis complete:')
    print(' - combined_yearly_trends.png')
    print(' - combined_paper_network_colored.png')
    print(' - combined_author_network_colored.png')


if __name__ == '__main__':
    main()



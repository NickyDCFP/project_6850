#!/usr/bin/env python3
"""
Refactored semantic convergence analysis with embedding caching, sampling and fast defaults.

Usage examples:
  # Quick smoke test (200 papers)
  python3 semantic_convergence_refactor.py --sample-size 200 --method kmeans --visualize

  # Full run (no sampling) - beware of runtime
  python3 semantic_convergence_refactor.py --no-sample --model all-MiniLM-L6-v2
"""

import argparse
import json
import os
import random
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import time

CACHE_DIR = 'data/cache'
os.makedirs(CACHE_DIR, exist_ok=True)


def load_jsonl(filepath: str) -> List[Dict]:
    papers = []
    if not os.path.exists(filepath):
        return papers
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    return papers


def create_text(paper: Dict) -> str:
    title = paper.get('title', '').strip()
    abstract = paper.get('abstract', '').strip()
    if abstract:
        return title + '. ' + abstract
    else:
        print("Abstract missing")
    return title


class RefactorAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f'Loading embedding model: {model_name}')
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.papers = None

    def compute_embeddings(self, papers: List[Dict], cache_name: str = None, batch_size: int = 64) -> np.ndarray:
        texts = [create_text(p) for p in papers]
        if cache_name:
            cache_path = os.path.join(CACHE_DIR, cache_name)
            if os.path.exists(cache_path):
                print(f'Loading embeddings cache: {cache_path}')
                return np.load(cache_path)
        print(f'Computing embeddings for {len(texts)} texts...')
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        if cache_name:
            np.save(os.path.join(CACHE_DIR, cache_name), embs)
        return embs

    def cluster(self, embeddings: np.ndarray, method: str = 'kmeans', n_clusters: int = 20, knn: int = 15):
        print(f'Clustering using {method}...')
        if method == 'kmeans':
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(embeddings)
        elif method == 'hierarchical':
            cl = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
            labels = cl.fit_predict(embeddings)
        elif method == 'dbscan':
            cl = DBSCAN(metric='cosine', eps=0.35, min_samples=5)
            labels = cl.fit_predict(embeddings)
        elif method == 'louvain':
            # Build a k-NN graph (weighted by cosine similarity) and run Louvain community detection
            try:
                import networkx as nx
                from sklearn.neighbors import NearestNeighbors
                # python-louvain module is typically installed as `python-louvain` and imported as `community`
                try:
                    import community as community_louvain
                except Exception:
                    # some installs expose community_louvain
                    import community.community_louvain as community_louvain
            except Exception as e:
                raise ImportError("To use louvain clustering please install dependencies: pip install python-louvain networkx scikit-learn") from e

            n_neighbors = max(2, int(knn))
            nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, embeddings.shape[0]-1), metric='cosine')
            nbrs.fit(embeddings)
            distances, indices = nbrs.kneighbors(embeddings)

            G = nx.Graph()
            n = embeddings.shape[0]
            G.add_nodes_from(range(n))
            for i in range(n):
                for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                    # cosine distance -> similarity
                    sim = 1.0 - float(dist)
                    if sim <= 0:
                        continue
                    if G.has_edge(i, int(j_idx)):
                        # keep max weight
                        if G[i][int(j_idx)].get('weight', 0) < sim:
                            G[i][int(j_idx)]['weight'] = sim
                    else:
                        G.add_edge(i, int(j_idx), weight=sim)

            partition = community_louvain.best_partition(G, weight='weight')
            # convert partition dict to labels array
            labels = np.array([int(partition.get(i, -1)) for i in range(n)], dtype=int)
        else:
            raise ValueError('Unknown method')
        return labels

    def analyze_clusters(self, papers: List[Dict], embeddings: np.ndarray, labels: np.ndarray, top_k_titles: int = 5):
        clusters = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(i)
        analysis = {}
        for lab, idxs in clusters.items():
            # Ensure label is a JSON-serializable key (convert numpy types to native int or fallback to str)
            try:
                # numpy integer types have .item(); this will convert to native python int
                if hasattr(lab, 'item'):
                    lab_key = lab.item()
                else:
                    lab_key = int(lab)
            except Exception:
                lab_key = str(lab)

            # compute weighted aggregated safety/ethics scores for the cluster
            weighted_s = 0.0
            weighted_e = 0.0
            weight_sum = 0.0
            top_titles = [str(papers[i].get('title','')) for i in idxs[:top_k_titles]]
            for i in idxs:
                p = papers[i]
                s = float(p.get('safety_score', 0) or 0)
                e = float(p.get('ethics_score', 0) or 0)
                total = s + e
                # soft weight: paper with larger total contributes more, but capped via / (1+total)
                w = float(total) / (1.0 + float(total)) if total > 0 else 0.0
                weighted_s += s * w
                weighted_e += e * w
                weight_sum += w

            if weight_sum > 0:
                agg_s = weighted_s / weight_sum
                agg_e = weighted_e / weight_sum
            else:
                # fallback to simple mean of raw scores if no weighted mass
                agg_s = float(sum(float(papers[i].get('safety_score', 0) or 0) for i in idxs) / max(1, len(idxs)))
                agg_e = float(sum(float(papers[i].get('ethics_score', 0) or 0) for i in idxs) / max(1, len(idxs)))

            src_counts = {}
            for i in idxs:
                src = papers[i].get('source','unknown')
                src_counts[src] = src_counts.get(src, 0) + 1

            analysis[lab_key] = {
                'size': int(len(idxs)),
                'sources': src_counts,
                'top_titles': top_titles,
                'aggregated': {
                    'safety_score': float(agg_s),
                    'ethics_score': float(agg_e),
                    'safety_proportion': float(agg_s / (agg_s + agg_e)) if (agg_s + agg_e) > 0 else 0.0,
                }
            }
        return analysis


def main():
    parser = argparse.ArgumentParser(description='Refactored semantic convergence analysis')
    parser.add_argument('--sample-size', type=int, default=200, help='Number of papers to sample for smoke test')
    parser.add_argument('--no-sample', action='store_true', help='Do not sample; use full dataset')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformers model')
    parser.add_argument('--cache-name', type=str, default=None, help='Cache name for embeddings (.npy)')
    parser.add_argument('--method', choices=['kmeans','hierarchical','dbscan','louvain'], default='kmeans')
    parser.add_argument('--knn', type=int, default=15, help='k for k-NN graph when using graph-based methods (louvain)')
    parser.add_argument('--n-clusters', type=int, default=20)
    parser.add_argument('--top-clusters', type=int, default=10, help='How many top clusters by size to include in summary')
    parser.add_argument('--top-titles', type=int, default=5, help='How many top titles per cluster to include')
    args = parser.parse_args()

    # Load filtered ethics papers (lightweight results), if present prefer filtered set
    filtered_path = 'data/filtered_ethics_safety_papers.jsonl'
    if os.path.exists(filtered_path):
        papers = load_jsonl(filtered_path)
    else:
        # Resilient fallback loader: read from DBLP and ACL and sample at 2x the requested sample-size
        fallback = []
        alt_paths = [
            'data/dblp_conferences_papers.jsonl',
            'data/acl_anthology_papers_v2.jsonl',
        ]
        target = args.sample_size * 2
        for p in alt_paths:
            if os.path.exists(p):
                print(f'Found fallback source: {p} - loading up to {target} lines')
                if p.endswith('.jsonl'):
                    with open(p, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= target:
                                break
                            try:
                                fallback.append(json.loads(line))
                            except Exception:
                                continue
                elif p.endswith('.csv'):
                    import csv
                    with open(p, newline='', encoding='utf-8') as fh:
                        reader = csv.DictReader(fh)
                        for i, row in enumerate(reader):
                            if i >= target:
                                break
                            fallback.append({'paper_id': row.get('paper_id') or row.get('id') or str(i),
                                             'title': row.get('title',''),
                                             'abstract': row.get('abstract',''),
                                             'source': row.get('source','csv')})
        papers = fallback

    if not papers:
        print('No papers found - run filtering first or place ethics_filter_results.jsonl in data/')
        return

    if not args.no_sample:
        random.seed(42)
        papers = random.sample(papers, min(args.sample_size, len(papers)))

    analyzer = RefactorAnalyzer(model_name=args.model)
    start = time.time()
    cache_name = args.cache_name or f'emb_{args.model.replace("/","_")}_{len(papers)}'
    embs = analyzer.compute_embeddings(papers, cache_name=cache_name)
    print(f'Embeddings computed in {time.time()-start:.1f}s')

    labels = analyzer.cluster(embs, method=args.method, n_clusters=args.n_clusters, knn=args.knn)
    analysis = analyzer.analyze_clusters(papers, embs, labels, top_k_titles=args.top_titles)

    # Build similarity network
    print('Building similarity network...')
    sim_matrix = cosine_similarity(embs)
    threshold = 0.75  # Only edges with similarity > 0.75
    G_sim = nx.Graph()
    n = len(papers)
    for i in range(n):
        s = float(papers[i].get('safety_score', 0) or 0)
        e = float(papers[i].get('ethics_score', 0) or 0)
        if s > e:
            color = 'red'  # safety
        elif e > s:
            color = 'blue'  # ethics
        else:
            color = 'purple'  # mixed
        G_sim.add_node(i, color=color, safety=s, ethics=e, title=papers[i].get('title', ''))
    
    for i in range(n):
        for j in range(i+1, n):
            sim = sim_matrix[i, j]
            if sim > threshold:
                G_sim.add_edge(i, j, weight=sim)
    
    print(f'Similarity network: {G_sim.number_of_nodes()} nodes, {G_sim.number_of_edges()} edges')
    
    # Save network for visualization
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G_sim, seed=42)
    node_colors = [G_sim.nodes[n]['color'] for n in G_sim.nodes()]
    edge_weights = [G_sim.edges[e]['weight'] for e in G_sim.edges()]
    # Normalize edge weights for alpha
    max_weight = max(edge_weights) if edge_weights else 1
    edge_alphas = [w / max_weight * 0.8 + 0.2 for w in edge_weights]  # 0.2 to 1.0
    
    nx.draw(G_sim, pos, node_color=node_colors, edge_color='gray', width=0.5, alpha=0.7, node_size=50, with_labels=False)
    plt.title(f'Semantic Similarity Network (threshold={threshold})', fontsize=14, fontweight='bold')
    plt.savefig('semantic_similarity_network.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('Network visualization saved: semantic_similarity_network.pdf')

    # derive top clusters by size
    sorted_clusters = sorted(analysis.items(), key=lambda kv: kv[1].get('size', 0), reverse=True)
    top_clusters = [{ 'cluster': k, 'size': v.get('size'), 'aggregated': v.get('aggregated')} for k, v in sorted_clusters[:args.top_clusters]]


    out = {
        'n': len(papers),
        'method': args.method,
        'n_clusters': args.n_clusters,
        'analysis': analysis,
        'top_clusters_by_size': top_clusters,
    }
    out_path = 'data/semantic_convergence_summary.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Results saved to {out_path}')
    
    # Save network for visualization
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G_sim, seed=42)
    node_colors = [G_sim.nodes[n]['color'] for n in G_sim.nodes()]
    edge_weights = [G_sim.edges[e]['weight'] for e in G_sim.edges()]
    # Normalize edge weights for alpha
    max_weight = max(edge_weights) if edge_weights else 1
    edge_alphas = [w / max_weight * 0.8 + 0.2 for w in edge_weights]  # 0.2 to 1.0
    
    nx.draw(G_sim, pos, node_color=node_colors, edge_color='gray', width=0.5, alpha=0.7, node_size=50, with_labels=False)
    plt.title(f'Semantic Similarity Network (threshold={threshold})', fontsize=14, fontweight='bold')
    plt.savefig('semantic_similarity_network.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('Network visualization saved: semantic_similarity_network.pdf')

    # derive top clusters by size
    sorted_clusters = sorted(analysis.items(), key=lambda kv: kv[1].get('size', 0), reverse=True)
    top_clusters = [{ 'cluster': k, 'size': v.get('size'), 'aggregated': v.get('aggregated')} for k, v in sorted_clusters[:args.top_clusters]]


    out = {
        'n': len(papers),
        'method': args.method,
        'n_clusters': args.n_clusters,
        'analysis': analysis,
        'top_clusters_by_size': top_clusters,
    }
    out_path = 'data/semantic_convergence_summary.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Results saved to {out_path}')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analyze Average Shortest Path (ASP) statistics for pure safety vs pure ethics
papers and authors in both coauthorship networks.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from analyze_aggregated import load_combined_data


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


def build_paper_coauthorship_network(papers: List[Dict], min_papers_per_author: int = 2) -> Tuple[nx.Graph, Dict, Dict]:
    """Build paper coauthorship network and categorize papers."""
    # Use ALL safety and ethics papers, not just top-scoring ones
    sel = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]
    print(f"Using {len(sel)} papers with safety or ethics scores (full dataset)")
    
    # Count papers per author to filter out authors with < min_papers_per_author papers
    author_paper_counts = defaultdict(int)
    for p in sel:
        for author in p.get('authors', []):
            if author:
                normalized_author = normalize_author(author)
                author_paper_counts[normalized_author] += 1
    
    # Keep only authors who appear in at least min_papers_per_author papers
    qualified_authors = {author for author, count in author_paper_counts.items() if count >= min_papers_per_author}
    print(f"Filtering to {len(qualified_authors)} authors who appear in ≥{min_papers_per_author} papers")
    
    # Filter papers to only include those with at least one qualified author
    filtered_papers = []
    for p in sel:
        paper_authors = [normalize_author(a) for a in p.get('authors', []) if a]
        if any(author in qualified_authors for author in paper_authors):
            # Update paper with only qualified authors
            p_copy = p.copy()
            p_copy['authors'] = [a for a in paper_authors if a in qualified_authors]
            if len(p_copy['authors']) > 0:  # Only keep papers with at least one qualified author
                filtered_papers.append(p_copy)
    
    sel = filtered_papers
    print(f"Final dataset: {len(sel)} papers with qualified authors")
    
    G = nx.Graph()
    paper_categories = {}
    paper_info = {}
    
    for i, p in enumerate(sel):
        safety_score = float(p.get('safety_score', 0) or 0)
        ethics_score = float(p.get('ethics_score', 0) or 0)
        
        # Categorize papers: pure = 100% one category, mixed = both categories
        if safety_score > 0 and ethics_score == 0:
            category = 'pure_safety'
        elif ethics_score > 0 and safety_score == 0:
            category = 'pure_ethics'
        elif safety_score > 0 and ethics_score > 0:
            category = 'mixed'
        else:
            category = 'neither'
            
        G.add_node(i, category=category, safety=safety_score, ethics=ethics_score)
        paper_categories[i] = category
        paper_info[i] = {
            'title': p.get('title', ''),
            'safety_score': safety_score,
            'ethics_score': ethics_score,
            'category': category
        }
    
    # Add edges between papers with shared authors
    author_to_papers = defaultdict(list)
    for i, p in enumerate(sel):
        for a in p.get('authors', []):
            if a:
                normalized_author = normalize_author(a)
                author_to_papers[normalized_author].append(i)
    
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
    
    return G, paper_categories, paper_info


def build_author_coauthorship_network(papers: List[Dict], min_papers_per_author: int = 2) -> Tuple[nx.Graph, Dict, Dict]:
    """Build author coauthorship network and categorize authors."""
    # First, filter to authors who appear in at least min_papers_per_author papers
    author_paper_counts = defaultdict(int)
    for p in papers:
        safety_score = float(p.get('safety_score', 0) or 0)
        ethics_score = float(p.get('ethics_score', 0) or 0)
        if safety_score > 0 or ethics_score > 0:  # Only count safety/ethics papers
            for author in p.get('authors', []):
                if author:
                    normalized_author = normalize_author(author)
                    author_paper_counts[normalized_author] += 1
    
    # Keep only authors who appear in at least min_papers_per_author papers
    qualified_authors = {author for author, count in author_paper_counts.items() if count >= min_papers_per_author}
    print(f"Filtering to {len(qualified_authors)} authors who appear in ≥{min_papers_per_author} safety/ethics papers")
    
    # Aggregate scores per qualified author
    author_scores = defaultdict(lambda: {'safety': 0.0, 'ethics': 0.0, 'papers': set()})
    
    for idx, p in enumerate(papers):
        safety_score = float(p.get('safety_score', 0) or 0)
        ethics_score = float(p.get('ethics_score', 0) or 0)
        
        if safety_score > 0 or ethics_score > 0:  # Only process safety/ethics papers
            for a in p.get('authors', []):
                if not a:
                    continue
                normalized_author = normalize_author(a)
                
                # Only include qualified authors
                if normalized_author in qualified_authors:
                    author_scores[normalized_author]['safety'] += safety_score
                    author_scores[normalized_author]['ethics'] += ethics_score
                    author_scores[normalized_author]['papers'].add(idx)
    
    # Build graph and categorize authors
    G = nx.Graph()
    author_categories = {}
    author_info = {}
    
    for author, scores in author_scores.items():
        safety_total = scores['safety']
        ethics_total = scores['ethics']
        
        if safety_total == 0 and ethics_total == 0:
            continue
            
        # Categorize authors: pure = 100% one category, mixed = both categories
        if safety_total > 0 and ethics_total == 0:
            category = 'pure_safety'
        elif ethics_total > 0 and safety_total == 0:
            category = 'pure_ethics'
        elif safety_total > 0 and ethics_total > 0:
            category = 'mixed'
        else:
            category = 'neither'
            
        G.add_node(author, category=category, safety=safety_total, ethics=ethics_total)
        author_categories[author] = category
        author_info[author] = {
            'safety_score': safety_total,
            'ethics_score': ethics_total,
            'num_papers': len(scores['papers']),
            'category': category
        }
    
    # Add coauthorship edges (only among qualified authors)
    paper_to_authors = defaultdict(list)
    for author, scores in author_scores.items():
        if author not in G.nodes():
            continue
        for paper_idx in scores['papers']:
            paper_to_authors[paper_idx].append(author)
    
    for authors_list in paper_to_authors.values():
        if len(authors_list) < 2:
            continue
        for i in range(len(authors_list)):
            for j in range(i + 1, len(authors_list)):
                u, v = authors_list[i], authors_list[j]
                if u == v:
                    continue
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    
    # Use the full author network (no degree filtering)
    print(f"Using {G.number_of_nodes()} authors from full dataset")
    
    return G, author_categories, author_info


def calculate_asp_statistics(G: nx.Graph, categories: Dict, entity_type: str = "paper"):
    """Calculate Average Shortest Path statistics between different categories."""
    
    if not nx.is_connected(G):
        print(f"⚠️  {entity_type.title()} network is not connected. Using largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        categories = {node: categories[node] for node in G.nodes() if node in categories}
    
    # Group nodes by category
    pure_safety = [node for node, cat in categories.items() if cat == 'pure_safety']
    pure_ethics = [node for node, cat in categories.items() if cat == 'pure_ethics']
    mixed = [node for node, cat in categories.items() if cat == 'mixed']
    
    print(f"\n📊 {entity_type.upper()} NETWORK COMPOSITION:")
    print(f"  Pure Safety (100%): {len(pure_safety)} {entity_type}s")
    print(f"  Pure Ethics (100%): {len(pure_ethics)} {entity_type}s")
    print(f"  Mixed (Both): {len(mixed)} {entity_type}s")
    print(f"  Total in largest component: {G.number_of_nodes()} {entity_type}s")
    
    if len(pure_safety) == 0 or len(pure_ethics) == 0:
        print(f"⚠️  Cannot calculate ASP: missing pure safety or pure ethics {entity_type}s")
        return
    
    # Calculate shortest paths
    print(f"\n📏 AVERAGE SHORTEST PATH ANALYSIS:")
    
    # Efficient BFS-based pairwise distance collection to avoid repeated shortest_path calls
    def gather_distances_between_sets(G_local, sources, targets, unique_pairs=False):
        dists = []
        targets_set = set(targets)
        if unique_pairs and set(sources) == set(targets):
            # compute only pairs (i < j) using ordering
            order = {n: i for i, n in enumerate(sources)}
            for i, u in enumerate(sources):
                lengths = nx.single_source_shortest_path_length(G_local, u)
                for v, dist in lengths.items():
                    j = order.get(v)
                    if j is None:
                        continue
                    if j > i:
                        dists.append(dist)
        else:
            # For each source, run single-source BFS and collect distances to all targets
            for u in sources:
                lengths = nx.single_source_shortest_path_length(G_local, u)
                for v in targets:
                    if v in lengths:
                        dists.append(lengths[v])
        return dists

    # ASP within pure safety
    if len(pure_safety) > 1:
        safety_paths = gather_distances_between_sets(G, pure_safety, pure_safety, unique_pairs=True)
        if safety_paths:
            avg_safety = np.mean(safety_paths)
            print(f"  Pure Safety ↔ Pure Safety: {avg_safety:.3f} (n={len(safety_paths)} pairs)")
        else:
            print(f"  Pure Safety ↔ Pure Safety: No connected pairs")

    # ASP within pure ethics
    if len(pure_ethics) > 1:
        ethics_paths = gather_distances_between_sets(G, pure_ethics, pure_ethics, unique_pairs=True)
        if ethics_paths:
            avg_ethics = np.mean(ethics_paths)
            print(f"  Pure Ethics ↔ Pure Ethics: {avg_ethics:.3f} (n={len(ethics_paths)} pairs)")
        else:
            print(f"  Pure Ethics ↔ Pure Ethics: No connected pairs")

    # ASP between pure safety and pure ethics
    cross_paths = gather_distances_between_sets(G, pure_safety, pure_ethics, unique_pairs=False)
    if cross_paths:
        avg_cross = np.mean(cross_paths)
        print(f"  Pure Safety ↔ Pure Ethics: {avg_cross:.3f} (n={len(cross_paths)} pairs)")
    else:
        print(f"  Pure Safety ↔ Pure Ethics: No connected pairs")

    # ASP involving 'mixed' category
    if len(mixed) > 0:
        mixed_safety_paths = gather_distances_between_sets(G, mixed, pure_safety, unique_pairs=False)
        if mixed_safety_paths:
            avg_mixed_safety = np.mean(mixed_safety_paths)
            print(f"  Mixed ↔ Pure Safety: {avg_mixed_safety:.3f} (n={len(mixed_safety_paths)} pairs)")

        mixed_ethics_paths = gather_distances_between_sets(G, mixed, pure_ethics, unique_pairs=False)
        if mixed_ethics_paths:
            avg_mixed_ethics = np.mean(mixed_ethics_paths)
            print(f"  Mixed ↔ Pure Ethics: {avg_mixed_ethics:.3f} (n={len(mixed_ethics_paths)} pairs)")


def compute_mixing_and_distance_stats(G: nx.Graph, categories: Dict, entity_type: str = "paper"):
    """Compute mixing matrix (edge counts by category pairs), assortativity, modularity (via greedy modularity),
    fraction of edges that are same-category (homophily), and median/percentile shortest-path distances between categories."""

    # Restrict to largest connected component for path distances and community detection
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(largest_cc).copy()
    else:
        Gc = G

    # Build mapping from node to category (only for nodes present in Gc)
    node_cat = {n: categories[n] for n in Gc.nodes() if n in categories}

    cats = ['pure_safety', 'pure_ethics', 'mixed']

    # Mixing matrix: count edges between categories (undirected)
    mix_counts = defaultdict(int)
    total_edges = 0
    for u, v in Gc.edges():
        cu = node_cat.get(u, 'neither')
        cv = node_cat.get(v, 'neither')
        key = tuple(sorted([cu, cv]))
        mix_counts[key] += 1
        total_edges += 1

    print(f"\n🔗 {entity_type.title()} NETWORK MIXING (largest component, edges={total_edges}):")
    for a in cats:
        for b in cats:
            key = tuple(sorted([a, b]))
            cnt = mix_counts.get(key, 0)
            print(f"  {a} - {b}: {cnt}")

    # Homophily: fraction of edges that connect same-category nodes
    same_cat_edges = sum(v for k, v in mix_counts.items() if k[0] == k[1])
    homophily = same_cat_edges / total_edges if total_edges else 0.0
    print(f"  Homophily (same-category edge fraction): {homophily:.3f}")

    # Assortativity by categorical attribute: need numeric mapping
    cat_to_int = {None: -1, 'pure_safety': 0, 'pure_ethics': 1, 'mixed': 2, 'neither': 3}
    # Create a copy of Gc with integer attribute
    H = Gc.copy()
    for n in H.nodes():
        H.nodes[n]['cat_int'] = cat_to_int.get(node_cat.get(n, 'neither'), 3)

    try:
        assort = nx.attribute_assortativity_coefficient(H, 'cat_int')
    except Exception:
        assort = float('nan')
    print(f"  Assortativity (categorical): {assort:.3f}")

    # Community detection -> modularity (greedy modularity communities)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(H))
        # compute modularity
        modularity = nx.algorithms.community.quality.modularity(H, communities)
    except Exception:
        modularity = float('nan')
    print(f"  Modularity (greedy communities): {modularity:.3f}")

    # Shortest-path distances percentiles between categories (median, 75th)
    def pairwise_distances(cat_a_nodes, cat_b_nodes):
        dists = []
        for u in cat_a_nodes:
            for v in cat_b_nodes:
                if u == v:
                    continue
                try:
                    d = nx.shortest_path_length(H, u, v)
                    dists.append(d)
                except nx.NetworkXNoPath:
                    pass
        return dists

    cat_nodes = {c: [n for n, ct in node_cat.items() if ct == c] for c in cats}
    print(f"\n📏 {entity_type.title()} DISTANCE PERCENTILES (largest component):")
    for a in cats:
        for b in cats:
            if a == b:
                nodes_a = cat_nodes[a]
                if len(nodes_a) < 2:
                    continue
                dists = pairwise_distances(nodes_a, nodes_a)
            else:
                dists = pairwise_distances(cat_nodes[a], cat_nodes[b])

            if not dists:
                continue
            med = np.median(dists)
            p75 = np.percentile(dists, 75)
            print(f"  {a} - {b}: median={med}, 75th={p75}, pairs={len(dists)}")


def build_top_n_paper_network(papers: List[Dict], top_n: int = 400) -> Tuple[nx.Graph, Dict]:
    """Build the same top-N paper coauthorship network used for the plotted `combined_paper_network_colored.png`.
    Returns graph and categories mapping (node -> category string).
    """
    # Select top_n by max(safety, ethics)
    scored = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]
    scored.sort(key=lambda p: max(p.get('safety_score', 0) or 0, p.get('ethics_score', 0) or 0), reverse=True)
    sel = scored[:top_n]
    G = nx.Graph()
    categories = {}
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
        G.add_node(i, safety=s, ethics=e, title=p.get('title', ''))
        categories[i] = cat

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
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G, categories


def component_and_reachable_stats(G: nx.Graph, categories: Dict, entity_type: str = 'paper'):
    """Report component counts, sizes, how many components mix categories, and reachable-pair ASPs."""
    comps = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in comps]
    total = G.number_of_nodes()
    print(f"\n🧩 {entity_type.title()} COMPONENTS: count={len(comps)}, total_nodes={total}")
    comp_size_hist = sorted(comp_sizes, reverse=True)[:10]
    print(f"  Top component sizes (top 10): {comp_size_hist}")

    # Count components that contain both pure_safety and pure_ethics
    mixed_components = 0
    pure_only_components = 0
    for c in comps:
        cats_in_c = {categories.get(n, 'neither') for n in c}
        if 'pure_safety' in cats_in_c and 'pure_ethics' in cats_in_c:
            mixed_components += 1
        elif cats_in_c.issubset({'pure_safety', 'neither'}):
            pure_only_components += 1

    print(f"  Components with both categories: {mixed_components}")
    print(f"  Components pure-only safety (or neither): {pure_only_components}")

    # Reachable-pair ASPs: compute shortest-path lengths within each component and combine
    all_dists = []
    for c in comps:
        if len(c) < 2:
            continue
        sub = G.subgraph(c)
        for u in sub.nodes():
            for v in sub.nodes():
                if u >= v:
                    continue
                try:
                    d = nx.shortest_path_length(sub, u, v)
                    all_dists.append(d)
                except nx.NetworkXNoPath:
                    pass

    if all_dists:
        print(f"\n📏 Reachable-pair ASP across all components: mean={np.mean(all_dists):.3f}, median={np.median(all_dists):.3f}, pairs={len(all_dists)}")
    else:
        print("\n📏 No reachable pairs to compute ASP")


def reachability_by_hops(G: nx.Graph, categories: Dict, max_k: int = 6):
    """Compute fraction of SxE, SxS, ExE pairs that are reachable within <=k hops for k=1..max_k.
    Returns dict with arrays for counts and fractions and sample sizes.
    Uses BFS up to cutoff=max_k from each source node.
    """
    nodes = list(G.nodes())
    # Identify pure safety and pure ethics nodes
    S_nodes = [n for n, c in categories.items() if c == 'pure_safety' and n in G]
    E_nodes = [n for n, c in categories.items() if c == 'pure_ethics' and n in G]

    Ns = len(S_nodes)
    Ne = len(E_nodes)
    total_SE = Ns * Ne
    total_SS = Ns * (Ns - 1) // 2
    total_EE = Ne * (Ne - 1) // 2

    # Prepare ordering for S-S and E-E unique counting
    S_index = {n: i for i, n in enumerate(sorted(S_nodes))}
    E_index = {n: i for i, n in enumerate(sorted(E_nodes))}

    # counters for k=1..max_k (use 1-based indexing)
    se_counts = [0] * (max_k + 1)
    ss_counts = [0] * (max_k + 1)
    ee_counts = [0] * (max_k + 1)

    # BFS from S nodes to count S->E and S->S (unique pairs via ordering)
    for s in S_nodes:
        lengths = nx.single_source_shortest_path_length(G, s, cutoff=max_k)
        for t, d in lengths.items():
            if d == 0:
                continue
            if d <= max_k:
                if t in E_index:
                    se_counts[d] += 1
                if t in S_index:
                    # unordered pair only count when index(t) > index(s)
                    if S_index[t] > S_index[s]:
                        ss_counts[d] += 1

    # BFS from E nodes to count E->E unique pairs
    for e in E_nodes:
        lengths = nx.single_source_shortest_path_length(G, e, cutoff=max_k)
        for t, d in lengths.items():
            if d == 0:
                continue
            if d <= max_k:
                if t in E_index:
                    if E_index[t] > E_index[e]:
                        ee_counts[d] += 1

    # Convert cumulative counts: counts within <=k
    se_cum = []
    ss_cum = []
    ee_cum = []
    s_acc = 0
    ss_acc = 0
    ee_acc = 0
    for k in range(1, max_k + 1):
        s_acc += se_counts[k]
        ss_acc += ss_counts[k]
        ee_acc += ee_counts[k]
        se_cum.append(s_acc)
        ss_cum.append(ss_acc)
        ee_cum.append(ee_acc)

    # Fractions
    se_frac = [c / total_SE if total_SE else 0.0 for c in se_cum]
    ss_frac = [c / total_SS if total_SS else 0.0 for c in ss_cum]
    ee_frac = [c / total_EE if total_EE else 0.0 for c in ee_cum]

    return {
        'Ns': Ns, 'Ne': Ne,
        'total_SE': total_SE, 'total_SS': total_SS, 'total_EE': total_EE,
        'se_counts_by_k': se_cum, 'ss_counts_by_k': ss_cum, 'ee_counts_by_k': ee_cum,
        'se_frac_by_k': se_frac, 'ss_frac_by_k': ss_frac, 'ee_frac_by_k': ee_frac,
    }


def permutation_reachability_test(G: nx.Graph, categories: Dict, max_k: int = 6, n_perms: int = 200, seed: int = 42):
    """Permutation test by shuffling labels among nodes. Returns permuted fractions for each k and p-values.
    Note: this preserves counts of labels but randomizes their assignment across nodes.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    orig_labels = [categories.get(n, 'neither') for n in nodes]
    # observed
    obs = reachability_by_hops(G, categories, max_k=max_k)
    obs_se = obs['se_frac_by_k']

    perms = np.zeros((n_perms, max_k))
    for i in range(n_perms):
        perm_labels = rng.permutation(orig_labels)
        perm_cat = {n: lab for n, lab in zip(nodes, perm_labels)}
        pr = reachability_by_hops(G, perm_cat, max_k=max_k)
        perms[i, :] = pr['se_frac_by_k']

    # compute p-values (one-sided: fraction of perms >= observed)
    pvals = []
    for k in range(max_k):
        p = (np.sum(perms[:, k] >= obs_se[k]) + 1) / (n_perms + 1)
        pvals.append(p)

    return {
        'obs_se_frac': obs_se,
        'perms': perms,
        'pvals': pvals,
        'obs': obs,
    }


def bootstrap_reachability(G: nx.Graph, categories: Dict, max_k: int = 6, n_boot: int = 200, seed: int = 42):
    """Bootstrap node-level resampling to get CIs for se_frac_by_k. Samples S nodes with replacement and computes fractions.
    Returns bootstrap matrix of shape (n_boot, max_k).
    """
    rng = np.random.default_rng(seed)
    S_nodes = [n for n, c in categories.items() if c == 'pure_safety' and n in G]
    E_nodes = [n for n, c in categories.items() if c == 'pure_ethics' and n in G]
    Ns = len(S_nodes)
    if Ns == 0 or len(E_nodes) == 0:
        return None

    boot = np.zeros((n_boot, max_k))
    for i in range(n_boot):
        sampled_S = list(rng.choice(S_nodes, size=Ns, replace=True))
        # compute counts for sampled S only
        se_counts = [0] * (max_k + 1)
        for s in sampled_S:
            lengths = nx.single_source_shortest_path_length(G, s, cutoff=max_k)
            for t, d in lengths.items():
                if d == 0:
                    continue
                if d <= max_k and categories.get(t) == 'pure_ethics':
                    se_counts[d] += 1
        # cumulative
        s_acc = 0
        se_cum = []
        for k in range(1, max_k + 1):
            s_acc += se_counts[k]
            se_cum.append(s_acc)
        total_SE = Ns * len(E_nodes)
        boot[i, :] = np.array([c / total_SE if total_SE else 0.0 for c in se_cum])

    return boot


def degree_preserving_nulls(G: nx.Graph, categories: Dict, max_k: int = 6, n_swaps: int = 1000, n_trials: int = 5, seed: int = 42):
    """Generate degree-preserving rewired graphs and compute reachability_by_hops for each trial.
    Uses double_edge_swap; returns array trials x k.
    """
    rng = np.random.default_rng(seed)
    trials = []
    for t in range(n_trials):
        H = G.copy()
        # number of swap attempts proportional to edges
        try:
            nx.double_edge_swap(H, n_swaps=n_swaps, max_tries=n_swaps * 10)
        except Exception:
            pass
        r = reachability_by_hops(H, categories, max_k=max_k)
        trials.append(r['se_frac_by_k'])
    return np.array(trials)


def degree_trim_global(G: nx.Graph, categories: Dict, percentile: float = 99.0):
    """Top-level degree trimming helper (keeps nodes with degree <= percentile threshold)."""
    degs = [d for _, d in G.degree()]
    if not degs:
        return G.copy(), {n: categories.get(n, 'neither') for n in G.nodes()}
    thresh = float(np.percentile(degs, percentile))
    keep = [n for n, d in G.degree() if d <= thresh]
    H = G.subgraph(keep).copy()
    new_cat = {n: categories.get(n, 'neither') for n in H.nodes()}
    return H, new_cat


def permutation_reachability_all(G: nx.Graph, categories: Dict, max_k: int = 6, n_perms: int = 200, seed: int = 42):
    """Permutation test returning arrays of permuted SE, SS, EE fractions per k.
    Returns perms_se (n_perms x k), perms_ss, perms_ee and observed fractions.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    orig_labels = [categories.get(n, 'neither') for n in nodes]
    # observed
    obs = reachability_by_hops(G, categories, max_k=max_k)
    obs_se = np.array(obs['se_frac_by_k'])
    obs_ss = np.array(obs['ss_frac_by_k'])
    obs_ee = np.array(obs['ee_frac_by_k'])
    obs_se_counts = np.array(obs['se_counts_by_k'])
    obs_ss_counts = np.array(obs['ss_counts_by_k'])
    obs_ee_counts = np.array(obs['ee_counts_by_k'])

    perms_se = np.zeros((n_perms, max_k))
    perms_ss = np.zeros((n_perms, max_k))
    perms_ee = np.zeros((n_perms, max_k))
    for i in range(n_perms):
        perm_labels = rng.permutation(orig_labels)
        perm_cat = {n: lab for n, lab in zip(nodes, perm_labels)}
        pr = reachability_by_hops(G, perm_cat, max_k=max_k)
    perms_se[i, :] = pr['se_frac_by_k']
    perms_ss[i, :] = pr['ss_frac_by_k']
    perms_ee[i, :] = pr['ee_frac_by_k']

    # also return observed absolute counts if available in obs
    obs_se_counts = np.array(obs.get('se_counts_by_k', [0]*len(obs_se)))
    obs_ss_counts = np.array(obs.get('ss_counts_by_k', [0]*len(obs_ss)))
    obs_ee_counts = np.array(obs.get('ee_counts_by_k', [0]*len(obs_ee)))

    return {
        'obs_se': obs_se, 'obs_ss': obs_ss, 'obs_ee': obs_ee,
        'obs_se_counts': obs_se_counts, 'obs_ss_counts': obs_ss_counts, 'obs_ee_counts': obs_ee_counts,
        'perms_se': perms_se, 'perms_ss': perms_ss, 'perms_ee': perms_ee
    }


def run_permutation_diff_test(G: nx.Graph, categories: Dict, name: str, trim_pct: float = 95.0, max_k: int = 6, n_perms: int = 200):
    """Apply degree trimming, run permutation_reachability_all, and compute p-values for SS-SE and EE-SE differences per k."""
    H, catH = degree_trim_global(G, categories, percentile=trim_pct)
    print(f"\n--- Permutation diff test: {name}, trim {trim_pct}th pct (nodes left: {H.number_of_nodes()}) ---")
    res = permutation_reachability_all(H, catH, max_k=max_k, n_perms=n_perms)
    obs_se = res['obs_se']
    obs_ss = res['obs_ss']
    obs_ee = res['obs_ee']
    obs_se_counts = np.array(res.get('perms', {}).get('obs', {}).get('se_counts_by_k', res.get('obs_se', []) ))
    # fallback: if counts are present in res (not in perms wrapper), try keys
    if 'se_counts_by_k' in res:
        obs_se_counts = np.array(res.get('se_counts_by_k'))
    perms_se = res['perms_se']
    perms_ss = res['perms_ss']
    perms_ee = res['perms_ee']

    # compute perm diffs: ss-se and ee-se
    perms_diff_ss_se = perms_ss - perms_se
    perms_diff_ee_se = perms_ee - perms_se

    obs_diff_ss_se = obs_ss - obs_se
    obs_diff_ee_se = obs_ee - obs_se

    pvals_ss_se = []
    pvals_ee_se = []
    for k in range(max_k):
        # p-value for observing diff as large as obs (one-sided)
        p_ss = (np.sum(perms_diff_ss_se[:, k] >= obs_diff_ss_se[k]) + 1) / (n_perms + 1)
        p_ee = (np.sum(perms_diff_ee_se[:, k] >= obs_diff_ee_se[k]) + 1) / (n_perms + 1)
        pvals_ss_se.append(p_ss)
        pvals_ee_se.append(p_ee)

    # print table
    print('k | obs_SE_count | obs_SS_count | obs_EE_count | obs_SE% | obs_SS% | obs_EE% | diff(SS-SE)% | p(SS>SE) | diff(EE-SE)% | p(EE>SE)')
    for k in range(max_k):
        se_count = int(res.get('obs_se_counts', np.zeros(max_k))[k])
        ss_count = int(res.get('obs_ss_counts', np.zeros(max_k))[k])
        ee_count = int(res.get('obs_ee_counts', np.zeros(max_k))[k])
        print(f"{k+1} | {se_count} | {ss_count} | {ee_count} | {obs_se[k]*100:5.3f} | {obs_ss[k]*100:5.3f} | {obs_ee[k]*100:5.3f} | {obs_diff_ss_se[k]*100:6.3f} | {pvals_ss_se[k]:.3f} | {obs_diff_ee_se[k]*100:6.3f} | {pvals_ee_se[k]:.3f}")

    return {
        'obs_se': obs_se, 'obs_ss': obs_ss, 'obs_ee': obs_ee,
        'obs_diff_ss_se': obs_diff_ss_se, 'obs_diff_ee_se': obs_diff_ee_se,
        'pvals_ss_se': pvals_ss_se, 'pvals_ee_se': pvals_ee_se,
        'perms': res
    }





def main():
    """Main analysis function."""
    print("🔍 Loading combined ACL and DBLP data...")
    
    papers = load_combined_data()
    
    
    # Normalize authors for both datasets
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
    
    print(f"Total papers: {len(all_papers)}")
    
    print("\n" + "="*80)
    print("📄 PAPER COAUTHORSHIP NETWORK ANALYSIS (AUTHORS WITH ≥2 PAPERS)")
    print("="*80)
    
    paper_graph, paper_categories, paper_info = build_paper_coauthorship_network(all_papers, min_papers_per_author=2)

    # --- K-hop reachability analysis (1..6) and plotting ---
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
        #plt.savefig(out_png, dpi=200)
        plt.close()

    print('\n▶ Computing k-hop reachability (k=1..6) for full paper graph...')
    paper_reach = reachability_by_hops(paper_graph, paper_categories, max_k=6)
    print('  Ns (pure safety):', paper_reach['Ns'], 'Ne (pure ethics):', paper_reach['Ne'])
    print('  Safety–Ethics reachable % by k:', [f*100 for f in paper_reach['se_frac_by_k']])
    plot_reachability(paper_reach, 'reachability_paper_by_k', title_prefix='Paper (full) - ')

    # Also analyze the top-N paper network that matches the plotted `combined_paper_network_colored.png`
    topG, top_cats = build_top_n_paper_network(all_papers, top_n=400)
    print('\n▶ Computing k-hop reachability for top-400 plotted paper network...')
    top_reach = reachability_by_hops(topG, top_cats, max_k=6)
    print('  Safety–Ethics reachable % by k (top-400):', [f*100 for f in top_reach['se_frac_by_k']])
    plot_reachability(top_reach, 'reachability_paper_top400_by_k', title_prefix='Paper (top-400) - ')

    # Now compute for author graph as well (but avoid further heavy ASPs)
    print('\n' + "="*80)
    print('👥 AUTHOR COAUTHORSHIP NETWORK REACHABILITY (AUTHORS WITH ≥2 PAPERS)')
    print("="*80)
    author_graph, author_categories, author_info = build_author_coauthorship_network(all_papers, min_papers_per_author=2)
    print('\n▶ Computing k-hop reachability for author graph...')
    author_reach = reachability_by_hops(author_graph, author_categories, max_k=6)
    print('  Safety–Ethics reachable % by k (authors):', [f*100 for f in author_reach['se_frac_by_k']])
    plot_reachability(author_reach, 'reachability_author_by_k', title_prefix='Author - ')

    print('\n✅ Reachability plots saved: reachability_paper_by_k.png, reachability_paper_top400_by_k.png, reachability_author_by_k.png')
    # Experiments: remove mixed nodes and degree-trimming
    def filter_remove_mixed(G, categories):
        keep = [n for n in G.nodes() if categories.get(n) in ('pure_safety', 'pure_ethics')]
        H = G.subgraph(keep).copy()
        new_cat = {n: categories[n] for n in H.nodes()}
        return H, new_cat

    def filter_degree_trim(G, categories, percentile=99.0):
        degs = [d for _, d in G.degree()]
        if not degs:
            return G.copy(), {n: categories[n] for n in G.nodes()}
        thresh = float(np.percentile(degs, percentile))
        keep = [n for n, d in G.degree() if d <= thresh]
        H = G.subgraph(keep).copy()
        new_cat = {n: categories.get(n, 'neither') for n in H.nodes()}
        return H, new_cat

    print('\n▶ Experiment 1: remove mixed nodes (papers)')
    paper_nomixed, paper_nomixed_cat = filter_remove_mixed(paper_graph, paper_categories)
    nomixed_reach = reachability_by_hops(paper_nomixed, paper_nomixed_cat, max_k=6)
    print('  Safety–Ethics reachable % by k (paper, no mixed):', [f*100 for f in nomixed_reach['se_frac_by_k']])
    plot_reachability(nomixed_reach, 'reachability_paper_nomixed_by_k', title_prefix='Paper (no mixed) - ')

    print('\n▶ Experiment 2: degree-trim hubs (top 1%) on paper graph')
    paper_trim, paper_trim_cat = filter_degree_trim(paper_graph, paper_categories, percentile=99.0)
    trim_reach = reachability_by_hops(paper_trim, paper_trim_cat, max_k=6)
    print('  Safety–Ethics reachable % by k (paper, trimmed):', [f*100 for f in trim_reach['se_frac_by_k']])
    plot_reachability(trim_reach, 'reachability_paper_trim_by_k', title_prefix='Paper (trim hubs) - ')

    print('\n▶ Experiment 3: remove mixed nodes from top-400')
    top_nomixedG = topG.copy()
    # top_cats uses indices 0..399
    top_nomixed_nodes = [n for n, c in top_cats.items() if c in ('pure_safety','pure_ethics')]
    top_nomixed = topG.subgraph(top_nomixed_nodes).copy()
    top_nomixed_cat = {n: top_cats[n] for n in top_nomixed.nodes()}
    top_nomixed_reach = reachability_by_hops(top_nomixed, top_nomixed_cat, max_k=6)
    print('  Safety–Ethics reachable % by k (top-400, no mixed):', [f*100 for f in top_nomixed_reach['se_frac_by_k']])
    plot_reachability(top_nomixed_reach, 'reachability_top400_nomixed_by_k', title_prefix='Top400 (no mixed) - ')

    print('\n▶ Experiment 4: degree-trim hubs on author graph (top 1%)')
    author_trim, author_trim_cat = filter_degree_trim(author_graph, author_categories, percentile=99.0)
    author_trim_reach = reachability_by_hops(author_trim, author_trim_cat, max_k=6)
    print('  Safety–Ethics reachable % by k (author, trimmed):', [f*100 for f in author_trim_reach['se_frac_by_k']])
    plot_reachability(author_trim_reach, 'reachability_author_trim_by_k', title_prefix='Author (trim hubs) - ')

    # Hub-trimming sweep
    sweep_percentiles = [99.0, 95.0, 90.0, 75.0]
    def run_trim_sweep(G, categories, name_prefix: str):
        results = {}
        for pctl in sweep_percentiles:
            H, catH = filter_degree_trim(G, categories, percentile=pctl)
            r = reachability_by_hops(H, catH, max_k=6)
            results[pctl] = r
            print(f"{name_prefix} trim {pctl}th pct -> S–E k=6: {r['se_frac_by_k'][-1]*100:.3f}% (nodes left: {H.number_of_nodes()})")
            plot_reachability(r, f"{name_prefix}_trim_{int(pctl)}", title_prefix=f"{name_prefix} trim {int(pctl)} - ")
        return results

    print('\n▶ Running hub-trim sweep for papers (full), top-400, and authors')
    paper_trim_sweep = run_trim_sweep(paper_graph, paper_categories, 'paper_full')
    top_trim_sweep = run_trim_sweep(topG, top_cats, 'paper_top400')
    author_trim_sweep = run_trim_sweep(author_graph, author_categories, 'author')

    print('\n✅ Hub-trim sweep complete; plots saved (one per percentile).')

    print('\n✅ Experiment plots saved: reachability_paper_nomixed_by_k.png, reachability_paper_trim_by_k.png, reachability_top400_nomixed_by_k.png, reachability_author_trim_by_k.png')
    # Run permutation diff tests on trimmed graphs (95th percentile)
    try:
        perm_paper = run_permutation_diff_test(paper_graph, paper_categories, name='paper_full', trim_pct=95.0, max_k=6, n_perms=200)
    except Exception as e:
        print('Permutation test (paper) failed:', e)

    try:
        perm_author = run_permutation_diff_test(author_graph, author_categories, name='author', trim_pct=95.0, max_k=6, n_perms=200)
    except Exception as e:
        print('Permutation test (author) failed:', e)

    return

    # Compute bridge metrics on the largest connected component (who sits on shortest paths)
    def compute_bridge_metrics(G: nx.Graph, categories: Dict, entity_type: str = 'paper'):
        print(f"\n🔎 {entity_type.title()} BRIDGE METRICS (largest connected component):")
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            H = G.subgraph(largest_cc).copy()
        else:
            H = G

        # nodes by category within H
        pure_safety = [n for n in H.nodes() if categories.get(n) == 'pure_safety']
        pure_ethics = [n for n in H.nodes() if categories.get(n) == 'pure_ethics']

        print(f"  Nodes in component: {H.number_of_nodes()}, edges: {H.number_of_edges()}")
        print(f"  Pure Safety: {len(pure_safety)}, Pure Ethics: {len(pure_ethics)}")

        if not pure_safety or not pure_ethics:
            print("  Not enough category nodes to compute bridge metrics")
            return

        # Betweenness centrality restricted to shortest paths between pure_safety and pure_ethics
        try:
            bc = nx.betweenness_centrality_subset(H, sources=pure_safety, targets=pure_ethics, normalized=False)
        except Exception:
            # fallback: compute full betweenness (more expensive)
            bc = nx.betweenness_centrality(H, normalized=False)

        bridge_nodes = [n for n, v in bc.items() if v > 0]
        num_bridge = len(bridge_nodes)
        frac_bridge_comp = num_bridge / H.number_of_nodes() if H.number_of_nodes() else 0.0
        frac_bridge_total = num_bridge / G.number_of_nodes() if G.number_of_nodes() else 0.0

        print(f"  Bridge nodes (on any S↔E shortest path): {num_bridge} ({frac_bridge_comp:.3%} of component, {frac_bridge_total:.3%} of graph)")

        # Reachable pairs distances S->E via single-source BFS per S
        dists = []
        reachable_pairs = 0
        for s in pure_safety:
            lengths = nx.single_source_shortest_path_length(H, s)
            for t in pure_ethics:
                if t in lengths:
                    dists.append(lengths[t])
                    reachable_pairs += 1

        total_pairs = len(pure_safety) * len(pure_ethics)
        reachable_frac = reachable_pairs / total_pairs if total_pairs else 0.0
        if dists:
            print(f"  Safety↔Ethics reachable pairs: {reachable_pairs}/{total_pairs} ({reachable_frac:.3%})")
            print(f"  Mean distance (reachable pairs): {np.mean(dists):.3f}, median: {np.median(dists):.3f}")
        else:
            print("  No reachable S↔E pairs in this component")

        # Top bridge nodes by centrality
        top_bridges = sorted(((n, bc.get(n, 0)) for n in bridge_nodes), key=lambda x: x[1], reverse=True)[:20]
        print("  Top bridge nodes (node, raw-count-of-shortest-paths):")
        for n, v in top_bridges:
            print(f"    {n}: {v}")

        return {
            'component_size': H.number_of_nodes(),
            'num_bridge': num_bridge,
            'frac_bridge_comp': frac_bridge_comp,
            'frac_bridge_total': frac_bridge_total,
            'reachable_pairs': reachable_pairs,
            'total_pairs': total_pairs,
            'mean_dist': np.mean(dists) if dists else None,
            'median_dist': np.median(dists) if dists else None,
        }

    # run bridge metrics on the paper largest component
    paper_bridge_stats = compute_bridge_metrics(paper_graph, paper_categories, entity_type='paper')

    # Also analyze the top-N paper network that matches the plotted `combined_paper_network_colored.png`
    topG, top_cats = build_top_n_paper_network(all_papers, top_n=400)
    print("\n=== ANALYSIS ON TOP-400 PAPER NETWORK (plotted figure) ===")
    component_and_reachable_stats(topG, top_cats, entity_type='paper_top400')
    compute_mixing_and_distance_stats(topG, top_cats, entity_type='paper_top400')
    
    print("\n" + "="*80)
    print("👥 AUTHOR COAUTHORSHIP NETWORK ANALYSIS (AUTHORS WITH ≥2 PAPERS)")
    print("="*80)
    
    author_graph, author_categories, author_info = build_author_coauthorship_network(all_papers, min_papers_per_author=2)
    calculate_asp_statistics(author_graph, author_categories, "author")
    compute_mixing_and_distance_stats(author_graph, author_categories, "author")
    
    print("\n" + "="*80)
    print("✅ ASP Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
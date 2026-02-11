#!/usr/bin/env python3
"""
Comprehensive network analysis for safety vs ethics research communities.
Computes detailed structural, separation, and temporal measures.
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import community


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


def build_filtered_networks(papers: List[Dict], min_papers_per_author: int = 3) -> Tuple[Dict, Dict]:
    """Build both paper and author networks with filtering."""
    
    # Filter to safety/ethics papers
    safety_ethics_papers = [p for p in papers if (p.get('safety_score', 0) or 0) > 0 or (p.get('ethics_score', 0) or 0) > 0]
    
    # Count papers per author to filter qualified authors
    author_paper_counts = defaultdict(int)
    for p in safety_ethics_papers:
        for author in p.get('authors', []):
            if author:
                normalized_author = normalize_author(author)
                author_paper_counts[normalized_author] += 1
    
    qualified_authors = {author for author, count in author_paper_counts.items() if count >= min_papers_per_author}
    print(f"Using {len(qualified_authors)} authors who appear in ≥{min_papers_per_author} safety/ethics papers")
    
    # Build paper network
    paper_network = build_paper_network(safety_ethics_papers, qualified_authors)
    
    # Build author network  
    author_network = build_author_network(safety_ethics_papers, qualified_authors)
    
    return paper_network, author_network


def build_paper_network(papers: List[Dict], qualified_authors: Set[str]) -> Dict:
    """Build paper coauthorship network."""
    G = nx.Graph()
    paper_info = {}
    
    # Add paper nodes
    for i, p in enumerate(papers):
        safety_score = float(p.get('safety_score', 0) or 0)
        ethics_score = float(p.get('ethics_score', 0) or 0)
        
        # Categorize papers by PURITY standard (strict)
        if safety_score > 0 and ethics_score == 0:
            category = 'pure_safety'
        elif ethics_score > 0 and safety_score == 0:
            category = 'pure_ethics'
        elif safety_score > 0 and ethics_score > 0:
            category = 'mixed'
        else:
            category = 'neither'
            
        # Filter authors to only qualified ones
        paper_authors = [normalize_author(a) for a in p.get('authors', []) if a and normalize_author(a) in qualified_authors]
        
        if len(paper_authors) > 0:  # Only include papers with qualified authors
            G.add_node(i, 
                      category=category, 
                      safety=safety_score, 
                      ethics=ethics_score,
                      year=p.get('year', 0),
                      authors=paper_authors,
                      title=p.get('title', ''))
            paper_info[i] = {
                'title': p.get('title', ''),
                'safety_score': safety_score,
                'ethics_score': ethics_score,
                'category': category,
                'year': p.get('year', 0),
                'authors': paper_authors
            }
    
    # Add edges between papers with shared authors
    author_to_papers = defaultdict(list)
    for node_id in G.nodes():
        for author in G.nodes[node_id]['authors']:
            author_to_papers[author].append(node_id)
    
    edge_count = 0
    for author, paper_list in author_to_papers.items():
        if len(paper_list) < 2:
            continue
        for i in range(len(paper_list)):
            for j in range(i+1, len(paper_list)):
                u, v = paper_list[i], paper_list[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
                    edge_count += 1
    
    # Get largest connected component
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    print(f"Paper network: {G.number_of_nodes()} papers, {G.number_of_edges()} edges")
    
    return {
        'graph': G,
        'info': paper_info,
        'type': 'paper'
    }


def build_author_network(papers: List[Dict], qualified_authors: Set[str]) -> Dict:
    """Build author coauthorship network."""
    author_scores = defaultdict(lambda: {'safety': 0.0, 'ethics': 0.0, 'papers': set(), 'years': []})
    
    # Aggregate author data
    for idx, p in enumerate(papers):
        safety_score = float(p.get('safety_score', 0) or 0)
        ethics_score = float(p.get('ethics_score', 0) or 0)
        year = p.get('year', 0)
        
        if safety_score > 0 or ethics_score > 0:
            for a in p.get('authors', []):
                if not a:
                    continue
                normalized_author = normalize_author(a)
                
                if normalized_author in qualified_authors:
                    author_scores[normalized_author]['safety'] += safety_score
                    author_scores[normalized_author]['ethics'] += ethics_score
                    author_scores[normalized_author]['papers'].add(idx)
                    if year > 0:
                        author_scores[normalized_author]['years'].append(year)
    
    # Build graph
    G = nx.Graph()
    author_info = {}
    
    for author, scores in author_scores.items():
        safety_total = scores['safety']
        ethics_total = scores['ethics']
        
        if safety_total == 0 and ethics_total == 0:
            continue
            
        # Categorize authors by DOMINANCE standard (which score is higher)
        total_score = safety_total + ethics_total
        if total_score > 0:
            if safety_total > ethics_total:
                category = 'safety_focused'
            elif ethics_total > safety_total:
                category = 'ethics_focused'
            else:  # safety_total == ethics_total
                category = 'balanced'
        else:
            category = 'neither'
            
        G.add_node(author, 
                  category=category, 
                  safety=safety_total, 
                  ethics=ethics_total,
                  num_papers=len(scores['papers']),
                  years=scores['years'])
        
        author_info[author] = {
            'safety_score': safety_total,
            'ethics_score': ethics_total,
            'num_papers': len(scores['papers']),
            'category': category,
            'years': scores['years']
        }
    
    # Add coauthorship edges
    paper_to_authors = defaultdict(list)
    for author, scores in author_scores.items():
        if author not in G.nodes():
            continue
        for paper_idx in scores['papers']:
            paper_to_authors[paper_idx].append(author)
    
    edge_count = 0
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
                    edge_count += 1
    
    # Get largest connected component
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    print(f"Author network: {G.number_of_nodes()} authors, {G.number_of_edges()} edges")
    
    return {
        'graph': G,
        'info': author_info,
        'type': 'author'
    }


def analyze_separation_measures(network: Dict) -> Dict:
    """Compute comprehensive separation measures between safety and ethics nodes."""
    G = network['graph']
    
    # Get node sets by category (different schemes for papers vs authors)
    if network['type'] == 'paper':
        # Papers use purity standard
        safety_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'pure_safety']
        ethics_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'pure_ethics']
        mixed_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'mixed']
        safety_label = "Pure Safety"
        ethics_label = "Pure Ethics" 
        mixed_label = "Mixed"
    else:
        # Authors use dominance standard
        safety_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'safety_focused']
        ethics_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'ethics_focused']
        mixed_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'balanced']
        safety_label = "Safety-Focused"
        ethics_label = "Ethics-Focused"
        mixed_label = "Balanced"
    
    print(f"\n📊 NETWORK COMPOSITION ({network['type'].upper()}):")
    print(f"  {safety_label}: {len(safety_nodes)} nodes")
    print(f"  {ethics_label}: {len(ethics_nodes)} nodes")
    print(f"  {mixed_label}: {len(mixed_nodes)} nodes")
    print(f"  Total: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if len(safety_nodes) == 0 or len(ethics_nodes) == 0:
        return {'error': 'Missing safety or ethics nodes'}
    
    results = {}
    
    # 1. SHORTEST PATH ANALYSIS
    print(f"\n🔍 SHORTEST PATH ANALYSIS:")
    
    # Safety-Ethics cross-group paths
    cross_paths = []
    unreachable_count = 0
    
    for s_node in safety_nodes:
        for e_node in ethics_nodes:
            try:
                path_len = nx.shortest_path_length(G, s_node, e_node)
                cross_paths.append(path_len)
            except nx.NetworkXNoPath:
                unreachable_count += 1
    
    if cross_paths:
        results['cross_paths'] = {
            'mean': np.mean(cross_paths),
            'median': np.median(cross_paths),
            'std': np.std(cross_paths),
            'distribution': dict(Counter(cross_paths)),
            'total_pairs': len(cross_paths) + unreachable_count,
            'reachable_pairs': len(cross_paths),
            'unreachable_pairs': unreachable_count,
            'percent_unreachable': (unreachable_count / (len(cross_paths) + unreachable_count)) * 100,
            'percent_distance_le2': (sum(1 for d in cross_paths if d <= 2) / len(cross_paths)) * 100,
            'percent_distance_ge3': (sum(1 for d in cross_paths if d >= 3) / len(cross_paths)) * 100
        }
        
        print(f"  {safety_label} ↔ {ethics_label}: {len(safety_nodes)} × {len(ethics_nodes)} = {len(safety_nodes) * len(ethics_nodes)} total pairs")
        print(f"  Mean distance: {results['cross_paths']['mean']:.3f}")
        print(f"  Median distance: {results['cross_paths']['median']:.1f}")
        print(f"  Reachable pairs: {results['cross_paths']['reachable_pairs']:,} ({100-results['cross_paths']['percent_unreachable']:.1f}%)")
        print(f"  Distance ≤ 2: {results['cross_paths']['percent_distance_le2']:.1f}%")
        print(f"  Distance ≥ 3: {results['cross_paths']['percent_distance_ge3']:.1f}%")
        print(f"  Distribution: {dict(list(results['cross_paths']['distribution'].items())[:5])}...")
    
    # Within-group paths
    safety_paths = []
    if len(safety_nodes) > 1:
        for i in range(len(safety_nodes)):
            for j in range(i+1, len(safety_nodes)):
                try:
                    path_len = nx.shortest_path_length(G, safety_nodes[i], safety_nodes[j])
                    safety_paths.append(path_len)
                except nx.NetworkXNoPath:
                    pass
        
        if safety_paths:
            results['safety_internal'] = {
                'mean': np.mean(safety_paths),
                'median': np.median(safety_paths)
            }
            print(f"  Within {safety_label}: mean={results['safety_internal']['mean']:.3f}, median={results['safety_internal']['median']:.1f}")
    
    ethics_paths = []
    if len(ethics_nodes) > 1:
        for i in range(len(ethics_nodes)):
            for j in range(i+1, len(ethics_nodes)):
                try:
                    path_len = nx.shortest_path_length(G, ethics_nodes[i], ethics_nodes[j])
                    ethics_paths.append(path_len)
                except nx.NetworkXNoPath:
                    pass
        
        if ethics_paths:
            results['ethics_internal'] = {
                'mean': np.mean(ethics_paths),
                'median': np.median(ethics_paths)
            }
            print(f"  Within {ethics_label}: mean={results['ethics_internal']['mean']:.3f}, median={results['ethics_internal']['median']:.1f}")
    
    # 2. EDGE DENSITY ACROSS COMMUNITIES
    print(f"\n🔗 EDGE DENSITY ANALYSIS:")
    
    # Cross-group edges
    cross_edges = 0
    for s_node in safety_nodes:
        for e_node in ethics_nodes:
            if G.has_edge(s_node, e_node):
                cross_edges += 1
    
    possible_cross_edges = len(safety_nodes) * len(ethics_nodes)
    cross_density = cross_edges / possible_cross_edges if possible_cross_edges > 0 else 0
    
    results['edge_density'] = {
        'cross_edges': cross_edges,
        'possible_cross_edges': possible_cross_edges,
        'cross_density': cross_density
    }
    
    print(f"  Cross-group edges: {cross_edges:,} / {possible_cross_edges:,} = {cross_density:.6f}")
    
    # Within-group densities
    safety_subgraph = G.subgraph(safety_nodes)
    safety_edges = safety_subgraph.number_of_edges()
    possible_safety_edges = len(safety_nodes) * (len(safety_nodes) - 1) // 2
    safety_density = safety_edges / possible_safety_edges if possible_safety_edges > 0 else 0
    
    ethics_subgraph = G.subgraph(ethics_nodes)
    ethics_edges = ethics_subgraph.number_of_edges()
    possible_ethics_edges = len(ethics_nodes) * (len(ethics_nodes) - 1) // 2
    ethics_density = ethics_edges / possible_ethics_edges if possible_ethics_edges > 0 else 0
    
    results['edge_density']['safety_density'] = safety_density
    results['edge_density']['ethics_density'] = ethics_density
    
    print(f"  Within {safety_label}: {safety_edges:,} / {possible_safety_edges:,} = {safety_density:.6f}")
    print(f"  Within {ethics_label}: {ethics_edges:,} / {possible_ethics_edges:,} = {ethics_density:.6f}")
    
    # 3. ASSORTATIVITY BY LABEL
    print(f"\n🎯 ASSORTATIVITY ANALYSIS:")
    
    # Create category mapping for assortativity
    category_map = {n: G.nodes[n]['category'] for n in G.nodes()}
    
    try:
        assortativity = nx.attribute_assortativity_coefficient(G, 'category')
        results['assortativity'] = assortativity
        print(f"  Assortativity by category: {assortativity:.4f}")
        if assortativity > 0.1:
            print(f"    → Strong tendency for same-category nodes to connect")
        elif assortativity > 0.05:
            print(f"    → Moderate tendency for same-category nodes to connect")
        else:
            print(f"    → Weak/no preference for same-category connections")
    except:
        results['assortativity'] = None
        print(f"  Assortativity: Could not calculate")
    
    # 4. BETWEENNESS CENTRALITY OF BRIDGING NODES
    print(f"\n🌉 BRIDGING NODES ANALYSIS:")
    
    betweenness = nx.betweenness_centrality(G)
    
    # Find top bridging nodes (high betweenness centrality)
    top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    
    results['top_bridges'] = []
    for node, centrality in top_bridges:
        node_info = network['info'].get(node, {})
        results['top_bridges'].append({
            'node': node,
            'centrality': centrality,
            'category': G.nodes[node]['category'],
            'info': node_info
        })
    
    print(f"  Top bridging {network['type']}s (by betweenness centrality):")
    for i, bridge in enumerate(results['top_bridges'][:5], 1):
        node_id = bridge['node']
        if network['type'] == 'paper':
            title = bridge['info'].get('title', 'Unknown')[:60] + "..."
            print(f"    {i}. [{bridge['category']}] {title}")
        else:  # author
            print(f"    {i}. [{bridge['category']}] {node_id} ({bridge['info'].get('num_papers', 0)} papers)")
        print(f"       Centrality: {bridge['centrality']:.4f}")
    
    return results


def analyze_community_structure(network: Dict) -> Dict:
    """Analyze community structure and modularity."""
    G = network['graph']
    
    print(f"\n🏘️  COMMUNITY DETECTION:")
    
    results = {}
    
    try:
        # Louvain community detection
        communities = community.louvain_communities(G, seed=42)
        modularity = community.modularity(G, communities)
        
        results['modularity'] = modularity
        results['num_communities'] = len(communities)
        
        print(f"  Modularity: {modularity:.4f}")
        print(f"  Number of communities: {len(communities)}")
        
        # Analyze category distribution within communities
        community_analysis = []
        for i, comm in enumerate(communities):
            if network['type'] == 'paper':
                # Papers use purity categories
                safety_count = sum(1 for node in comm if G.nodes[node]['category'] == 'pure_safety')
                ethics_count = sum(1 for node in comm if G.nodes[node]['category'] == 'pure_ethics')
                mixed_count = sum(1 for node in comm if G.nodes[node]['category'] == 'mixed')
            else:
                # Authors use dominance categories
                safety_count = sum(1 for node in comm if G.nodes[node]['category'] == 'safety_focused')
                ethics_count = sum(1 for node in comm if G.nodes[node]['category'] == 'ethics_focused')
                mixed_count = sum(1 for node in comm if G.nodes[node]['category'] == 'balanced')
            
            total = len(comm)
            
            community_analysis.append({
                'community': i,
                'size': total,
                'safety': safety_count,
                'ethics': ethics_count,
                'mixed': mixed_count,
                'safety_pct': safety_count / total * 100 if total > 0 else 0,
                'ethics_pct': ethics_count / total * 100 if total > 0 else 0,
                'mixed_pct': mixed_count / total * 100 if total > 0 else 0
            })
        
        # Sort by community size
        community_analysis.sort(key=lambda x: x['size'], reverse=True)
        results['community_analysis'] = community_analysis
        
        print(f"  Largest communities:")
        for i, comm in enumerate(community_analysis[:5], 1):
            print(f"    {i}. Size: {comm['size']:3d} | Safety: {comm['safety_pct']:4.1f}% | Ethics: {comm['ethics_pct']:4.1f}% | Mixed: {comm['mixed_pct']:4.1f}%")
        
    except Exception as e:
        print(f"  Community detection failed: {e}")
        results['error'] = str(e)
    
    return results


def analyze_temporal_patterns(network: Dict) -> Dict:
    """Analyze temporal patterns in connectivity."""
    G = network['graph']
    
    print(f"\n📅 TEMPORAL ANALYSIS:")
    
    results = {}
    
    # Get years for each node
    node_years = {}
    for node in G.nodes():
        if network['type'] == 'paper':
            year = G.nodes[node].get('year', 0)
            if year > 0:
                node_years[node] = year
        else:  # author
            years = G.nodes[node].get('years', [])
            if years:
                node_years[node] = {
                    'first': min(years),
                    'last': max(years),
                    'span': max(years) - min(years) + 1
                }
    
    if network['type'] == 'paper':
        # Analyze cross-category connections by year
        safety_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'pure_safety']
        ethics_nodes = [n for n in G.nodes() if G.nodes[n]['category'] == 'pure_ethics']
        
        cross_edges_by_year = defaultdict(int)
        total_edges_by_year = defaultdict(int)
        
        for edge in G.edges():
            node1, node2 = edge
            year1 = node_years.get(node1, 0)
            year2 = node_years.get(node2, 0)
            
            if year1 > 0 and year2 > 0:
                # Use the later year for the edge
                edge_year = max(year1, year2)
                total_edges_by_year[edge_year] += 1
                
                # Check if it's a cross-category edge
                cat1 = G.nodes[node1]['category']
                cat2 = G.nodes[node2]['category']
                
                if ((cat1 == 'pure_safety' and cat2 == 'pure_ethics') or 
                    (cat1 == 'pure_ethics' and cat2 == 'pure_safety')):
                    cross_edges_by_year[edge_year] += 1
        
        # Calculate cross-connection rates by year
        cross_rates = {}
        for year in sorted(total_edges_by_year.keys()):
            if total_edges_by_year[year] > 0:
                cross_rates[year] = cross_edges_by_year[year] / total_edges_by_year[year] * 100
        
        results['cross_rates_by_year'] = cross_rates
        
        print(f"  Cross-category connection rates by year:")
        for year in sorted(cross_rates.keys())[-10:]:  # Last 10 years
            print(f"    {year}: {cross_rates[year]:.2f}% ({cross_edges_by_year[year]} / {total_edges_by_year[year]} edges)")
    
    else:  # author network
        # Analyze author career spans and collaboration patterns
        career_spans = [data['span'] for data in node_years.values() if isinstance(data, dict)]
        
        if career_spans:
            results['career_spans'] = {
                'mean': np.mean(career_spans),
                'median': np.median(career_spans),
                'max': max(career_spans)
            }
            
            print(f"  Author career spans:")
            print(f"    Mean: {results['career_spans']['mean']:.1f} years")
            print(f"    Median: {results['career_spans']['median']:.1f} years")
            print(f"    Max: {results['career_spans']['max']} years")
    
    return results


def main():
    """Main analysis function."""
    print("🔍 Loading combined ACL and DBLP data...")
    
    acl_path = 'data/acl_anthology_papers_v2.jsonl'
    dblp_path = 'data/dblp_conferences_papers.jsonl'
    
    acl_papers = load_jsonl(acl_path)
    dblp_papers = load_jsonl(dblp_path)
    
    print(f"Loaded {len(acl_papers)} ACL papers and {len(dblp_papers)} DBLP papers")
    
    # Normalize authors for both datasets
    all_papers = []
    for p in acl_papers + dblp_papers:
        authors = []
        for a in p.get('authors', []):
            a = (a or '').strip()
            if not a:
                continue
            authors.append(normalize_author(a))
        p['authors'] = authors
        all_papers.append(p)
    
    print(f"Total papers: {len(all_papers)}")
    
    # Build networks
    paper_network, author_network = build_filtered_networks(all_papers, min_papers_per_author=2)
    
    # Analyze both networks
    for network_name, network in [("PAPER", paper_network), ("AUTHOR", author_network)]:
        print(f"\n" + "="*80)
        print(f"📊 {network_name} NETWORK COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # Separation measures
        separation_results = analyze_separation_measures(network)
        
        # Community structure
        community_results = analyze_community_structure(network)
        
        # Temporal patterns
        temporal_results = analyze_temporal_patterns(network)
        
        print(f"\n✅ {network_name} network analysis complete!")
    
    print(f"\n" + "="*80)
    print("🎯 COMPREHENSIVE NETWORK ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
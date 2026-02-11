#!/usr/bin/env python3
"""
Export a fixed-position combined author co-authorship network for D3.js.
Nodes are positioned with strict left(-blue) ↔ right(+red) forcing based on
the author safety proportion, and include top papers per author.
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx


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


def filter_top_papers(papers: List[Dict], top_n: int = 300) -> List[Dict]:
    """Filter papers to only include the top N papers by citations or relevance scores."""
    # First try citations if available
    papers_with_scores = []
    has_citations = False
    
    for p in papers:
        citations = p.get('citations', 0)
        if citations is None:
            citations = 0
        elif isinstance(citations, str):
            try:
                citations = int(citations)
            except (ValueError, TypeError):
                citations = 0
        
        if citations > 0:
            has_citations = True
        
        # If no citations, use safety + ethics scores as relevance measure
        safety_score = p.get('safety_score', 0) or 0
        ethics_score = p.get('ethics_score', 0) or 0
        relevance_score = safety_score + ethics_score
        
        # Use citations if available, otherwise use relevance score
        score = citations if has_citations else relevance_score
        papers_with_scores.append((p, score, citations, relevance_score))
    
    # Sort by score descending and take top N
    papers_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_papers = [p for p, _, _, _ in papers_with_scores[:top_n]]
    
    if has_citations:
        print(f"Filtered to top {len(top_papers)} papers by citations")
        if papers_with_scores:
            max_citations = papers_with_scores[0][2]
            min_citations = papers_with_scores[min(top_n-1, len(papers_with_scores)-1)][2] if len(papers_with_scores) >= top_n else papers_with_scores[-1][2]
            print(f"Citation range: {min_citations} - {max_citations}")
    else:
        print(f"Filtered to top {len(top_papers)} papers by relevance scores (safety + ethics)")
        if papers_with_scores:
            max_relevance = papers_with_scores[0][3]
            min_relevance = papers_with_scores[min(top_n-1, len(papers_with_scores)-1)][3] if len(papers_with_scores) >= top_n else papers_with_scores[-1][3]
            print(f"Relevance score range: {min_relevance} - {max_relevance}")
    
    return top_papers


def normalize_author(name: str) -> str:
    if not name:
        return name
    s = name.strip()
    parts = s.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 3:
        parts = parts[:-1]
    return ' '.join(parts)


def build_author_graph(papers: List[Dict]) -> Tuple[nx.Graph, Dict[str, Dict]]:
    """Return (graph, author_info) with aggregated scores and paper lists per author."""
    author_info: Dict[str, Dict] = defaultdict(lambda: {
        's': 0.0, 'e': 0.0, 'papers': []
    })
    # Aggregate authors and collect their papers
    for p in papers:
        s = float(p.get('safety_score', 0) or 0)
        e = float(p.get('ethics_score', 0) or 0)
        title = p.get('title', '')
        year = int(p.get('year') or 0)
        venue = p.get('venue', '')
        url = p.get('url') or p.get('ee') or ''
        for a in p.get('authors', []):
            if not a:
                continue
            author_info[a]['s'] += s
            author_info[a]['e'] += e
            author_info[a]['papers'].append({
                'title': title,
                'year': year,
                'venue': venue,
                's': s,
                'e': e,
                'url': url,
            })

    # Build co-authorship graph
    G = nx.Graph()
    for a, info in author_info.items():
        total = info['s'] + info['e']
        if total <= 0:
            # Skip authors with neither safety nor ethics signal
            continue
        G.add_node(a, s=info['s'], e=info['e'])

    paper_to_authors: Dict[int, List[str]] = {}
    for idx, p in enumerate(papers):
        auths = [a for a in p.get('authors', []) if a in G.nodes]
        if len(auths) >= 2:
            paper_to_authors[idx] = auths
    for auths in paper_to_authors.values():
        for i in range(len(auths)):
            for j in range(i+1, len(auths)):
                u, v = auths[i], auths[j]
                if G.has_edge(u, v):
                    G[u][v]['w'] += 1
                else:
                    G.add_edge(u, v, w=1)

    # Retain only authors present in G
    author_info = {a: author_info[a] for a in G.nodes}
    return G, author_info


def force_positions(G: nx.Graph, bins: int = 20) -> Dict[str, Tuple[float, float]]:
    """Compute fixed positions with strict left/right forcing using safety proportion.
    Ensures zero node overlap through collision detection and avoidance.
    Returns mapping: node -> (x,y), with x in [-1,1] and y in [-1,1].
    """
    import random
    import math
    random.seed(42)  # For reproducible results
    
    # Calculate node sizes based on degree (matching the frontend)
    degrees = dict(G.degree())
    node_radii = {}
    for n in G.nodes:
        s = float(G.nodes[n].get('s', 0.0))
        e = float(G.nodes[n].get('e', 0.0))
        total = s + e
        # Match the radius calculation from the HTML: r = 4 + Math.min(10, Math.sqrt(d.degree) + (d.total||0))
        radius = 4 + min(10, math.sqrt(degrees.get(n, 0)) + total)
        node_radii[n] = radius
    
    # Bin nodes by safety proportion
    bins_map: Dict[int, List[str]] = defaultdict(list)
    safety_prop: Dict[str, float] = {}
    for n in G.nodes:
        s = float(G.nodes[n].get('s', 0.0))
        e = float(G.nodes[n].get('e', 0.0))
        t = s + e
        p = 0.5 if t == 0 else (s / t)
        safety_prop[n] = p
        idx = max(0, min(bins - 1, int(p * bins)))
        bins_map[idx].append(n)

    # Within each bin, order authors by degree desc, then name
    for b in range(bins):
        nodes = bins_map.get(b, [])
        if nodes:
            nodes.sort(key=lambda n: (-(degrees.get(n, 0)), n))
            bins_map[b] = nodes
    
    pos: Dict[str, Tuple[float, float]] = {}
    placed_positions = []  # List of (x, y, radius) for collision detection
    
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def has_collision(x, y, radius, placed):
        for px, py, pr in placed:
            if distance((x, y), (px, py)) < (radius + pr) * 1.1:  # 10% padding
                return True
        return False
    
    def find_collision_free_position(target_x, target_y, radius, placed, x_range, max_attempts=20):
        """Find a collision-free position near the target position"""
        min_x, max_x = x_range
        
        # Try the exact target first
        if not has_collision(target_x, target_y, radius, placed):
            return target_x, target_y
        
        # Simple grid-based search with fewer attempts
        for attempt in range(max_attempts):
            # Increase search radius with each attempt
            search_radius = (attempt + 1) * 0.05
            
            # Try fewer angles at this radius
            for angle_step in range(4):
                angle = (angle_step / 4) * 2 * math.pi
                x = target_x + search_radius * math.cos(angle)
                y = target_y + search_radius * math.sin(angle)
                
                # Constrain to valid ranges
                x = max(min_x, min(max_x, x))
                y = max(-1.0, min(1.0, y))
                
                if not has_collision(x, y, radius, placed):
                    return x, y
        
        # If all else fails, place sequentially with guaranteed spacing
        y_offset = len(placed) * 0.03
        return max(min_x, min(max_x, target_x)), max(-1.0, min(1.0, target_y + y_offset))
    
    # Calculate viewport scaling to match frontend (pad = 40, width/height from visualization)
    # We'll normalize radii to coordinate space
    viewport_width = 800  # Approximate viewport width
    coord_to_pixel_scale = viewport_width / 2.0  # Since coords are [-1,1]
    
    # Process bins from left to right
    for b in range(bins):
        nodes = bins_map.get(b, [])
        if not nodes:
            continue
        
        # Calculate bin boundaries
        bin_width = 2.0 / bins
        bin_left = -1.0 + b * bin_width
        bin_right = -1.0 + (b + 1) * bin_width
        bin_center = (bin_left + bin_right) / 2
        
        # Allow some expansion beyond bin boundaries for collision avoidance
        expanded_left = max(-1.0, bin_left - bin_width * 0.3)
        expanded_right = min(1.0, bin_right + bin_width * 0.3)
        
        for i, n in enumerate(nodes):
            # Calculate normalized radius for collision detection
            pixel_radius = node_radii[n]
            coord_radius = pixel_radius / coord_to_pixel_scale
            
            # Base position within bin
            k = len(nodes)
            base_y = -1.0 + 2.0 * (i + 0.5) / max(1, k)
            target_x = bin_center
            
            # Find collision-free position
            final_x, final_y = find_collision_free_position(
                target_x, base_y, coord_radius, placed_positions, 
                (expanded_left, expanded_right)
            )
            
            pos[n] = (final_x, final_y)
            placed_positions.append((final_x, final_y, coord_radius))
    
    return pos


def export_web_data(out_dir: str = 'web', top_authors: int = 1000):
    os.makedirs(out_dir, exist_ok=True)
    # Load combined datasets
    acl = load_jsonl('data/acl_anthology_papers_v2.jsonl')
    dblp = load_jsonl('data/dblp_conferences_papers.jsonl')
    collected = load_jsonl('data/collected_papers.jsonl')
    
    print(f"Loaded {len(acl)} ACL papers, {len(dblp)} DBLP papers, {len(collected)} collected papers")
    
    all_papers: List[Dict] = []
    for p in acl + dblp + collected:
        # Filter to only papers with safety/ethics scores > 0 to reduce size
        safety_score = p.get('safety_score', 0) or 0
        ethics_score = p.get('ethics_score', 0) or 0
        if safety_score > 0 or ethics_score > 0:
            # Normalize authors field (different sources may have different formats)
            if 'authors' in p and isinstance(p['authors'], list):
                if p['authors'] and isinstance(p['authors'][0], dict):
                    # Handle author objects with 'name' field (from collected_papers)
                    p['authors'] = [normalize_author(a.get('name', '')) for a in p['authors'] if a.get('name', '').strip()]
                else:
                    # Handle simple author name lists (from ACL/DBLP)
                    p['authors'] = [normalize_author(a) for a in p['authors'] if (a or '').strip()]
            else:
                p['authors'] = []
            all_papers.append(p)
    
    print(f"Filtered to {len(all_papers)} papers with safety/ethics scores > 0")
    
    # Build the full graph first to calculate degrees
    G_full, author_info_full = build_author_graph(all_papers)
    
    # Filter to top authors by degree (number of collaborations)
    degrees = dict(G_full.degree())
    top_authors_list = sorted(degrees.keys(), key=lambda a: degrees[a], reverse=True)[:top_authors]
    
    print(f"Filtered to top {len(top_authors_list)} authors by collaboration degree")
    if top_authors_list:
        max_degree = degrees[top_authors_list[0]]
        min_degree = degrees[top_authors_list[-1]]
        print(f"Degree range: {min_degree} - {max_degree} collaborations")
    
    # Filter papers to only include those with at least one top author
    papers = []
    for p in all_papers:
        authors = p.get('authors', [])
        if any(a in top_authors_list for a in authors):
            papers.append(p)
    
    # Build filtered graph
    G, author_info = build_author_graph(papers)
    
    # Keep only the top authors in the final graph
    nodes_to_remove = [n for n in G.nodes if n not in top_authors_list]
    G.remove_nodes_from(nodes_to_remove)
    author_info = {a: author_info[a] for a in G.nodes}
    pos = force_positions(G, bins=24)
    
    # Calculate cross-community connectivity metrics
    def calculate_cross_connectivity(G, author_info):
        """Calculate how well each author is connected to the 'other side'"""
        metrics = {}
        
        for author in G.nodes:
            s = float(G.nodes[author].get('s', 0.0))
            e = float(G.nodes[author].get('e', 0.0))
            total = s + e
            
            if total == 0:
                author_side = 'neutral'
                author_proportion = 0.5
            else:
                safety_prop = s / total
                author_side = 'safety' if safety_prop > 0.6 else ('ethics' if safety_prop < 0.4 else 'neutral')
                author_proportion = safety_prop
            
            # Count connections to each side
            safety_connections = 0
            ethics_connections = 0
            neutral_connections = 0
            total_connections = 0
            
            for neighbor in G.neighbors(author):
                neighbor_s = float(G.nodes[neighbor].get('s', 0.0))
                neighbor_e = float(G.nodes[neighbor].get('e', 0.0))
                neighbor_total = neighbor_s + neighbor_e
                
                if neighbor_total == 0:
                    neutral_connections += 1
                else:
                    neighbor_safety_prop = neighbor_s / neighbor_total
                    if neighbor_safety_prop > 0.6:
                        safety_connections += 1
                    elif neighbor_safety_prop < 0.4:
                        ethics_connections += 1
                    else:
                        neutral_connections += 1
                
                total_connections += 1
            
            # Calculate cross-community metrics
            if author_side == 'safety':
                cross_connections = ethics_connections
                same_side_connections = safety_connections
            elif author_side == 'ethics':
                cross_connections = safety_connections
                same_side_connections = ethics_connections
            else:  # neutral
                cross_connections = max(safety_connections, ethics_connections)
                same_side_connections = neutral_connections
            
            # Bridge score: how much they connect different communities
            bridge_score = cross_connections / max(1, total_connections)
            
            # Diversity score: how evenly distributed their connections are
            if total_connections > 0:
                safety_ratio = safety_connections / total_connections
                ethics_ratio = ethics_connections / total_connections
                neutral_ratio = neutral_connections / total_connections
                # Shannon entropy-like diversity measure
                entropy_parts = []
                for ratio in [safety_ratio, ethics_ratio, neutral_ratio]:
                    if ratio > 0:
                        entropy_parts.append(-ratio * math.log2(ratio))
                diversity_score = sum(entropy_parts) / math.log2(3)  # Normalize to [0,1]
            else:
                diversity_score = 0
            
            metrics[author] = {
                'side': author_side,
                'safety_proportion': author_proportion,
                'total_connections': total_connections,
                'safety_connections': safety_connections,
                'ethics_connections': ethics_connections,
                'neutral_connections': neutral_connections,
                'cross_connections': cross_connections,
                'same_side_connections': same_side_connections,
                'bridge_score': bridge_score,
                'diversity_score': diversity_score
            }
        
        return metrics
    
    import math
    cross_metrics = calculate_cross_connectivity(G, author_info)

    # Prepare nodes
    nodes = []
    degrees = dict(G.degree())
    for n in G.nodes:
        s = float(G.nodes[n].get('s', 0.0))
        e = float(G.nodes[n].get('e', 0.0))
        total = s + e
        ainfo = author_info[n]
        cross_info = cross_metrics.get(n, {})
        
        # Top papers by total score
        top_papers = sorted(ainfo['papers'], key=lambda x: (x['s'] + x['e']), reverse=True)[:5]
        nodes.append({
            'id': n,
            's': s,
            'e': e,
            'total': total,
            'degree': degrees.get(n, 0),
            'x': pos[n][0],
            'y': pos[n][1],
            'papers': top_papers,
            # Cross-community analytics
            'side': cross_info.get('side', 'neutral'),
            'safety_proportion': cross_info.get('safety_proportion', 0.5),
            'bridge_score': cross_info.get('bridge_score', 0),
            'diversity_score': cross_info.get('diversity_score', 0),
            'cross_connections': cross_info.get('cross_connections', 0),
            'same_side_connections': cross_info.get('same_side_connections', 0),
            'safety_connections': cross_info.get('safety_connections', 0),
            'ethics_connections': cross_info.get('ethics_connections', 0),
        })

    # Prepare links
    links = []
    for u, v, d in G.edges(data=True):
        links.append({'source': u, 'target': v, 'w': int(d.get('w', 1))})

    data = {'nodes': nodes, 'links': links}
    with open(os.path.join(out_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Wrote {len(nodes)} nodes and {len(links)} links to {os.path.join(out_dir, 'data.json')}")


def main():
    export_web_data('web', top_authors=1000)


if __name__ == '__main__':
    main()



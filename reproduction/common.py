"""Shared loading, labelling, graph construction and metrics for the reproduction scripts.

Two versions are kept side by side wherever the released code differs from the paper's text:

  legacy            -- what the released scripts actually compute (reproduces the paper's numbers
                       where possible).
  paper-definition  -- what Sections 3.1-3.2 of the paper describe.
"""
import os
import sys
from collections import Counter, defaultdict

import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analyze_aggregated import load_combined_data  # noqa: E402
from network_asp_analysis import build_author_coauthorship_network, normalize_author  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
S, E, M = 'pure_safety', 'pure_ethics', 'mixed'


def load_papers(corpus='full'):
    return load_combined_data(corpus=corpus)


def category(safety, ethics):
    if safety > 0 and ethics == 0:
        return S
    if ethics > 0 and safety == 0:
        return E
    if safety > 0 and ethics > 0:
        return M
    return 'neither'


def write_csv(name, header, rows):
    import csv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ---------------------------------------------------------------- legacy author graph (Table 2)

def legacy_author_categories(papers, min_papers=2):
    """unified_homophily_removal_analysis.load_author_categories: raw (un-normalized) names."""
    scores = defaultdict(lambda: [0.0, 0.0, 0])
    for p in papers:
        for a in p.get('authors', []):
            scores[a][0] += p.get('safety_score', 0)
            scores[a][1] += p.get('ethics_score', 0)
            scores[a][2] += 1
    cats = {}
    for a, (s, e, n) in scores.items():
        if n >= min_papers:
            c = category(s, e)
            cats[a] = M if c == 'neither' else c
    return cats


def legacy_author_graph(papers, removed=frozenset()):
    """Exact (and much faster) port of unified_homophily_removal_analysis.load_author_graph.

    For every author X and every pair of X's papers (i, j), every author of paper i is linked to
    every author of paper j -- i.e. authors are linked when they share a co-author, not only when
    they co-author. Endpoints need >= 2 papers and must not be in `removed`; removed authors still
    act as the shared co-author X. Only the largest connected component is kept, which drops all
    ACL authors ("Last, First" names never match DBLP's "First Last").
    """
    author_papers = defaultdict(list)
    for p in papers:
        pid = p.get('paper_id') or p.get('id') or str(hash(str(p)))
        for a in p.get('authors', []):
            author_papers[a].append(pid)
    keep = {a for a, ps in author_papers.items() if len(ps) >= 2 and a not in removed}
    paper_authors = defaultdict(set)
    for a, ps in author_papers.items():
        for pid in ps:
            paper_authors[pid].add(a)
    paper_authors = {pid: sorted(aa & keep) for pid, aa in paper_authors.items()}

    weights = Counter()
    for ps in author_papers.values():
        for i in range(len(ps)):
            first = paper_authors[ps[i]]
            if not first:
                continue
            for j in range(i + 1, len(ps)):
                for a1 in first:
                    for a2 in paper_authors[ps[j]]:
                        if a1 != a2:
                            weights[(a1, a2) if a1 < a2 else (a2, a1)] += 1
    G = nx.Graph()
    for (u, v), w in weights.items():
        G.add_edge(u, v, weight=w)
    if G.number_of_nodes() and not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def legacy_paper_graph(papers, removed=frozenset()):
    """unified_homophily_removal_analysis.load_paper_graph: raw names, largest component only."""
    G, cats = paper_graph(papers, removed, name=lambda a: a, dedupe=False)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        cats = {n: cats[n] for n in G}
    return G, cats


# ---------------------------------------------------------------- graphs as the paper describes

def coauthorship_author_graph(papers, min_papers=2):
    """Direct co-authorship among authors with >= min_papers papers (network_asp_analysis builder)."""
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        G, cats, _ = build_author_coauthorship_network(papers, min_papers_per_author=min_papers)
    return G, cats


def paper_graph(papers, removed=frozenset(), name=normalize_author, dedupe=True):
    """Papers linked when they share an author; weight = number of shared authors. All papers kept.

    dedupe=False reproduces the released code, where an author listed twice on a paper adds a self-loop.
    """
    G = nx.Graph()
    cats = {}
    author_to_papers = defaultdict(list)
    for i, p in enumerate(papers):
        if i in removed:
            continue
        G.add_node(i)
        cats[i] = category(p.get('safety_score', 0), p.get('ethics_score', 0))
        names = [name(a) for a in p.get('authors', [])]
        for a in (dict.fromkeys(names) if dedupe else names):
            author_to_papers[a].append(i)
    for plist in author_to_papers.values():
        for x in range(len(plist)):
            for y in range(x + 1, len(plist)):
                u, v = plist[x], plist[y]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    return G, cats


def remove_top_degree(G, n):
    order = [x for x, _ in sorted(G.degree(), key=lambda t: -t[1])]
    H = G.copy()
    H.remove_nodes_from(order[:n])
    return H


# ---------------------------------------------------------------- homophily metrics

def edge_type_counts(G, cats, weighted=False):
    counts = Counter()
    for u, v, d in G.edges(data=True):
        counts[frozenset((cats.get(u, M), cats.get(v, M)))] += d.get('weight', 1) if weighted else 1
    return counts


def homophily_legacy(G, cats, weighted=False):
    """compute_author_homophily / compute_paper_homophily: edges touching a mixed node are ignored.

    Returns (global, safety, ethics) in percent.
    """
    c = edge_type_counts(G, cats, weighted)
    ss, ee, se = c[frozenset({S})], c[frozenset({E})], c[frozenset({S, E})]
    pct = lambda a, b: 100 * a / b if b else float('nan')
    return pct(ss + ee, ss + ee + se), pct(ss, ss + se), pct(ee, ee + se)


def homophily_paper_definition(G, cats, weighted=False):
    """Sec 3.2.1: fraction of ALL edges (or edge weight) joining safety-safety or ethics-ethics.

    Per group: share of the edges touching a pure-safety (pure-ethics) node that go to another
    pure-safety (pure-ethics) node. Returns (global, safety, ethics) in percent.
    """
    c = edge_type_counts(G, cats, weighted)
    ss, ee, se = c[frozenset({S})], c[frozenset({E})], c[frozenset({S, E})]
    sm, em = c[frozenset({S, M})], c[frozenset({E, M})]
    total = sum(c.values())
    pct = lambda a, b: 100 * a / b if b else float('nan')
    return pct(ss + ee, total), pct(ss, ss + se + sm), pct(ee, ee + se + em)

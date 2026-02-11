"""Filter out the top greedy-attack authors (from greedy_attack_sequence_minp2.csv) and analyze how clusters and reachability change.

Reads perturb_grid_out/greedy_attack_sequence_minp2.csv, removes top-N authors, rebuilds author graph (min_p=2), computes component sizes, mixed component counts, reachability_by_hops (k=6), and writes summary CSV to perturb_grid_out/greedy_removals_cluster_effects_minp2.csv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv, time
from collections import Counter
import networkx as nx
from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network, reachability_by_hops

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)


def remove_authors_from_papers(papers, remove_set):
    new = []
    for p in papers:
        new_auths = []
        for a in p.get('authors', []):
            name = a.get('name') if isinstance(a, dict) else a
            if name not in remove_set:
                new_auths.append(a)
        if new_auths:
            np = dict(p); np['authors'] = new_auths; new.append(np)
    return new


def analyze(G, cats):
    # components
    comps = list(nx.connected_components(G))
    sizes = sorted([len(c) for c in comps], reverse=True)
    largest = sizes[0] if sizes else 0
    total = G.number_of_nodes()
    frac_in_largest = largest / total if total else 0.0
    # mixed components
    mixed_comps = 0
    pure_only = 0
    for c in comps:
        cats_in = {cats.get(n,'neither') for n in c}
        if 'pure_safety' in cats_in and 'pure_ethics' in cats_in:
            mixed_comps += 1
        elif cats_in.issubset({'pure_safety','neither'}):
            pure_only += 1
    # reachability
    reach = reachability_by_hops(G, cats, max_k=6)
    se5 = reach['se_frac_by_k'][4]
    ss5 = reach['ss_frac_by_k'][4]
    ee5 = reach['ee_frac_by_k'][4]
    return {
        'nodes': total,
        'edges': G.number_of_edges(),
        'num_components': len(comps),
        'largest_component_size': largest,
        'frac_in_largest': frac_in_largest,
        'mixed_components': mixed_comps,
        'pure_only_components': pure_only,
        'se@5': se5,
        'ss@5': ss5,
        'ee@5': ee5
    }


def run(top_n=100, min_p=2):
    papers = load_combined_data()
    # read greedy removals
    greedy_csv = os.path.join(OUT, 'greedy_attack_sequence_minp2.csv')
    removals = []
    with open(greedy_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            removals.append(row['removed_node'])
    to_remove = set(removals[:top_n])
    print('Top', top_n, 'to remove (from greedy):', list(removals[:min(top_n,10)]))

    # baseline graph
    G_base, cats_base, info_base = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    print('Baseline authors:', G_base.number_of_nodes())
    base_stats = analyze(G_base, cats_base)

    # filtered graph with removals applied (remove from papers then rebuild graph)
    filtered_papers = remove_authors_from_papers(papers, to_remove)
    G_filtered, cats_filtered, info_filtered = build_author_coauthorship_network(filtered_papers, min_papers_per_author=min_p)
    print('After removal authors:', G_filtered.number_of_nodes())
    filtered_stats = analyze(G_filtered, cats_filtered)

    outcsv = os.path.join(OUT, f'greedy_removals_cluster_effects_minp{min_p}.csv')
    with open(outcsv,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['metric','baseline','after_removal','delta'])
        for k in base_stats.keys():
            b = base_stats[k]
            a = filtered_stats[k]
            d = b - a if isinstance(b,(int,float)) and isinstance(a,(int,float)) else ''
            w.writerow([k,b,a,d])
    print('Wrote', outcsv)
    print('\nBASELINE:', base_stats)
    print('\nAFTER REMOVAL:', filtered_stats)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--min_p', type=int, default=2)
    args = parser.parse_args()
    start=time.time(); run(top_n=args.top_n, min_p=args.min_p); print('Done in', time.time()-start)

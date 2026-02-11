#!/usr/bin/env python3
"""Compute k-hop reachability for paper graph after removing mixed papers.

Builds the full paper coauthorship graph (min_p=0), removes nodes labeled 'mixed', and computes reachability_by_hops for k=1..12.
Writes CSV summary and saves a plot.
"""
from network_asp_analysis import load_jsonl, build_paper_coauthorship_network, reachability_by_hops
import matplotlib.pyplot as plt
import csv


def main():
    # load papers via analyze_aggregated loader to match earlier runs
    from analyze_aggregated import load_combined_data
    papers = load_combined_data()

    # build full paper graph with no author filtering
    G, categories, info = build_paper_coauthorship_network(papers, min_papers_per_author=0)
    print(f"Built paper graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # remove mixed nodes
    keep = [n for n, c in categories.items() if c in ('pure_safety','pure_ethics')]
    H = G.subgraph(keep).copy()
    new_cat = {n: categories[n] for n in H.nodes()}
    print(f"After removing mixed: nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")

    max_k = 12
    res = reachability_by_hops(H, new_cat, max_k=max_k)

    # write CSV
    out_csv = 'reachability_paper_nomixed_k12.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['k','SE_frac','SS_frac','EE_frac','Ns','Ne','total_SE','total_SS','total_EE'])
        for i in range(max_k):
            writer.writerow([i+1, res['se_frac_by_k'][i], res['ss_frac_by_k'][i], res['ee_frac_by_k'][i], res['Ns'], res['Ne'], res['total_SE'], res['total_SS'], res['total_EE']])

    # plot percentages
    ks = list(range(1, max_k+1))
    plt.figure(figsize=(8,5))
    plt.plot(ks, [f*100 for f in res['ss_frac_by_k']], marker='o', label='Safety–Safety')
    plt.plot(ks, [f*100 for f in res['se_frac_by_k']], marker='o', label='Safety–Ethics')
    plt.plot(ks, [f*100 for f in res['ee_frac_by_k']], marker='o', label='Ethics–Ethics')
    plt.xticks(ks)
    plt.xlabel('hops (k)')
    plt.ylabel('Reachable pairs (%)')
    plt.title('Paper reachability (no mixed) - <=k hops')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('reachability_paper_nomixed_k12.png', dpi=200)
    plt.close()

    print('Wrote:', out_csv, 'and reachability_paper_nomixed_k12.png')
    print('Ns, Ne:', res['Ns'], res['Ne'])
    print('Safety–Ethics reachable % by k:', [f*100 for f in res['se_frac_by_k']])


if __name__ == '__main__':
    main()

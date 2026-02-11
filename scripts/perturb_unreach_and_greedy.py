#!/usr/bin/env python3
"""Greedy attack that maximizes newly unreachable Safety-Ethics author pairs.
This is a lightweight version with a time limit to keep runs short.
"""
import argparse, csv, os, time, random
from collections import defaultdict
import networkx as nx

# Import helpers from the repo
from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network


def remove_authors_from_papers(papers, remove_set):
    """Return a new papers list with authors in remove_set removed from author lists."""
    new = []
    for p in papers:
        new_auths = []
        for a in p.get('authors', []):
            # author entries may be dicts with 'name' or plain strings
            name = a.get('name') if isinstance(a, dict) else a
            if name not in remove_set:
                new_auths.append(a)
        if new_auths:
            np = dict(p)
            np['authors'] = new_auths
            new.append(np)
    return new

OUT = "perturb_grid_out"
os.makedirs(OUT, exist_ok=True)


def reachable_pairs(G, cats, S_label='pure_safety', E_label='pure_ethics'):
    S = [n for n,c in cats.items() if c==S_label]
    E = [n for n,c in cats.items() if c==E_label]
    comp = {n: cid for cid, comp_nodes in enumerate(nx.connected_components(G)) for n in comp_nodes}
    same_comp = 0
    total = 0
    for s in S:
        for e in E:
            total += 1
            if comp.get(s) == comp.get(e):
                same_comp += 1
    return {'total': total, 'reachable': same_comp, 'unreachable': total - same_comp}


def greedy_max_unreachable(papers, min_p, limit=20, candidate_pool=50, time_limit=160):
    start = time.time()
    G_auth, cats, info = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    base = reachable_pairs(G_auth, cats)
    print("Baseline reachable S-E pairs:", base['reachable'], "of", base['total'])
    removed = set()
    seq = []

    # heuristic ranking
    deg = dict(G_auth.degree())
    cross_deg = {}
    for n in G_auth.nodes():
        myc = cats.get(n)
        cross_deg[n] = sum(1 for nbr in G_auth[n] if cats.get(nbr) != myc)
    scores = {n: deg.get(n,0) * cross_deg.get(n,0) for n in G_auth.nodes()}
    ranked = [n for n,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    for step in range(1, limit+1):
        elapsed = time.time() - start
        if elapsed > time_limit:
            print(f"Time limit exceeded ({elapsed:.1f}s). Stopping at step {step-1}.")
            break
        shortlist = [n for n in ranked if n not in removed][:candidate_pool]
        if not shortlist:
            print('No more candidates')
            break
        best_c = None
        best_gain = 0
        best_r = None
        # evaluate each candidate (rebuild filtered graph and count reachable pairs)
        for cand in shortlist:
            cand_removed = removed.union({cand})
            filtered = remove_authors_from_papers(papers, cand_removed)
            try:
                G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
            except Exception:
                continue
            r = reachable_pairs(G_r, cats_r)
            gain = base['reachable'] - r['reachable']
            if gain > best_gain:
                best_gain = gain
                best_c = cand
                best_r = r
        if best_c is None:
            print('No improving candidate found in shortlist')
            break
        removed.add(best_c)
        base['reachable'] = best_r['reachable']
        seq.append({'step': step, 'removed': best_c, 'reachable_after': best_r['reachable'], 'lost_pairs': best_gain})
        print(f"step {step}: remove {best_c} -> lost_pairs {best_gain} reachable_now {best_r['reachable']} elapsed {time.time()-start:.1f}s")
    out = os.path.join(OUT, f'greedy_unreach_minp{min_p}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['step','removed','reachable_after','lost_pairs'])
        writer.writeheader(); writer.writerows(seq)
    print('Wrote', out)
    return seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_p', type=int, default=2)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--candidate_pool', type=int, default=50)
    parser.add_argument('--time_limit', type=int, default=160, help='seconds')
    args = parser.parse_args()

    papers = load_combined_data()
    # quick baseline
    G, cats, info = build_author_coauthorship_network(papers, min_papers_per_author=args.min_p)
    base_reach = reachable_pairs(G, cats)
    print('Baseline reachable S-E pairs:', base_reach)

    greedy_max_unreachable(papers, min_p=args.min_p, limit=args.limit, candidate_pool=args.candidate_pool, time_limit=args.time_limit)

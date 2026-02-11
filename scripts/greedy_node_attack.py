"""Greedy node attack: iteratively remove the author whose removal most reduces SE@5.

Run: python3 scripts/greedy_node_attack.py --min_p 2 --limit 100
Outputs: perturb_grid_out/greedy_attack_sequence_minp{min_p}.csv and greedy_attack_plot_minp{min_p}.png
"""
import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time, random, csv
import networkx as nx
from analyze_aggregated import load_combined_data
from network_asp_analysis import build_author_coauthorship_network, reachability_by_hops

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)
random.seed(3)


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


def compute_reach5(G, cats):
    r = reachability_by_hops(G, cats, max_k=5)
    se = r['se_frac_by_k'][4]
    ss = r['ss_frac_by_k'][4]
    ee = r['ee_frac_by_k'][4]
    delta = se - ((ss + ee) / 2.0)
    return {'se': se, 'ss': ss, 'ee': ee, 'delta': delta}


def run(limit=100, min_p=2, save_files=True, top_eval=50):
    """Run greedy attack for given min_p and return sequence list of dicts.

    Performs exact marginal evaluation among up to `top_eval` heuristic-ranked candidates
    each iteration (slower but selects the node that best reduces delta = SE - mean(SS,EE)).
    """
    papers = load_combined_data()
    G_auth, cats_auth, info_auth = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    print('Authors:', G_auth.number_of_nodes())
    base = compute_reach5(G_auth, cats_auth)
    baseline = base['se']
    baseline_delta = base['delta']
    print('Baseline SE@5', baseline, ' SS@5', base['ss'], ' EE@5', base['ee'], ' DELTA', base['delta'])

    removed = set()
    seq = []

    # candidate pool: nodes in G_auth
    candidates = list(G_auth.nodes())

    # heuristic ranking for candidate shortlist (degree * cross_type_degree)
    deg = dict(G_auth.degree())
    cross_deg = {}
    for n in candidates:
        mycat = cats_auth.get(n)
        cd = 0
        for nbr in G_auth[n]:
            if cats_auth.get(nbr) != mycat:
                cd += 1
        cross_deg[n] = cd
    scores = {n: deg.get(n,0) * cross_deg.get(n,0) for n in candidates}
    ranked_all = [n for n,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    print('Candidate pool size', len(ranked_all))

    for step in range(1, limit+1):
        # build shortlist of unevaluated candidates (top of heuristic ranking)
        shortlist = [n for n in ranked_all if n not in removed]
        if not shortlist:
            print('No more candidates')
            break
        shortlist = shortlist[:max(top_eval, 1)]

        # exact marginal evaluation among shortlist
        best_node = None
        best_delta_after = None
        # evaluate each candidate's marginal effect
        for cand in shortlist:
            # simulate removal of cand in addition to already removed set
            cand_removed = removed.union({cand})
            filtered = remove_authors_from_papers(papers, cand_removed)
            try:
                G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
            except Exception:
                # if rebuilding fails (graph too small), skip candidate
                continue
            m = compute_reach5(G_r, cats_r)
            delta_after = m['delta']
            # choose node that minimizes delta_after (most negative)
            if best_node is None or (delta_after < best_delta_after):
                best_node = cand
                best_delta_after = delta_after

        if best_node is None:
            print('No viable candidate found in shortlist')
            break

        # commit best_node removal
        removed.add(best_node)
        # rebuild filtered data for reporting
        filtered_commit = remove_authors_from_papers(papers, removed)
        G_commit, cats_commit, _ = build_author_coauthorship_network(filtered_commit, min_papers_per_author=min_p)
        metrics = compute_reach5(G_commit, cats_commit)
        seq.append({'step':step,'removed_node':best_node,'se_after':metrics['se'],'ss_after':metrics['ss'],'ee_after':metrics['ee'],'delta_after':metrics['delta'],'abs_drop': baseline - metrics['se']})
        baseline = metrics['se']
        baseline_delta = metrics['delta']
        print(f"step {step}: remove {best_node} -> SE@5 {metrics['se']} SS@5 {metrics['ss']} EE@5 {metrics['ee']} DELTA {metrics['delta']}")
        # early stopping if SE very low
        if metrics['se'] < 0.001:
            break

    outcsv = os.path.join(OUT, f'greedy_attack_sequence_minp{min_p}.csv')
    if save_files:
        with open(outcsv,'w',newline='',encoding='utf-8') as f:
            fieldnames = ['step','removed_node','se_after','ss_after','ee_after','delta_after','abs_drop']
            w=csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(seq)

        # plot
        try:
            import matplotlib.pyplot as plt
            steps = [s['step'] for s in seq]
            se_vals = [s['se_after']*100 for s in seq]
            delta_vals = [s['delta_after']*100 for s in seq]
            plt.figure(figsize=(6,3))
            plt.plot(steps, se_vals, marker='o', label='SE@5 %')
            plt.plot(steps, delta_vals, marker='x', label='Delta (SE - mean(SS,EE)) %')
            plt.xlabel('removed nodes (greedy)')
            plt.ylabel('percentage points')
            plt.title(f'Greedy attack collapse (min_p={min_p})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT,f'greedy_attack_plot_minp{min_p}.png'), dpi=150)
            plt.close()
            # also save delta-only plot
            try:
                plt.figure(figsize=(6,3))
                plt.plot(steps, delta_vals, marker='o', color='C1')
                plt.xlabel('removed nodes (greedy)')
                plt.ylabel('Delta %')
                plt.title(f'Greedy attack delta (min_p={min_p})')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(OUT,f'greedy_attack_delta_plot_minp{min_p}.png'), dpi=150)
                plt.close()
            except Exception:
                pass
            print('Wrote', outcsv, f'and greedy_attack_plot_minp{min_p}.png')
        except Exception as e:
            print('Plot failed',e)

    return seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_p', type=int, default=2)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--top_eval', type=int, default=50, help='number of heuristic-top candidates to exactly evaluate per iteration')
    args = parser.parse_args()
    start=time.time(); run(limit=args.limit, min_p=args.min_p, save_files=True, top_eval=args.top_eval); print('Done in', time.time()-start)

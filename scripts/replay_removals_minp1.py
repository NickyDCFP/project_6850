"""Replay a precomputed removal sequence (from min_p=2 run) but build the network with min_p=1.

Saves results to perturb_grid_out/replay_removals_minp1.csv and plots.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv, time
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


def compute_reach5(G, cats):
    r = reachability_by_hops(G, cats, max_k=5)
    se = r['se_frac_by_k'][4]
    ss = r['ss_frac_by_k'][4]
    ee = r['ee_frac_by_k'][4]
    delta = se - ((ss + ee) / 2.0)
    return {'se': se, 'ss': ss, 'ee': ee, 'delta': delta}


def run(replay_csv, limit=100, min_p=1, save_files=True):
    papers = load_combined_data()
    # build base network with min_p=1
    G_auth, cats_auth, info_auth = build_author_coauthorship_network(papers, min_papers_per_author=min_p)
    print('Authors (min_p=1):', G_auth.number_of_nodes())
    base = compute_reach5(G_auth, cats_auth)
    print('Baseline SE@5', base['se'], 'SS@5', base['ss'], 'EE@5', base['ee'], 'DELTA', base['delta'])

    # read replay removal order
    removals = []
    with open(replay_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            removals.append(row['removed_node'])

    removed = set()
    seq = []
    baseline = base['se']

    for step, node in enumerate(removals[:limit], start=1):
        removed.add(node)
        filtered = remove_authors_from_papers(papers, removed)
        try:
            G_r, cats_r, _ = build_author_coauthorship_network(filtered, min_papers_per_author=min_p)
        except Exception as e:
            print('build failed at step', step, 'after removing', node, 'error', e)
            break
        metrics = compute_reach5(G_r, cats_r)
        seq.append({'step':step,'removed_node':node,'se_after':metrics['se'],'ss_after':metrics['ss'],'ee_after':metrics['ee'],'delta_after':metrics['delta'],'abs_drop': baseline - metrics['se']})
        baseline = metrics['se']
        print(f"step {step}: remove {node} -> SE@5 {metrics['se']} SS@5 {metrics['ss']} EE@5 {metrics['ee']} DELTA {metrics['delta']}")

    outcsv = os.path.join(OUT, f'replay_removals_minp{min_p}.csv')
    if save_files:
        with open(outcsv,'w',newline='',encoding='utf-8') as f:
            fieldnames = ['step','removed_node','se_after','ss_after','ee_after','delta_after','abs_drop']
            w=csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(seq)
        try:
            import matplotlib.pyplot as plt
            steps = [s['step'] for s in seq]
            se_vals = [s['se_after']*100 for s in seq]
            delta_vals = [s['delta_after']*100 for s in seq]
            plt.figure(figsize=(6,3))
            plt.plot(steps, se_vals, marker='o', label='SE@5 %')
            plt.plot(steps, delta_vals, marker='x', label='Delta %')
            plt.xlabel('removed nodes (replay)')
            plt.ylabel('percentage points')
            plt.title(f'Replay removals on min_p={min_p}')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(OUT,f'replay_removals_minp{min_p}_plot.png'), dpi=150)
            plt.close()
            print('Wrote', outcsv, 'and plot')
        except Exception as e:
            print('Plot failed',e)
    return seq


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_csv', type=str, default=os.path.join(OUT,'greedy_attack_sequence_minp2.csv'))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--min_p', type=int, default=1)
    args = parser.parse_args()
    start=time.time(); run(args.replay_csv, limit=args.limit, min_p=args.min_p); print('Done in', time.time()-start)

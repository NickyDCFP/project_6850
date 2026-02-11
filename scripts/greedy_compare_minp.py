"""Run greedy_node_attack for multiple min_p values and compare collapse curves.

Run: python3 scripts/greedy_compare_minp.py
Outputs: perturb_grid_out/greedy_attack_sequence_minp{p}.csv and perturb_grid_out/greedy_compare_minp.png
"""
import os, time
ROOT = os.path.dirname(os.path.dirname(__file__))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.greedy_node_attack import run as greedy_run
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUT, exist_ok=True)

def main(minps=[3,4,5], limit=100):
    results = {}
    for p in minps:
        print('Running greedy for min_p=', p)
        seq = greedy_run(limit=limit, min_p=p, save_files=True)
        se_vals = [s['se_after']*100 for s in seq]
        results[p] = se_vals

    # align lengths
    maxlen = max(len(v) for v in results.values())
    for p in results:
        if len(results[p]) < maxlen:
            last = results[p][-1] if results[p] else 0
            results[p] = results[p] + [last]*(maxlen-len(results[p]))

    # plot
    plt.figure(figsize=(7,3))
    for p,v in sorted(results.items()):
        plt.plot(range(1,1+len(v)), v, marker='o', label=f'min_p={p}')
    plt.xlabel('removed nodes (greedy)')
    plt.ylabel('SE@5 %')
    plt.title('Greedy attack collapse: compare min_p')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(OUT,'greedy_compare_minp.png')
    plt.savefig(out, dpi=150)
    print('Wrote', out)

if __name__ == '__main__':
    main()

import csv
import os
from pathlib import Path
import pandas as pd

OUT_DIR = Path('perturb_grid_out')
CSV_TEMPLATE = 'greedy_attack_sequence_minp{}.csv'
MIN_PS = [3,4,5]
CHECKS = [1,5,10,50,100]

rows = []
for p in MIN_PS:
    path = OUT_DIR/CSV_TEMPLATE.format(p)
    if not path.exists():
        print('Missing', path)
        continue
    df = pd.read_csv(path)
    for c in CHECKS:
        # find row with step == c, or closest lower step
        sub = df[df['step']<=c]
        if sub.empty:
            continue
        r = sub.iloc[-1]
        rows.append({
            'min_p': p,
            'checkpoint': c,
            'step': int(r['step']),
            'removed_node': r['removed_node'],
            'se_after': float(r['se_after']),
            'ss_after': float(r['ss_after']),
            'ee_after': float(r['ee_after']),
            'delta_after': float(r['delta_after'])
        })

out_df = pd.DataFrame(rows)
out_csv = OUT_DIR/'greedy_checkpoints_summary.csv'
out_df.to_csv(out_csv, index=False)
print('Wrote', out_csv)

# print a small markdown table
for p in MIN_PS:
    sub = out_df[out_df['min_p']==p]
    if sub.empty:
        continue
    print(f"\n## min_p={p}")
    print('|checkpoint|step|removed_node|se|ss|ee|delta|')
    print('|---:|---:|---|---:|---:|---:|---:|')
    for _,r in sub.iterrows():
        print(f"|{int(r['checkpoint'])}|{int(r['step'])}|{r['removed_node']}|{r['se_after']:.4f}|{r['ss_after']:.4f}|{r['ee_after']:.4f}|{r['delta_after']:.4f}|")

# optionally create a combined png plot of delta curves
try:
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    for p in MIN_PS:
        path = OUT_DIR/CSV_TEMPLATE.format(p)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ax.plot(df['step'], df['delta_after'], label=f'min_p={p}')
    ax.set_xlabel('removals')
    ax.set_ylabel('delta (SE - mean(SS,EE))')
    ax.legend()
    out_png = OUT_DIR/'greedy_checkpoints_delta_compare.png'
    fig.savefig(out_png)
    print('Wrote', out_png)
except Exception as e:
    print('Could not make combined plot:', e)

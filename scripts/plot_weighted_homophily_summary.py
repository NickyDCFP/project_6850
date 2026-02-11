"""Plot summary of weighted homophily and weighted ASP stats across removal fractions.
Reads perturb_grid_out/weighted_homophily_filtered_minp2_summary.csv and writes two PNGs to perturb_grid_out/.
"""
import os, csv
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, 'perturb_grid_out')
SUMMARY = os.path.join(OUT, 'weighted_homophily_filtered_minp2_summary.csv')

fracs = []
labels = []
auth_hom = []
paper_hom = []
auth_se_med_w = []
paper_se_med_w = []
auth_se_med_hops = []
paper_se_med_hops = []

with open(SUMMARY, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        labels.append(row['label'])
        fracs.append(float(row['frac']))
        auth_hom.append(float(row['author_weighted_homophily']))
        paper_hom.append(float(row['paper_weighted_homophily']))
        # handle possible empty fields
        def maybe_float(x):
            try:
                return float(x)
            except Exception:
                return float('nan')
        auth_se_med_w.append(maybe_float(row['author_se_median_weighted_dist']))
        paper_se_med_w.append(maybe_float(row['paper_se_median_weighted_dist']))
        auth_se_med_hops.append(maybe_float(row['author_se_median_hops']))
        paper_se_med_hops.append(maybe_float(row['paper_se_median_hops']))

# Plot 1: homophily vs fraction
plt.figure(figsize=(6,4))
plt.plot([f*100 for f in fracs], auth_hom, marker='o', label='Author weighted homophily')
plt.plot([f*100 for f in fracs], paper_hom, marker='s', label='Paper weighted homophily')
plt.xlabel('Removed fraction (%)')
plt.ylabel('Weighted homophily (fraction)')
plt.title('Weighted homophily vs removal fraction (min_p=2)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'weighted_homophily_vs_frac.png'), dpi=150)
plt.close()

# Plot 2: median SE weighted dist and median SE hops vs fraction
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot([f*100 for f in fracs], auth_se_med_w, marker='o', label='Author SE median weighted dist')
plt.plot([f*100 for f in fracs], paper_se_med_w, marker='s', label='Paper SE median weighted dist')
plt.xlabel('Removed fraction (%)')
plt.ylabel('Median weighted SE distance')
plt.title('Median weighted SE distance')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1,2,2)
plt.plot([f*100 for f in fracs], auth_se_med_hops, marker='o', label='Author SE median hops')
plt.plot([f*100 for f in fracs], paper_se_med_hops, marker='s', label='Paper SE median hops')
plt.xlabel('Removed fraction (%)')
plt.ylabel('Median SE hops')
plt.title('Median SE hops')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'weighted_SE_stats_vs_frac.png'), dpi=150)
plt.close()

print('Wrote PNGs to', OUT)

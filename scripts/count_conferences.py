#!/usr/bin/env python3
"""
Count papers per conference for ACL and DBLP datasets.
"""

import os, json
from glob import glob
from collections import defaultdict

def load_papers():
    papers = []
    # Load from final_dataset (DBLP)
    for filepath in glob('data/final_dataset/*.jsonl'):
        source = os.path.basename(filepath).split("_")[0]
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                paper['source_file'] = source
                papers.append(paper)
    # Load from acl_anthology_papers_v2.jsonl
    with open('data/acl_anthology_papers_v2.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line.strip())
            papers.append(paper)
    return papers

def count_per_conference(papers):
    counts = defaultdict(int)
    for p in papers:
        if 'source' in p and p['source'] == 'acl_anthology':
            counts['ACL Anthology'] += 1
        elif 'source_file' in p:
            counts[p['source_file'].upper()] += 1
    return counts

def generate_latex_table(counts):
    # Sort by count descending
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    latex = r"""\begin{table}[h]
\centering
\begin{tabular}{l r}
\hline
Conference & Number of Papers \\
\hline
"""
    for conf, count in sorted_counts:
        latex += f"{conf} & {count} \\\\\n"
    latex += r"""\hline
\end{tabular}
\caption{Number of papers per conference, aggregated across years.}
\label{tab:conference_counts}
\end{table}
"""
    return latex

def main():
    papers = load_papers()
    print(f'Total papers: {len(papers)}')
    counts = count_per_conference(papers)
    latex = generate_latex_table(counts)
    print(latex)
    # Save to file
    with open('conference_counts.tex', 'w') as f:
        f.write(latex)
    print('Saved to conference_counts.tex')

if __name__ == '__main__':
    main()
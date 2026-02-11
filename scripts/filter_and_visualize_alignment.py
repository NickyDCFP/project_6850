#!/usr/bin/env python3
"""
Filter papers from final_dataset that mention "alignment" at least twice in title + abstract.
Save to filtered_alignment folder, then visualize networks.
"""

import os, json
from glob import glob

def load_papers_from_final_dataset():
    papers = []
    # Load from final_dataset
    for filepath in glob('data/final_dataset/*.jsonl'):
        source = os.path.basename(filepath).split("_")[0]
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                paper['source_file'] = source
                if source in ['aies', 'fat', 'satml']:
                    if paper.get('ethics_score', 0) == 0 and paper.get('safety_score', 0) == 0:
                        if source == 'satml':
                            paper['safety_score'] = 1
                        else:
                            paper['ethics_score'] = 1
                paper['category'] = 'both' if paper.get('safety_score', 0) > 0 and paper.get('ethics_score', 0) > 0 else \
                                    'safety' if paper.get('safety_score', 0) > 0 else \
                                    'ethics' if paper.get('ethics_score', 0) > 0 else 'none'
                paper['year'] = int(paper.get('year', 0))
                papers.append(paper)
    # Load from acl_anthology_papers_v2.jsonl
    with open('data/acl_anthology_papers_v2.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line.strip())
            # Already has scores and category
            papers.append(paper)
    return papers

def filter_alignment_papers(papers):
    filtered = []
    for p in papers:
        text = (p.get('title', '') + ' ' + p.get('abstract', '')).lower()
        count = text.count('alignment')
        if count >= 2:
            filtered.append(p)
    return filtered

def save_filtered_papers(papers, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'filtered_alignment_papers.jsonl')
    with open(out_file, 'w', encoding='utf-8') as f:
        for p in papers:
            f.write(json.dumps(p) + '\n')
    print(f'Saved {len(papers)} filtered papers to {out_file}')

def visualize_filtered_networks(filtered_papers, out_dir):
    # Import plot functions
    from analyze_aggregated import plot_colored_paper_network, plot_author_network_colored
    # Plot
    plot_colored_paper_network(filtered_papers, os.path.join(out_dir, 'alignment_paper_network_colored.png'), top_n=100)
    plot_author_network_colored(filtered_papers, os.path.join(out_dir, 'alignment_author_network_colored.png'), top_n=100)
    print('Visualizations saved to', out_dir)

def main():
    papers = load_papers_from_final_dataset()
    print(f'Loaded {len(papers)} papers from final_dataset.')
    
    filtered = filter_alignment_papers(papers)
    print(f'Filtered to {len(filtered)} papers with "alignment" >=2 times.')
    
    out_dir = 'filtered_alignment'
    save_filtered_papers(filtered, out_dir)
    
    visualize_filtered_networks(filtered, out_dir)

if __name__ == '__main__':
    main()
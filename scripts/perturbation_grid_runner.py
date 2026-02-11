"""Automated perturbation + metrics grid runner.

This script enumerates a combinatorial grid of graph filters and perturbations
and computes a set of interpret-able metrics per experiment. Results are
appended to `perturb_grid_out/grid_results.csv` and basic plots are saved.

Run from repo root: python3 scripts/perturbation_grid_runner.py
"""
import os
import sys
import csv
import time
import random
from itertools import product
from collections import defaultdict
import networkx as nx
import math

# ensure repo root is on sys.path so imports like `analyze_aggregated` resolve
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTDIR = os.path.join(ROOT, 'perturb_grid_out')
os.makedirs(OUTDIR, exist_ok=True)

try:
    from analyze_aggregated import load_combined_data
    from network_asp_analysis import (
        build_author_coauthorship_network,
        build_paper_coauthorship_network,
        reachability_by_hops,
    )
except Exception:
    raise

random.seed(0)

def compute_conductance(G, categories, label):
    nodes = {n for n,c in categories.items() if c == label}
    if not nodes:
        return None
    cut = nx.cut_size(G, nodes)
    vol = sum(dict(G.degree(nodes)).values())
    return cut / vol if vol > 0 else None

def bridge_fraction_by_authors(G_author, author_cats, top_authors, sample_target=2000, max_pairs=200000):
    S = [n for n,c in author_cats.items() if c == 'pure_safety']
    E = [n for n,c in author_cats.items() if c == 'pure_ethics']
    if not S or not E:
        return {'valid':0,'through_top':0,'frac':None}
    valid = 0; thru = 0; pairs_checked = 0
    pairs = []
    total_possible = len(S)*len(E)
    if total_possible <= max_pairs:
        for s in S:
            for e in E:
                pairs.append((s,e))
    else:
        seen=set(); attempts=0
        while len(pairs) < max_pairs and attempts < max_pairs*20:
            a=random.choice(S); b=random.choice(E)
            if (a,b) in seen: attempts+=1; continue
            seen.add((a,b)); pairs.append((a,b))
    for u,v in pairs:
        pairs_checked += 1
        try:
            path = nx.shortest_path(G_author, u, v)
        except nx.NetworkXNoPath:
            continue
        valid += 1
        internal = path[1:-1]
        if any(a in top_authors for a in internal):
            thru += 1
        if valid >= sample_target:
            break
    return {'valid': valid, 'through_top': thru, 'frac': thru / valid if valid else None, 'pairs_checked': pairs_checked}

def paper_path_mediated_by_authors(path, paper_info, top_authors):
    for i in range(len(path)-1):
        a1 = set(paper_info.get(path[i], {}).get('authors', []))
        a2 = set(paper_info.get(path[i+1], {}).get('authors', []))
        if a1.intersection(a2).intersection(top_authors):
            return True
    return False

def bridge_fraction_by_papers(G_paper, paper_cats, paper_info, top_authors, sample_target=2000, max_pairs=200000):
    S = [n for n,c in paper_cats.items() if c == 'pure_safety']
    E = [n for n,c in paper_cats.items() if c == 'pure_ethics']
    if not S or not E:
        return {'valid':0,'through_top':0,'frac':None}
    valid = 0; thru = 0; pairs_checked = 0
    pairs = []
    total_possible = len(S)*len(E)
    if total_possible <= max_pairs:
        for s in S:
            for e in E:
                pairs.append((s,e))
    else:
        seen=set(); attempts=0
        while len(pairs) < max_pairs and attempts < max_pairs*20:
            a=random.choice(S); b=random.choice(E)
            if (a,b) in seen: attempts+=1; continue
            seen.add((a,b)); pairs.append((a,b))
    for u,v in pairs:
        pairs_checked += 1
        try:
            path = nx.shortest_path(G_paper, u, v)
        except nx.NetworkXNoPath:
            continue
        valid += 1
        if paper_path_mediated_by_authors(path, paper_info, top_authors):
            thru += 1
        if valid >= sample_target:
            break
    return {'valid': valid, 'through_top': thru, 'frac': thru / valid if valid else None, 'pairs_checked': pairs_checked}

def run_single_experiment(cfg, papers):
    res = {'cfg': cfg}
    entity = cfg['entity']
    minp = cfg['min_p']
    # build fresh graphs from papers (or full dataset)
    if entity == 'author':
        G, cats, info = build_author_coauthorship_network(papers, min_papers_per_author=minp)
    else:
        G, cats, info = build_paper_coauthorship_network(papers, min_papers_per_author=minp)

    # optionally remove mixed
    if cfg.get('remove_mixed'):
        mixed = [n for n,c in cats.items() if c=='mixed']
        G.remove_nodes_from(mixed)
        for m in mixed: cats.pop(m, None)

    # optionally apply k-core
    if cfg.get('kcore', 0) > 0:
        k = cfg['kcore']
        try:
            G = nx.k_core(G, k=k)
        except Exception:
            # if k-core fails (too big), skip
            pass
        cats = {n: cats[n] for n in G.nodes() if n in cats}

    # baseline reachability by hops
    kmax = cfg.get('kmax', 7)
    reach = reachability_by_hops(G, cats, max_k=kmax)
    # store k=1..kmax se/ss/ee fractions
    for k in range(1, kmax+1):
        res[f'se_frac_k{k}'] = float(reach['se_frac_by_k'][k-1])
        res[f'ss_frac_k{k}'] = float(reach['ss_frac_by_k'][k-1])
        res[f'ee_frac_k{k}'] = float(reach['ee_frac_by_k'][k-1])
    res['nodes'] = G.number_of_nodes(); res['edges'] = G.number_of_edges()

    # determine top authors set (if requested)
    topK = cfg.get('topK', 0)
    top_authors = set()
    if topK > 0:
        # if author-entity, use degree in the graph; if paper-entity, infer from paper info
        if entity == 'author':
            degs = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            top_authors = set([n for n,_ in degs[:topK]])
        else:
            # infer top authors by counting occurrences in paper info
            count = defaultdict(int)
            for pid, pinfo in info.items():
                for a in pinfo.get('authors', []):
                    name = a.get('name') if isinstance(a, dict) else a
                    if name: count[name] += 1
            top_authors = set(sorted(count, key=lambda x: -count[x])[:topK])

    # bridge concentration
    if entity == 'author':
        bf = bridge_fraction_by_authors(G, cats, top_authors, sample_target=2000)
    else:
        bf = bridge_fraction_by_papers(G, cats, info, top_authors, sample_target=2000)
    res['bridge_valid'] = bf['valid']; res['bridge_through_top'] = bf['through_top']; res['bridge_frac'] = float(bf['frac']) if bf['frac'] is not None else None

    # conductance
    res['cond_s'] = compute_conductance(G, cats, 'pure_safety')
    res['cond_e'] = compute_conductance(G, cats, 'pure_ethics')

    return res

def run_grid(outdir=OUTDIR):
    papers = load_combined_data()
    entities = ['author','paper']
    min_ps = [0,2]
    remove_mixed_opts = [False, True]
    topKs = [0,100,500]
    kcores = [0,2]
    kmax = 7
    grid = list(product(entities, min_ps, remove_mixed_opts, topKs, kcores))
    header = ['entity','min_p','remove_mixed','topK','kcore','nodes','edges'] + [f'{p}k{k}' for k in range(1,kmax+1) for p in ('se_frac_','ss_frac_','ee_frac_')] + ['bridge_valid','bridge_through_top','bridge_frac','cond_s','cond_e']
    outcsv = os.path.join(outdir,'grid_results_v2.csv')
    write_header = not os.path.exists(outcsv)
    with open(outcsv,'a',newline='',encoding='utf-8') as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(header)
        total = len(grid)
        for i,(entity,minp,remove_mixed,topK,kcore) in enumerate(grid,1):
            cfg = {'entity':entity,'min_p':minp,'remove_mixed':remove_mixed,'topK':topK,'kcore':kcore,'kmax':kmax}
            print(f"[{i}/{total}] {cfg}")
            try:
                r = run_single_experiment(cfg, papers)
            except Exception as e:
                print('Experiment failed:',e)
                continue
            row = [entity,minp,remove_mixed,topK,kcore,r.get('nodes'),r.get('edges')] + [r.get(f'se_frac_k{k}') for k in range(1,kmax+1)] + [r.get('bridge_valid'),r.get('bridge_through_top'),r.get('bridge_frac'),r.get('cond_s'),r.get('cond_e')]
            w.writerow(row)

if __name__ == '__main__':
    start=time.time()
    run_grid()
    print('Done in', time.time()-start)

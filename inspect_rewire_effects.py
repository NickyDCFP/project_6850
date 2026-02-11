#!/usr/bin/env python3
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import networkx as nx

def load_jsonl(path: str) -> List[Dict]:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def normalize_author(name: str) -> str:
    if not name:
        return name
    s = name.strip()
    parts = s.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return ' '.join(parts)

def build_paper_graph(papers, min_papers_per_author=None):
    author_counts = defaultdict(int)
    for p in papers:
        for a in p.get('authors', []):
            if a:
                author_counts[normalize_author(a)] += 1
    if min_papers_per_author is None:
        qualified = set(author_counts.keys())
    else:
        qualified = {a for a,c in author_counts.items() if c>=min_papers_per_author}

    sel = [p for p in papers if (p.get('safety_score',0) or 0)>0 or (p.get('ethics_score',0) or 0)>0]
    filtered = []
    for p in sel:
        authors = [normalize_author(a) for a in p.get('authors',[]) if a]
        q = [a for a in authors if a in qualified]
        if q:
            p2 = p.copy(); p2['authors']=q; filtered.append(p2)

    G = nx.Graph(); cats={}
    for i,p in enumerate(filtered):
        s=float(p.get('safety_score',0) or 0); e=float(p.get('ethics_score',0) or 0)
        if s>0 and e==0: cat='pure_safety'
        elif e>0 and s==0: cat='pure_ethics'
        elif s>0 and e>0: cat='mixed'
        else: cat='neither'
        G.add_node(i); cats[i]=cat
    author_to_papers=defaultdict(list)
    for i,p in enumerate(filtered):
        for a in p.get('authors',[]):
            author_to_papers[a].append(i)
    for plist in author_to_papers.values():
        if len(plist)<2: continue
        for i in range(len(plist)):
            for j in range(i+1,len(plist)):
                u,v=plist[i],plist[j]
                if G.has_edge(u,v): G[u][v]['weight']+=1
                else: G.add_edge(u,v,weight=1)
    return G,cats

def edge_category(u, v, cats):
    cu=cats.get(u,'neither'); cv=cats.get(v,'neither')
    if cu=='pure_safety' and cv=='pure_safety': return 'SS'
    if cu=='pure_ethics' and cv=='pure_ethics': return 'EE'
    if (cu=='pure_safety' and cv=='pure_ethics') or (cu=='pure_ethics' and cv=='pure_safety'): return 'SE'
    return 'OTHER'

def constrained_rewire_increase_within(G, cats, target_swaps_fraction=0.05, max_attempts_per_swap=50, seed=42):
    random.seed(seed)
    H=G.copy()
    edge_list=list(H.edges())
    m=len(edge_list)
    target_swaps=max(1,int(m*target_swaps_fraction))
    successful=0; attempts=0; max_attempts=target_swaps*max_attempts_per_swap
    while successful<target_swaps and attempts<max_attempts:
        attempts+=1
        e1=random.choice(edge_list); e2=random.choice(edge_list)
        if e1==e2: continue
        a,b=e1; c,d=e2
        if len({a,b,c,d})<4: continue
        propose=[(a,c,b,d),(a,d,b,c)]
        for u1,v1,u2,v2 in propose:
            if H.has_edge(u1,v1) or H.has_edge(u2,v2): continue
            if u1==v1 or u2==v2: continue
            cat1=edge_category(u1,v1,cats); cat2=edge_category(u2,v2,cats)
            if cat1=='SE' or cat2=='SE': continue
            cur1=edge_category(a,b,cats); cur2=edge_category(c,d,cats)
            before=(1 if cur1 in ('SS','EE') else 0)+(1 if cur2 in ('SS','EE') else 0)
            after=(1 if cat1 in ('SS','EE') else 0)+(1 if cat2 in ('SS','EE') else 0)
            se_before=(1 if cur1=='SE' else 0)+(1 if cur2=='SE' else 0)
            se_after=(1 if cat1=='SE' else 0)+(1 if cat2=='SE' else 0)
            if (after>before and se_after<=se_before) or (after==before and se_after<se_before):
                H.remove_edge(a,b); H.remove_edge(c,d); H.add_edge(u1,v1); H.add_edge(u2,v2)
                edge_list.remove((a,b)); edge_list.remove((c,d))
                edge_list.append((u1,v1)); edge_list.append((u2,v2))
                successful+=1
                break
    return H

def analyze(G,cats):
    counts={'SS':0,'EE':0,'SE':0,'OTHER':0}
    for u,v in G.edges():
        cat=edge_category(u,v,cats)
        counts[cat]+=1
    comps=list(nx.connected_components(G))
    comp_sizes=sorted([len(c) for c in comps],reverse=True)
    giant=comp_sizes[0] if comp_sizes else 0
    return counts, len(comps), giant

def main():
    try:
        from analyze_aggregated import load_combined_data
        papers=load_combined_data()
    except Exception:
        papers=load_jsonl('sample.jsonl')
    for p in papers:
        p['authors']=[normalize_author(a) for a in p.get('authors',[]) if a]
    thresholds=[None,2,3]
    for thr in thresholds:
        label='no_minp' if thr is None else f'minp_{thr}'
        G,cats=build_paper_graph(papers,thr)
        before_counts, before_ncomp, before_giant = analyze(G,cats)
        H=constrained_rewire_increase_within(G,cats,seed=42)
        after_counts, after_ncomp, after_giant = analyze(H,cats)
        print(f"\n=== {label} ===")
        print(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        print("before edges:", before_counts)
        print("before components:", before_ncomp, "giant:", before_giant)
        print("after edges:", after_counts)
        print("after components:", after_ncomp, "giant:", after_giant)

if __name__=='__main__':
    main()

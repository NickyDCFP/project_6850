#!/usr/bin/env python3
"""Test adding a few targeted S-E bridge edges between high-degree safety and ethics papers and measure reachability effect.
"""
import json
import os
from collections import defaultdict
from typing import List, Tuple

import networkx as nx

def load_jsonl(path: str):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
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
            p2=p.copy(); p2['authors']=q; filtered.append(p2)

    G=nx.Graph(); cats={}
    for i,p in enumerate(filtered):
        s=float(p.get('safety_score',0) or 0); e=float(p.get('ethics_score',0) or 0)
        if s>0 and e==0: cat='pure_safety'
        elif e>0 and s==0: cat='pure_ethics'
        elif s>0 and e>0: cat='mixed'
        else: cat='neither'
        G.add_node(i); cats[i]=cat
    author_to_papers=defaultdict(list)
    for i,p in enumerate(filtered):
        for a in p.get('authors',[]): author_to_papers[a].append(i)
    for plist in author_to_papers.values():
        if len(plist)<2: continue
        for i in range(len(plist)):
            for j in range(i+1,len(plist)):
                u,v=plist[i],plist[j]
                if G.has_edge(u,v): G[u][v]['weight']+=1
                else: G.add_edge(u,v,weight=1)
    return G,cats

def reachability_by_hops(G,cats,max_k=7):
    S=[n for n,c in cats.items() if c=='pure_safety' and n in G]
    E=[n for n,c in cats.items() if c=='pure_ethics' and n in G]
    Ns=len(S); Ne=len(E)
    total_SE=Ns*Ne; total_SS=Ns*(Ns-1)//2; total_EE=Ne*(Ne-1)//2
    S_index={n:i for i,n in enumerate(sorted(S))}; E_index={n:i for i,n in enumerate(sorted(E))}
    se_counts=[0]*(max_k+1); ss_counts=[0]*(max_k+1); ee_counts=[0]*(max_k+1)
    for s in S:
        lengths=nx.single_source_shortest_path_length(G,s,cutoff=max_k)
        for t,d in lengths.items():
            if d==0: continue
            if d<=max_k:
                if t in E_index: se_counts[d]+=1
                if t in S_index and S_index[t]>S_index[s]: ss_counts[d]+=1
    for e in E:
        lengths=nx.single_source_shortest_path_length(G,e,cutoff=max_k)
        for t,d in lengths.items():
            if d==0: continue
            if d<=max_k and t in E_index and E_index[t]>E_index[e]: ee_counts[d]+=1
    se_cum=[]; ss_cum=[]; ee_cum=[]; sacc=ssacc=eeacc=0
    for k in range(1,max_k+1):
        sacc+=se_counts[k]; ssacc+=ss_counts[k]; eeacc+=ee_counts[k]
        se_cum.append(sacc); ss_cum.append(ssacc); ee_cum.append(eeacc)
    se_frac=[c/total_SE if total_SE else 0 for c in se_cum]
    ss_frac=[c/total_SS if total_SS else 0 for c in ss_cum]
    ee_frac=[c/total_EE if total_EE else 0 for c in ee_cum]
    return {'se_frac_by_k':se_frac,'ss_frac_by_k':ss_frac,'ee_frac_by_k':ee_frac}

def add_bridges_and_eval(G,cats,bridges:int):
    # choose top-degree S nodes and top-degree E nodes
    degs=sorted(G.degree(),key=lambda x:x[1],reverse=True)
    S_nodes=[n for n,c in cats.items() if c=='pure_safety' and n in G]
    E_nodes=[n for n,c in cats.items() if c=='pure_ethics' and n in G]
    # rank S by degree
    S_sorted=sorted(S_nodes,key=lambda n:G.degree(n),reverse=True)
    E_sorted=sorted(E_nodes,key=lambda n:G.degree(n),reverse=True)
    G2=G.copy()
    added=[]
    for i in range(min(bridges,len(S_sorted),len(E_sorted))):
        u=S_sorted[i]; v=E_sorted[i]
        if not G2.has_edge(u,v):
            G2.add_edge(u,v,weight=1); added.append((u,v))
    return G2, added

def print_comparison(before, after):
    ks=list(range(1,8))
    print('k | SE_before | SE_after | delta(pp)')
    for i,k in enumerate(ks):
        sb=before['se_frac_by_k'][i]*100; sa=after['se_frac_by_k'][i]*100
        print(f'{k} | {sb:.4f} | {sa:.4f} | {sa-sb:+.4f}')

def main():
    try:
        from analyze_aggregated import load_combined_data
        papers=load_combined_data()
    except Exception:
        papers=load_jsonl('sample.jsonl')
    for p in papers:
        p['authors']=[normalize_author(a) for a in p.get('authors',[]) if a]

    for thr in [None,2,3]:
        label='no_minp' if thr is None else f'minp_{thr}'
        print(f'\n=== {label} ===')
        G,cats=build_paper_graph(papers,thr)
        baseline=reachability_by_hops(G,cats,max_k=7)
        for nbridges in [1,3,5]:
            G2,added=add_bridges_and_eval(G,cats,nbridges)
            after=reachability_by_hops(G2,cats,max_k=7)
            print(f'bridges={nbridges}, added={len(added)}')
            print_comparison(baseline,after)

if __name__=='__main__':
    main()

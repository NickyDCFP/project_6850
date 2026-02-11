#!/usr/bin/env python3
"""
Fast citation enhancement for top authors only.
Extract papers from top collaborative authors and enhance those with citation data.
"""

import json
import sys
sys.path.append('src')
from enhance_with_citations import CitationEnhancer

def get_top_authors_papers():
    """Get papers from the current network's top authors"""
    print("📊 Loading current network data...")
    
    if not os.path.exists('web/data.json'):
        print("❌ Network data not found. Run export_author_network_web.py first")
        return []
    
    with open('web/data.json', 'r') as f:
        network_data = json.load(f)
    
    # Get top authors by degree
    top_authors = sorted(network_data['nodes'], key=lambda x: x['degree'], reverse=True)[:100]
    print(f"🎯 Focusing on top 100 authors by collaboration degree")
    
    # Collect all papers from these authors
    all_papers = []
    author_papers = {}
    
    for author in top_authors:
        author_name = author['id']
        papers = author.get('papers', [])
        author_papers[author_name] = len(papers)
        
        for paper in papers:
            # Add author info for tracking
            paper_copy = paper.copy()
            paper_copy['top_author'] = author_name
            all_papers.append(paper_copy)
    
    print(f"📚 Collected {len(all_papers)} papers from top authors")
    
    # Show author distribution
    print(f"📈 Papers per top author (sample):")
    for i, (author, count) in enumerate(sorted(author_papers.items(), key=lambda x: x[1], reverse=True)[:10]):
        print(f"   {i+1}. {author}: {count} papers")
    
    return all_papers

def enhance_top_papers():
    """Enhance papers from top authors with citation data"""
    import os
    
    papers = get_top_authors_papers()
    if not papers:
        return
    
    print(f"\n🔗 Enhancing {len(papers)} papers with citation data...")
    
    enhancer = CitationEnhancer(delay=0.5)  # Faster for smaller dataset
    enhanced_papers = []
    
    for i, paper in enumerate(papers):
        if i % 20 == 0:
            print(f"   Progress: {i}/{len(papers)} ({i/len(papers)*100:.1f}%)")
        
        enhanced_paper = paper.copy()
        
        # Try Semantic Scholar
        title = paper.get('title', '')
        if title:
            ss_data = enhancer.get_semantic_scholar_data(title)
            if ss_data:
                enhanced_paper.update(ss_data)
                print(f"   ✅ {title[:50]}... -> {ss_data.get('citation_count', 0)} citations")
        
        enhanced_papers.append(enhanced_paper)
    
    # Save enhanced papers
    output_file = 'data/top_authors_papers_with_citations.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in enhanced_papers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # Analysis
    with_citations = [p for p in enhanced_papers if p.get('citation_count', 0) > 0]
    
    print(f"\n📊 CITATION ANALYSIS:")
    print(f"   Papers processed: {len(enhanced_papers)}")
    print(f"   With citation data: {len(with_citations)} ({len(with_citations)/len(enhanced_papers)*100:.1f}%)")
    
    if with_citations:
        citation_counts = [p['citation_count'] for p in with_citations]
        citation_counts.sort(reverse=True)
        
        print(f"   Citation statistics:")
        print(f"      Range: {min(citation_counts)} - {max(citation_counts)}")
        print(f"      Average: {sum(citation_counts)/len(citation_counts):.1f}")
        print(f"      Median: {citation_counts[len(citation_counts)//2]}")
        
        print(f"\n🏆 Top cited papers:")
        top_cited = sorted(with_citations, key=lambda x: x.get('citation_count', 0), reverse=True)[:10]
        for i, paper in enumerate(top_cited):
            author = paper.get('top_author', 'Unknown')
            title = paper.get('title', 'No title')[:60]
            citations = paper.get('citation_count', 0)
            print(f"      {i+1}. {citations:3d} cites | {author} | {title}...")
    
    print(f"\n💾 Saved to: {output_file}")
    return enhanced_papers

if __name__ == "__main__":
    import os
    enhanced = enhance_top_papers()
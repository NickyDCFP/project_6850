#!/usr/bin/env python3
"""
Enhanced citation-aware scraper for ACL Anthology and DBLP conferences.
Focuses on collecting citation data (in-degree) for network analysis.
"""

import json
import time
import requests
from typing import List, Dict, Optional
import os
from datetime import datetime

class CitationEnhancer:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Tool for Academic Analysis)'
        })
    
    def get_semantic_scholar_data(self, title: str, authors: List[str] = None) -> Optional[Dict]:
        """
        Get citation data from Semantic Scholar API
        """
        try:
            # Try searching by title first
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': title,
                'fields': 'paperId,title,authors,citationCount,referenceCount,year,venue,doi'
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    # Return the first match (usually most relevant)
                    paper = data['data'][0]
                    return {
                        'semantic_scholar_id': paper.get('paperId'),
                        'citation_count': paper.get('citationCount', 0),
                        'reference_count': paper.get('referenceCount', 0),
                        'ss_title': paper.get('title'),
                        'ss_year': paper.get('year'),
                        'ss_venue': paper.get('venue')
                    }
            
            time.sleep(self.delay)  # Rate limiting
            
        except Exception as e:
            print(f"    Error getting Semantic Scholar data for '{title[:50]}...': {e}")
        
        return None
    
    def get_crossref_data(self, doi: str) -> Optional[Dict]:
        """
        Get citation data from Crossref API using DOI
        """
        if not doi:
            return None
            
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                work = data.get('message', {})
                return {
                    'crossref_citation_count': work.get('is-referenced-by-count', 0),
                    'crossref_reference_count': len(work.get('reference', [])),
                    'crossref_title': work.get('title', [None])[0],
                    'crossref_doi': work.get('DOI')
                }
                
            time.sleep(self.delay)
            
        except Exception as e:
            print(f"    Error getting Crossref data for DOI {doi}: {e}")
        
        return None

def enhance_papers_with_citations(input_file: str, output_file: str, source_name: str):
    """
    Read papers from input file and enhance them with citation data
    """
    print(f"\n🔗 Enhancing {source_name} papers with citation data...")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"   ❌ Input file not found: {input_file}")
        return
    
    enhancer = CitationEnhancer(delay=1.0)  # 1 second delay to be respectful
    enhanced_papers = []
    
    # Load existing papers
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = [json.loads(line) for line in f if line.strip()]
    
    print(f"   📚 Processing {len(papers)} papers...")
    
    for i, paper in enumerate(papers):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(papers)} ({i/len(papers)*100:.1f}%)")
        
        enhanced_paper = paper.copy()
        
        # Try Semantic Scholar first (more comprehensive)
        title = paper.get('title', '')
        authors = paper.get('authors', [])
        
        if title:
            ss_data = enhancer.get_semantic_scholar_data(title, authors)
            if ss_data:
                enhanced_paper.update(ss_data)
                print(f"   ✅ {title[:60]}... -> {ss_data.get('citation_count', 0)} citations")
        
        # Try Crossref if we have a DOI and didn't get Semantic Scholar data
        if not enhanced_paper.get('citation_count') and paper.get('doi'):
            crossref_data = enhancer.get_crossref_data(paper['doi'])
            if crossref_data:
                enhanced_paper.update(crossref_data)
        
        # Add timestamp
        enhanced_paper['citation_enhanced_at'] = datetime.now().isoformat()
        enhanced_papers.append(enhanced_paper)
        
        # Save progress every 100 papers
        if (i + 1) % 100 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                for p in enhanced_papers:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in enhanced_papers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # Summary
    with_citations = sum(1 for p in enhanced_papers if p.get('citation_count', 0) > 0)
    print(f"   📊 Summary:")
    print(f"      Papers processed: {len(enhanced_papers)}")
    print(f"      With citation data: {with_citations} ({with_citations/len(enhanced_papers)*100:.1f}%)")
    
    if with_citations > 0:
        citation_counts = [p.get('citation_count', 0) for p in enhanced_papers if p.get('citation_count', 0) > 0]
        print(f"      Citation range: {min(citation_counts)} - {max(citation_counts)}")
        print(f"      Average citations: {sum(citation_counts)/len(citation_counts):.1f}")

def main():
    print("🎯 CITATION-ENHANCED SCRAPER")
    print("Focusing on ACL Anthology and DBLP conferences with citation data")
    print("=" * 60)
    
    # Enhance ACL papers
    enhance_papers_with_citations(
        input_file='data/acl_anthology_papers_v2.jsonl',
        output_file='data/acl_anthology_papers_with_citations.jsonl',
        source_name='ACL Anthology'
    )
    
    # Enhance DBLP papers
    enhance_papers_with_citations(
        input_file='data/dblp_conferences_papers.jsonl',
        output_file='data/dblp_conferences_papers_with_citations.jsonl',
        source_name='DBLP Conferences'
    )
    
    print(f"\n✅ Citation enhancement complete!")
    print(f"📁 Enhanced files:")
    print(f"   - data/acl_anthology_papers_with_citations.jsonl")
    print(f"   - data/dblp_conferences_papers_with_citations.jsonl")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Update export_author_network_web.py to use citation-enhanced files")
    print(f"   2. Modify filtering to use citation_count instead of relevance scores")
    print(f"   3. Add citation metrics to network visualization")

if __name__ == "__main__":
    main()
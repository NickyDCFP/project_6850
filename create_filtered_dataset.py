#!/usr/bin/env python3
"""
Create filtered combined dataset using Gemini-powered filtering system.
Applies strict LLM-based filtering to identify genuine AI ethics/safety research.
"""

import json
import os
from typing import Dict, List, Tuple
from collections import Counter
import argparse
import time

from tqdm import tqdm

# Import our Gemini filter
from llm_paper_filter import GeminiEthicsSafetyPaperFilter, FilterResult

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    papers = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    return papers

def save_jsonl(papers: List[Dict], filepath: str):
    """Save papers to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')

def filter_and_combine_datasets(batch_size: int = 100, delay: float = 2.0):
    """Main function to filter and combine datasets using Gemini."""
    
    print("🔍 Loading datasets...")
    
    # Load both datasets
    acl_papers = load_jsonl('data/acl_anthology_papers_v2.jsonl')
    dblp_papers = load_jsonl('data/dblp_conferences_papers.jsonl')
    
    print(f"Loaded {len(acl_papers)} ACL papers and {len(dblp_papers)} DBLP papers")
    
    # Get ethics/safety papers from each dataset
    acl_ethics = acl_papers
    dblp_ethics = dblp_papers
    
    print(f"Found {len(acl_ethics)} ACL ethics/safety papers")
    print(f"Found {len(dblp_ethics)} DBLP ethics/safety papers")
    
    # Combine all ethics/safety papers
    all_ethics_papers = acl_ethics + dblp_ethics
    print(f"Total ethics/safety papers: {len(all_ethics_papers)}")
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key before running filtering")
        return
    
    # Initialize Gemini filter
    print(f"\n🤖 Applying Gemini-powered filtering...")
    print(f"Processing {batch_size} papers per batch with {delay}s delay...")
    filter_system = GeminiEthicsSafetyPaperFilter()
    
    # Filter papers in batches
    filtered_papers = []
    filter_stats = Counter()
    category_stats = Counter()
    
    for batch_start in tqdm(range(0, len(all_ethics_papers), batch_size)):
        batch_end = min(batch_start + batch_size, len(all_ethics_papers))
        batch_papers = all_ethics_papers[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {len(all_ethics_papers)})...")
        
        # Filter batch
        batch_results = filter_system.filter_batch(batch_papers)
        
        # Process results
        for paper, result in batch_results:
            filter_stats[result.include] += 1
            category_stats[result.category] += 1
            
            if result.include:
                # Add filter metadata
                paper_with_filter = paper.copy()
                paper_with_filter['filter_category'] = result.category
                paper_with_filter['filter_confidence'] = result.confidence
                paper_with_filter['filter_reasoning'] = result.reasoning
                paper_with_filter['filter_method'] = 'gemini'
                filtered_papers.append(paper_with_filter)
        
        # Progress update
        current_included = filter_stats[True]
        current_processed = sum(filter_stats.values())
        current_rate = current_included / current_processed * 100 if current_processed > 0 else 0
        print(f"  Batch complete: {len([r for _, r in batch_results if r.include])}/{len(batch_results)} included")
        print(f"  Running total: {current_included}/{current_processed} ({current_rate:.1f}%) included")
    
    print(f"\n📊 Final Filtering Results:")
    print(f"  Total papers processed: {len(all_ethics_papers)}")
    print(f"  Papers included: {filter_stats[True]} ({filter_stats[True]/len(all_ethics_papers)*100:.1f}%)")
    print(f"  Papers excluded: {filter_stats[False]} ({filter_stats[False]/len(all_ethics_papers)*100:.1f}%)")
    
    print(f"\n📂 Category Breakdown:")
    for category, count in category_stats.most_common():
        if category != 'not_ai_ethics':  # Skip the main exclusion category
            print(f"  {category}: {count}")
    
    # Source breakdown for included papers
    source_breakdown = Counter(p['source'] for p in filtered_papers)
    print(f"\n📊 Source Breakdown (Included Papers):")
    for source, count in source_breakdown.items():
        print(f"  {source.upper()}: {count}")
    
    # Save filtered dataset
    output_path = 'data/filtered_ethics_safety_papers.jsonl'
    save_jsonl(filtered_papers, output_path)
    print(f"\n💾 Saved {len(filtered_papers)} filtered papers to {output_path}")
    
    # Save filtering metadata
    metadata = {
        'filter_type': 'gemini',
        'filter_model': 'gemini-1.5-flash',
        'total_input_papers': len(all_ethics_papers),
        'acl_input_papers': len(acl_ethics),
        'dblp_input_papers': len(dblp_ethics),
        'filtered_papers': len(filtered_papers),
        'inclusion_rate': filter_stats[True] / len(all_ethics_papers),
        'filter_stats': dict(filter_stats),
        'category_stats': dict(category_stats),
        'source_breakdown': dict(source_breakdown),
        'batch_size': batch_size,
        'delay': delay,
        'created_at': '2025-09-28'
    }
    
    with open('data/filter_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved filtering metadata to data/filter_metadata.json")
    
    # Show some examples of included papers
    print(f"\n✅ Example Included Papers:")
    for i, paper in enumerate(filtered_papers[:8]):
        print(f"  {i+1}. [{paper['filter_category']}] {paper.get('title', 'No title')[:80]}...")
        print(f"     Source: {paper['source'].upper()} | Confidence: {paper['filter_confidence']:.2f}")
        print(f"     Reasoning: {paper['filter_reasoning']}")
    
    return filtered_papers, metadata

def analyze_filtered_dataset():
    """Analyze the filtered dataset."""
    
    print("📊 Analyzing Gemini-filtered dataset...")
    
    try:
        # Load filtered papers
        filtered_papers = load_jsonl('data/filtered_ethics_safety_papers.jsonl')
        
        # Load metadata
        with open('data/filter_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  Total filtered papers: {len(filtered_papers)}")
        print(f"  Inclusion rate: {metadata['inclusion_rate']*100:.1f}%")
        print(f"  Filter method: {metadata.get('filter_type', 'unknown')}")
        
        # Analyze by category
        categories = Counter(p['filter_category'] for p in filtered_papers)
        print(f"\n📂 Categories in Filtered Dataset:")
        for category, count in categories.most_common():
            percentage = count / len(filtered_papers) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Analyze by source
        sources = Counter(p['source'] for p in filtered_papers)
        print(f"\n📊 Sources in Filtered Dataset:")
        for source, count in sources.items():
            percentage = count / len(filtered_papers) * 100
            print(f"  {source.upper()}: {count} ({percentage:.1f}%)")
        
        # Analyze by year
        years = Counter(p.get('year', 'Unknown') for p in filtered_papers)
        print(f"\n📅 Years in Filtered Dataset:")
        for year in sorted(years.keys(), reverse=True)[:10]:
            print(f"  {year}: {years[year]}")
        
        # Confidence analysis
        confidences = [p.get('filter_confidence', 0.0) for p in filtered_papers]
        if confidences and any(c > 0 for c in confidences):
            avg_confidence = sum(confidences) / len(confidences)
            print(f"\n🎯 Filter Confidence:")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  High confidence (>0.7): {sum(1 for c in confidences if c > 0.7)}")
            print(f"  Medium confidence (0.4-0.7): {sum(1 for c in confidences if 0.4 <= c <= 0.7)}")
            print(f"  Low confidence (<0.4): {sum(1 for c in confidences if c < 0.4)}")
        
        # Show example exclusions from metadata
        if 'category_stats' in metadata:
            exclusion_count = metadata['category_stats'].get('not_ai_ethics', 0)
            print(f"\n❌ Exclusions: {exclusion_count} papers marked as 'not_ai_ethics'")
        
    except FileNotFoundError:
        print("❌ Filtered dataset not found. Run filtering first with --filter flag.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Gemini-powered dataset filtering and analysis")
    parser.add_argument('--filter', action='store_true', help='Apply Gemini filtering and create combined dataset')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing filtered dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of papers per batch (default: 50)')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between API calls in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    if args.filter:
        print("🚀 Starting Gemini-powered filtering process...")
        print("This will take a while due to API rate limiting.")
        filter_and_combine_datasets(batch_size=args.batch_size, delay=args.delay)
    elif args.analyze:
        analyze_filtered_dataset()
    else:
        print("Use --filter to create filtered dataset, or --analyze to analyze existing dataset")
        print("Make sure to set GEMINI_API_KEY environment variable for filtering")

if __name__ == "__main__":
    main()
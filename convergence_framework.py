#!/usr/bin/env python3
"""
Theoretical Framework: Demonstrating Convergent Interests Using Semantic Similarity

This document outlines the approach for quantitatively demonstrating convergent interests
across research silos using dense embeddings, with robustness to different framings.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import argparse

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    papers = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line.strip()))
    except FileNotFoundError:
        return []
    return papers

class ConvergenceFramework:
    """
    Theoretical framework for measuring convergent interests across research silos.
    """
    
    def __init__(self):
        """Initialize the framework."""
        pass
    
    def theoretical_approach(self):
        """
        Outline the theoretical approach for semantic convergence analysis.
        """
        
        print("🎯 THEORETICAL FRAMEWORK: SEMANTIC CONVERGENCE ANALYSIS")
        print("="*70)
        
        print("""
1. DENSE EMBEDDINGS FOR CROSS-SILO ANALYSIS

   Goal: Identify convergent interests despite different framings/vocabularies
   
   a) Embedding Strategy:
      - Use transformer-based sentence embeddings (e.g., SentenceTransformers)
      - Combine title + abstract for richer semantic representation
      - Models like 'all-MiniLM-L6-v2' are good for cross-domain similarity
   
   b) Robustness to Framings:
      - Embeddings capture semantic meaning beyond surface vocabulary
      - Different communities may use different terms for same concepts
      - Example: "algorithmic bias" (CS) vs "digital discrimination" (law)

2. CLUSTERING FOR THEMATIC CONVERGENCE

   a) Multi-level Clustering:
      - Hierarchical clustering with cosine similarity
      - DBSCAN for density-based natural groupings
      - K-means when approximate theme count is known
   
   b) Cross-Silo Metrics:
      - Diversity Score: Entropy of source distribution in clusters
      - Coverage Score: How many different communities are represented
      - Coherence Score: Internal semantic similarity within clusters

3. QUANTITATIVE CONVERGENCE MEASURES

   a) Cluster Diversity Index (CDI):
      CDI = -Σ(pi * log2(pi))  where pi = proportion from source i
      
      High CDI → High convergence (multiple sources equally represented)
      Low CDI → Silo dominance (one source dominates cluster)
   
   b) Inter-Silo Similarity (ISS):
      For papers from different sources in same cluster:
      ISS = mean(cosine_similarity(embeddings_across_sources))
      
      High ISS → Strong semantic alignment despite different origins
   
   c) Framing Robustness Score (FRS):
      FRS = correlation(similarity_scores, source_differences)
      
      Low FRS → Robust to framing (similarity independent of source)
      High FRS → Framing-dependent (same-source papers more similar)

4. METHODOLOGICAL STEPS

   Step 1: Data Preparation
   - Filter to ethics papers from different communities (ACL, law journals, etc.)
   - Ensure comparable abstracts/descriptions available
   
   Step 2: Embedding Generation  
   - Use pre-trained sentence transformers
   - Cache embeddings for efficiency
   - Normalize vectors for cosine similarity
   
   Step 3: Clustering Analysis
   - Apply multiple clustering methods
   - Validate with silhouette scores
   - Identify clusters with high diversity scores
   
   Step 4: Convergence Detection
   - Calculate CDI for each cluster
   - Measure ISS within high-diversity clusters  
   - Compute FRS across the dataset
   
   Step 5: Validation & Interpretation
   - Manual inspection of convergent clusters
   - Domain expert validation
   - Temporal analysis (do convergent themes emerge over time?)

5. EXPECTED RESULTS & INTERPRETATION

   a) Strong Convergence Evidence:
      - High-diversity clusters (CDI > 1.0)
      - High inter-silo similarity (ISS > 0.7)
      - Low framing dependency (FRS < 0.3)
      - Representative papers show genuine thematic overlap
   
   b) Convergence Patterns:
      - "Algorithmic fairness" cluster spanning CS, law, policy
      - "AI transparency" cluster across technical and social sciences
      - "Bias detection" cluster from ML and social psychology
   
   c) Temporal Convergence:
      - Early papers: high source segregation
      - Recent papers: increasing cross-silo similarity
      - Measure trend in average CDI over time

6. ROBUSTNESS CHECKS

   a) Multiple Embedding Models:
      - Test with different transformer models
      - Compare domain-specific vs general embeddings
      - Ensemble approaches for stability
   
   b) Clustering Validation:
      - Multiple clustering algorithms
      - Different parameter settings
      - Stability analysis (bootstrap clustering)
   
   c) Manual Validation:
      - Expert review of convergent clusters
      - Qualitative analysis of representative papers
      - Cross-validation with citation networks

7. STATISTICAL SIGNIFICANCE

   a) Null Hypothesis Testing:
      - H0: No semantic convergence (random clustering by topic)
      - H1: Systematic cross-silo semantic alignment
   
   b) Permutation Tests:
      - Randomly shuffle source labels
      - Compare observed CDI to null distribution
      - p-value for statistical significance
   
   c) Effect Size Measures:
      - Cohen's d for CDI differences
      - Confidence intervals for ISS
      - Bootstrap significance tests
        """)
        
    def convergence_metrics_demo(self):
        """Demonstrate the convergence metrics with synthetic data."""
        
        print("\n📊 CONVERGENCE METRICS DEMONSTRATION")
        print("="*50)
        
        # Synthetic example data
        clusters = {
            'high_convergence': {
                'acl_papers': 15,
                'law_papers': 12, 
                'policy_papers': 8,
                'avg_similarity': 0.78
            },
            'moderate_convergence': {
                'acl_papers': 25,
                'law_papers': 5,
                'policy_papers': 3,
                'avg_similarity': 0.65
            },
            'low_convergence': {
                'acl_papers': 30,
                'law_papers': 1,
                'policy_papers': 0,
                'avg_similarity': 0.82
            }
        }
        
        for cluster_name, data in clusters.items():
            total = data['acl_papers'] + data['law_papers'] + data['policy_papers']
            
            # Calculate CDI (Cluster Diversity Index)
            proportions = [
                data['acl_papers'] / total,
                data['law_papers'] / total, 
                data['policy_papers'] / total
            ]
            cdi = -sum(p * np.log2(p) for p in proportions if p > 0)
            
            # Interpretation
            if cdi > 1.2:
                convergence_level = "HIGH"
            elif cdi > 0.8:
                convergence_level = "MODERATE"  
            else:
                convergence_level = "LOW"
            
            print(f"\n{cluster_name.upper()}:")
            print(f"  Papers: ACL={data['acl_papers']}, Law={data['law_papers']}, Policy={data['policy_papers']}")
            print(f"  CDI: {cdi:.3f} → {convergence_level} convergence")
            print(f"  Avg Similarity: {data['avg_similarity']:.3f}")
            
            if convergence_level == "HIGH":
                print(f"  💡 Strong evidence of cross-silo alignment")
            elif convergence_level == "LOW":
                print(f"  ⚠️  Potential silo - dominated by single community")

def analyze_existing_data():
    """Analyze existing data for potential convergence patterns."""
    
    print("\n🔍 ANALYZING EXISTING DATA FOR CONVERGENCE POTENTIAL")
    print("="*60)
    
    # Load ethics filter results if available
    filter_results = load_jsonl('data/ethics_filter_results.jsonl')
    
    if not filter_results:
        print("❌ No filter results found. Run filtering first:")
        print("   python3 create_filtered_dataset.py --filter")
        return
    
    # Basic analysis of existing data
    source_counts = Counter(paper.get('source', 'unknown') for paper in filter_results)
    category_counts = Counter(paper.get('filter_category', 'unknown') for paper in filter_results)
    
    print(f"📊 Available Ethics Papers: {len(filter_results)} total")
    print(f"   Sources: {dict(source_counts)}")
    print(f"   Categories: {dict(category_counts)}")
    
    # Check for potential cross-source categories
    cross_source_potential = defaultdict(set)
    for paper in filter_results:
        category = paper.get('filter_category', 'unknown')
        source = paper.get('source', 'unknown')
        cross_source_potential[category].add(source)
    
    print(f"\n🎯 Potential Cross-Source Convergence:")
    for category, sources in cross_source_potential.items():
        if len(sources) > 1:
            print(f"   {category}: {list(sources)} ({len(sources)} sources)")
    
    # Recommendations
    print(f"\n💡 Recommendations for Semantic Analysis:")
    
    if len(filter_results) > 100:
        print(f"   ✅ Sufficient data volume ({len(filter_results)} papers)")
    else:
        print(f"   ⚠️  Limited data volume ({len(filter_results)} papers) - consider expanding")
    
    if len(source_counts) >= 2:
        print(f"   ✅ Multiple sources available ({len(source_counts)} sources)")
        print(f"   📈 Ready for cross-silo convergence analysis")
        print(f"\n   Next steps:")
        print(f"   1. Install dependencies: bash install_semantic_deps.sh")
        print(f"   2. Run analysis: python3 semantic_convergence_analysis.py --method hierarchical --visualize")
    else:
        print(f"   ❌ Need multiple sources for convergence analysis")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Semantic convergence theoretical framework")
    parser.add_argument('--demo', action='store_true', help='Show theoretical framework and metrics')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing data potential')
    
    args = parser.parse_args()
    
    framework = ConvergenceFramework()
    
    if args.demo:
        framework.theoretical_approach()
        framework.convergence_metrics_demo()
    elif args.analyze:
        analyze_existing_data()
    else:
        framework.theoretical_approach()
        analyze_existing_data()

if __name__ == "__main__":
    main()
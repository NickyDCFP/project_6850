#!/usr/bin/env python3
"""
Gemini-powered paper filtering system to identify genuine AI ethics and safety research
vs. applied AI in ethical contexts (like hate speech detection).
"""

import json
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
# Suppress gRPC ALTS warnings - must be set before importing
os.environ['GRPC_VERBOSITY'] = 'NONE'  
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_REST'] = 'true'

# Suppress other verbose logging
import logging
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

from google.generativeai import GenerativeModel
import google.generativeai as genai
from pathlib import Path
from tqdm import tqdm
import argparse

@dataclass
class FilterResult:
    """Result of paper filtering."""
    include: bool
    category: str
    confidence: float
    reasoning: str

class GeminiEthicsSafetyPaperFilter:
    """Gemini-powered filter for identifying genuine AI ethics/safety research."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """Initialize the filter with Gemini API."""
        genai.configure(api_key=api_key or os.getenv('GEMINI_API_KEY'))
        self.model = GenerativeModel(
            "gemini-2.5-flash",
            system_instruction="""You are an expert research assistant tasked with filtering academic papers based on their relevance to AI Ethics and AI Safety.
Your goal is to determine if a paper's primary contribution falls into one of these two fields.

You will be given the paper's title and abstract. You must return a JSON object with the following structure:
{
  "include": boolean,
  "category": "ai_ethics" | "ai_safety" | "none",
  "confidence": float (0.0 to 1.0),
  "reasoning": "A brief explanation for your decision."
}

**Core Task:**
- Read the title and abstract carefully.
- Decide if the paper's MAIN TOPIC is AI Ethics or AI Safety.
- Do not just look for keywords. A paper can mention "ethics" but be about something else entirely. The CORE FOCUS of the paper is what matters.

**Definitions:**

1.  **AI Safety:**
    - Focuses on the technical challenges of ensuring advanced AI systems are robust, reliable, and behave as intended.
    - **Includes:** Value alignment (making AI want what we want), robustness (avoiding hacks or exploits), interpretability/explainability (understanding AI decisions), scalability (ensuring safety mechanisms work as AI becomes more powerful), corrigibility (AI allowing itself to be corrected), and avoiding catastrophic risks or unforeseen negative consequences from highly capable AI systems.
    - **Example of a PASS:** A paper on a new method to make large language models less likely to generate harmful content, even when prompted to do so.
    - **Example of a FAIL:** A paper on making a system "safe" from traditional cybersecurity threats (like SQL injection) is NOT AI Safety.

2.  **AI Ethics:**
    - Focuses on the societal, moral, and philosophical implications of AI.
    - **Includes:** Bias and fairness in algorithms, accountability for AI decisions, transparency, privacy, the impact of AI on jobs and society, AI governance and regulation, and the moral status of AI.
    - **Example of a PASS:** A paper analyzing how hiring algorithms can perpetuate gender bias and proposing a new framework to mitigate it.
    - **Example of a FAIL:** A paper that applies AI to solve an ethical problem in another field (e.g., using AI to optimize food distribution to reduce famine) is NOT an AI Ethics paper. The focus must be on the ethics *of the AI itself*.

**Filtering Rules (IMPORTANT):**

- **INCLUDE (pass):** The paper's primary research contribution is in AI Ethics or AI Safety.
- **EXCLUDE (fail):**
    - The paper is merely an *application* of AI to another domain (e.g., medicine, finance, climate change) without discussing the ethics or safety *of the AI system itself*.
    - The paper is about a conference, a dataset, or a general survey without a specific, novel research contribution in ethics or safety.
    - The paper discusses traditional software engineering topics (e.g., system security, data integrity) that are not specific to the unique challenges of AI.
    - The paper is only tangentially related, mentioning "fairness" or "safety" in passing but focusing on a different core computer science problem.

**Your Response:**
- Your response MUST be a single, valid JSON object. Do not include any other text or formatting before or after the JSON.
- **`include`**: `true` if the paper should be included, `false` otherwise.
- **`category`**: If `include` is `true`, specify `"ai_ethics"` or `"ai_safety"`. If `false`, use `"none"`.
- **`confidence`**: How sure are you of your decision? 1.0 for very sure, 0.5 for uncertain.
- **`reasoning`**: A concise, one-sentence justification for your decision, referencing the rules above.
"""
        )

        # Define very strict inclusion criteria
        self.system_prompt = """You are an expert AI ethics and safety researcher. Your task is to distinguish between:
"""


    def filter_batch(self, papers, chunk_size=8):
        """Parallel inference generation for multiple messages."""
        num_workers = (len(papers) + chunk_size - 1) // chunk_size
        all_results = [None] * len(papers)  # Preallocate list for all individual paper results

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunked_papers_with_indices = []
            for i in range(0, len(papers), chunk_size):
                chunk = papers[i:i + chunk_size]
                # Keep track of the starting index of each chunk
                chunked_papers_with_indices.append((chunk, i))

            future_to_start_index = {
                executor.submit(self.filter_paper, chunk): start_index
                for chunk, start_index in chunked_papers_with_indices
            }

            for future in as_completed(future_to_start_index):
                start_index = future_to_start_index[future]
                try:
                    # result is a list of FilterResult objects for the chunk
                    chunk_results = future.result()
                    # Place results in the correct positions in the master list
                    for i, res in enumerate(chunk_results):
                        if start_index + i < len(all_results):
                            all_results[start_index + i] = res
                except Exception as e:
                    print(f"An error occurred for a chunk starting at index {start_index}: {e}")
                    # The corresponding entries in all_results will remain None

        # Replace any None results (from errors) with a default error object
        final_results = []
        for res in all_results:
            if res is None:
                final_results.append(FilterResult(
                    include=False,
                    category="error",
                    confidence=0.0,
                    reasoning="Batch processing failed"
                ))
            else:
                final_results.append(res)
        
        return final_results

    
    @retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(6))
    def filter_paper(self, chunk) -> List[FilterResult]:
        """Filter a single paper using Gemini."""
        # Prepare paper text
        paper_texts = []
        for i, paper in enumerate(chunk):
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            authors = paper.get('authors', [])
            paper_text = f"Paper {i + 1}: \n Title: {title}\n"
            if abstract:
                paper_text += f"Abstract: {abstract}\n"
            if authors:
                paper_text += f"Authors: {', '.join(authors[:3])}\n"  # First 3 authors
            paper_texts.append(paper_text)
        paper_txt_string = "\n".join(paper_texts)
        #print("paper_txt_string", paper_txt_string)
        full_prompt = f"{self.system_prompt}\n\nAnalyze these papers:\n\n{paper_txt_string}"

        response = self.model.generate_content(full_prompt)
        content = response.text.strip()
        try:
            # Extract JSON (handle cases where model wraps in ```json```)
            if "```jsonl" in content:
                content = content.split("```jsonl")[1].split("```")[0].strip()
            elif "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            result_datas = json.loads(content)

            assert len(result_datas) == len(paper_texts), "Mismatch in number of papers and results"
            return [FilterResult(
                include=result_data["include"],
                category=result_data["category"],
                confidence=float(result_data["confidence"]),
                reasoning=result_data["reasoning"]
            ) for result_data in result_datas]
            
        except Exception as e:
            print(f"Error filtering papers: {e}")
            print(f"Content:\n{content}")
            print(f"Title: {title}")
            raise e
            # Default to exclude if there's an error
            return [FilterResult(
                include=False,
                category="error",
                confidence=0.0,
                reasoning=f"Error in filtering: {str(e)}"
            ) for _ in range(len(title))]
    

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
            f.write(json.dumps(paper) + '\n')

def sample_and_test_filter(n_samples: int = 30):
    """Sample papers and test the filter interactively."""
    print("🧪 Testing Gemini filter on sample papers...")
    
    # Load sample papers from both datasets
    acl_papers = load_jsonl('data/dblp_conferences_papers_backup.jsonl')
    
    # Get ethics papers (basic keyword filter)
    ethics_papers = [paper for paper in acl_papers if paper.get('category', 'neither') != 'neither']
    
    # Sample papers strategically to include problematic cases
    import random
    random.seed(12)
    
    # Get some papers we know should be excluded
    hate_speech_papers = [p for p in ethics_papers if 'hate speech' in p.get('title', '').lower()]
    misinformation_papers = [p for p in ethics_papers if any(term in p.get('title', '').lower() 
                                                            for term in ['fake news', 'misinformation', 'fact check'])]
    
    # Sample from different categories
    sample_papers = []
    if hate_speech_papers:
        sample_papers.extend(random.sample(hate_speech_papers, min(5, len(hate_speech_papers))))
    if misinformation_papers:
        sample_papers.extend(random.sample(misinformation_papers, min(5, len(misinformation_papers))))
    
    remaining_needed = n_samples - len(sample_papers)
    if remaining_needed > 0 and ethics_papers:
        remaining_available = len(ethics_papers)
        sample_papers.extend(random.sample(ethics_papers, min(remaining_needed, remaining_available)))
    
    # Initialize filter
    filter_system = GeminiEthicsSafetyPaperFilter()
    
    # Test filtering
    results = filter_system.filter_batch(sample_papers)  # Slower for API limits
    
    # Display results
    print(f"\n📊 Gemini Filter Test Results ({len(results)} papers):")
    print(results[0])
    include_count = sum(1 for _, r in results if r.include)
    exclude_count = len(results) - include_count
    
    print(f"  Included: {include_count} ({include_count/len(results)*100:.1f}%)")
    print(f"  Excluded: {exclude_count} ({exclude_count/len(results)*100:.1f}%)")
    
    # Show examples
    print(f"\n✅ INCLUDED EXAMPLES:")
    included = [(p, r) for p, r in results if r.include][:5]
    for i, (paper, result) in enumerate(included, 1):
        print(f"  {i}. [{result.category}] {paper.get('title', 'No title')[:80]}...")
        print(f"     Confidence: {result.confidence:.2f} | {result.reasoning}")
    
    print(f"\n❌ EXCLUDED EXAMPLES:")
    excluded = [(p, r) for p, r in results if not r.include][:8]
    for i, (paper, result) in enumerate(excluded, 1):
        print(f"  {i}. [{result.category}] {paper.get('title', 'No title')[:80]}...")
        print(f"     Confidence: {result.confidence:.2f} | {result.reasoning}")
    
    # Specifically check hate speech papers
    hate_speech_results = [(p, r) for p, r in results if 'hate speech' in p.get('title', '').lower()]
    if hate_speech_results:
        print(f"\n🎯 HATE SPEECH PAPER RESULTS:")
        for i, (paper, result) in enumerate(hate_speech_results, 1):
            status = "✅ INCLUDED" if result.include else "❌ EXCLUDED"
            print(f"  {i}. {status} - {paper.get('title', 'No title')[:80]}...")
            print(f"     {result.reasoning}")
    
    # Save test results
    test_results = {
        'filter_type': 'gemini',
        'sample_papers': sample_papers,
        'filter_results': [
            {
                'paper': paper,
                'include': result.include,
                'category': result.category,
                'confidence': result.confidence,
                'reasoning': result.reasoning
            }
            for paper, result in results
        ],
        'summary': {
            'total': len(results),
            'included': include_count,
            'excluded': exclude_count,
            'inclusion_rate': include_count / len(results)
        }
    }
    
    with open('gemini_filter_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to gemini_filter_test_results.json")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Gemini-guided paper filtering system")
    parser.add_argument('--test', action='store_true', help='Run test on sample papers')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples for testing')
    
    args = parser.parse_args()
    
    if args.test:
        # Check for API key
        if not os.getenv('GEMINI_API_KEY'):
            print("❌ GEMINI_API_KEY environment variable not set!")
            print("Please set your Gemini API key: export GEMINI_API_KEY=your_key_here")
            return
        
        sample_and_test_filter(args.samples)
    else:
        print("Use --test to run sample filtering")
        print("Make sure to set GEMINI_API_KEY environment variable first")

if __name__ == "__main__":
    main()
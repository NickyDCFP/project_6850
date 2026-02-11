#!/usr/bin/env python3
"""
Performs a focused diagnostic test on the Semantic Scholar title search functionality.

This script takes a single paper title, loads the API key from the .env file,
and calls the title search function in verbose mode to inspect the exact API
request and response.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from add_abstracts import search_semanticscholar_by_title

def main():
    # Load environment variables from .env file
    load_dotenv(repo_root / '.env')

    # Set the verbose flag for the add_abstracts module
    os.environ['ADD_ABSTRACTS_VERBOSE'] = 'true'

    # Example title from NeurIPS 2020 known to be missing an abstract
    test_title = "Unsupervised Learning of Visual 3D Keypoints for Control"

    print(f"--- Starting Diagnostic for Title Search ---")
    print(f"Searching for title: '{test_title}'")
    print("-" * 40)

    # Call the function which now has built-in diagnostics
    abstract = search_semanticscholar_by_title(test_title)

    print("-" * 40)
    print("--- Diagnostic Complete ---")
    if abstract:
        print(f"SUCCESS: Found abstract of length {len(abstract)}.")
        print(f"Abstract: {abstract[:150]}...")
    else:
        print("FAILURE: No abstract was recovered.")
        print("Check the diagnostic output above to see the raw API response.")

if __name__ == '__main__':
    main()

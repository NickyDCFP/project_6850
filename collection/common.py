"""Paths, venue list and small helpers shared by the collection-update scripts."""
import json
import os
import re
import sys
import unicodedata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

RAW_DIR = os.path.join(REPO_ROOT, 'data', 'raw')            # downloaded dumps (gitignored)
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'updated')
COLLECTED_DIR = os.path.join(OUT_DIR, 'collected')          # every paper, one file per venue-year
FILTERED_DIR = os.path.join(OUT_DIR, 'filtered')            # papers that pass both filters
CITATIONS_DIR = os.path.join(OUT_DIR, 'citations')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')                  # API responses, so every stage can resume
SHIPPED_DATASET_DIR = os.path.join(REPO_ROOT, 'data', 'final_dataset')

FIRST_YEAR = 2020
# DBLP venue tokens, as used in the shipped file names, -> DBLP key prefix.
DBLP_VENUES = {
    'aaai': 'conf/aaai/',
    'aies': 'conf/aies/',
    'fat': 'conf/fat/',
    'iclr': 'conf/iclr/',
    'icml': 'conf/icml/',
    'neurips': 'conf/nips/',
    'satml': 'conf/satml/',
}
# Venues whose every paper goes to the LLM filter, not only keyword matches (as in scripts/advanced_filter.py).
SMALL_VENUES = {'aies', 'fat', 'satml'}


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def norm_title(title):
    """Lowercase ASCII letters and digits only, for matching titles across sources."""
    s = unicodedata.normalize('NFKD', title or '').encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '', s)


def has_abstract(rec):
    return len((rec.get('abstract') or '').strip()) > 50

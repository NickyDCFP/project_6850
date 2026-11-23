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
# Every paper id in collected/, per file (committed; collected/ itself is gitignored). The collect scripts keep
# only these papers, so a newer dump rebuilds the same paper set instead of adding newly indexed papers.
MANIFEST = os.path.join(OUT_DIR, 'collected_manifest.tsv')

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


def load_manifest():
    """{'<venue>_<year>': [ids in file order]} from MANIFEST, or None if it does not exist."""
    if not os.path.exists(MANIFEST):
        return None
    out = {}
    with open(MANIFEST, encoding='utf-8') as f:
        next(f)  # header
        for line in f:
            stem, pid = line.rstrip('\n').split('\t')
            out.setdefault(stem, []).append(pid)
    return out


def apply_manifest(groups, manifest, owns):
    """Restrict freshly parsed {stem: [records]} to the manifest's papers, in manifest order.

    `owns(stem)` says which manifest files this script produces. Papers are matched by id across all parsed
    files, so a paper whose year changed in the dump still lands in its original file. Returns the new
    groups and (dropped, missing): ids parsed but not in the manifest, and manifest ids absent from the dump.
    """
    by_id = {r['id']: r for recs in groups.values() for r in recs}
    stems = [s for s in manifest if owns(s)]
    wanted = {pid for s in stems for pid in manifest[s]}
    out = {s: [by_id[pid] for pid in manifest[s] if pid in by_id] for s in stems}
    dropped = sorted(set(by_id) - wanted)
    missing = sorted(wanted - set(by_id))
    return out, dropped, missing


def report_manifest(dropped, missing):
    print(f'manifest: kept the pinned paper set; ignored {len(dropped)} papers not in it '
          f'(newly indexed since the corpus was built; pass --all-papers to include them)')
    if missing:
        print(f'WARNING: {len(missing)} pinned papers are not in this dump (removed or re-keyed upstream), e.g. '
              + ', '.join(missing[:5]))


def update_manifest(groups, owns):
    """Replace this script's files in MANIFEST with the ids in `groups` (used with --all-papers)."""
    manifest = {s: ids for s, ids in (load_manifest() or {}).items() if not owns(s)}
    manifest.update({s: [r['id'] for r in recs] for s, recs in groups.items()})
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST + '.tmp', 'w', encoding='utf-8') as f:
        f.write('file\tid\n')
        for s in sorted(manifest):
            f.writelines(f'{s}\t{pid}\n' for pid in manifest[s])
    os.replace(MANIFEST + '.tmp', MANIFEST)
    print(f'manifest updated: {MANIFEST}')

"""Table 1: safety / ethics / mixed split of papers and authors.

  python -m reproduction.table1_labels

Does NOT reproduce the paper's split. 662 FAccT/AIES/SaTML papers match no keyword at all, and
analyze_aggregated.load_combined_data labels them by venue (FAccT/AIES -> ethics, SaTML -> safety).
Labelling the 93 SaTML ones as ethics instead (their stored keyword_category) gives exactly the
paper's 2,959 pure-ethics papers, but 142 papers would still have to move from safety to mixed.
"""
from collections import Counter

from reproduction.common import E, M, S, category, load_papers, normalize_author, write_csv

PAPER = {'papers': (2874, 2959, 609), 'authors': (9634, 7264, 3792), 'authors >=2 papers': (1233, 1003, 2328)}


def author_counts(papers, min_papers, name):
    scores = {}
    for p in papers:
        for a in {name(x) for x in p['authors']}:
            s, e, n = scores.get(a, (0.0, 0.0, 0))
            scores[a] = (s + p['safety_score'], e + p['ethics_score'], n + 1)
    return Counter(category(s, e) for s, e, n in scores.values() if n >= min_papers)


def main():
    papers = load_papers('full')
    raw = lambda a: a
    rows = [
        ('papers', Counter(category(p['safety_score'], p['ethics_score']) for p in papers)),
        ('authors', author_counts(papers, 1, raw)),
        ('authors >=2 papers', author_counts(papers, 2, raw)),
        ('authors >=2 papers (normalized names, as in the networks)', author_counts(papers, 2, normalize_author)),
    ]
    zero = sum(1 for p in papers if p['source_file'] == 'satml' and p.get('keyword_matches', {}) == {'safety': 0, 'ethics': 0})
    print(f'Table 1 (all {len(papers)} papers): pure safety / pure ethics / mixed')
    out = []
    for name, c in rows:
        ref = PAPER.get(name)
        print(f'  {name:<58} {c[S]:>6} {c[E]:>6} {c[M]:>6}' + (f'   paper: {ref[0]} / {ref[1]} / {ref[2]}' if ref else ''))
        out.append([name, c[S], c[E], c[M]])
    print(f'\n  SaTML papers with zero keyword matches (labelled safety by the loader): {zero}')
    write_csv('table1_labels.csv', ['row', 'pure_safety', 'pure_ethics', 'mixed'], out)


if __name__ == '__main__':
    main()

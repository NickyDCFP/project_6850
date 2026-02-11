#!/usr/bin/env python3
"""
Aggregate per-author cumulative safety + ethics statistics.

Outputs:
 - author_scores_aggregated.csv

Usage:
 python3 compute_author_scores.py --min-papers 2

This script reuses the repository's loading/normalization where possible.
"""
import argparse
import csv
import os
from collections import defaultdict

from analyze_aggregated import load_combined_data, normalize_author


def aggregate_authors(papers):
    # Map normalized author -> aggregation dict
    agg = defaultdict(lambda: {
        "author": "",
        "n_papers": 0,
        "total_s": 0.0,
        "total_e": 0.0,
        "n_mixed": 0,
        "venues": set(),
        "paper_ids": [],
    })

    for p in papers:
        authors = p.get("authors", []) or []
        s = float(p.get("safety_score", 0) or 0)
        e = float(p.get("ethics_score", 0) or 0)
        paper_id = p.get("paper_id") or p.get("id") or p.get("title")
        venue = p.get("venue") or p.get("booktitle") or p.get("source") or ""

        for a in authors:
            na = normalize_author(a)
            if not na:
                continue
            entry = agg[na]
            entry["author"] = na
            entry["n_papers"] += 1
            entry["total_s"] += s
            entry["total_e"] += e
            if (s > 0) and (e > 0):
                entry["n_mixed"] += 1
            if venue:
                entry["venues"].add(venue)
            if paper_id:
                entry["paper_ids"].append(paper_id)

    # Finalize entries into list of dicts
    rows = []
    for na, v in agg.items():
        total_se = v["total_s"] + v["total_e"]
        avg_s = v["total_s"] / v["n_papers"] if v["n_papers"] else 0.0
        avg_e = v["total_e"] / v["n_papers"] if v["n_papers"] else 0.0
        purity = (v["total_s"] / total_se) if total_se > 0 else None
        rows.append({
            "author": v["author"],
            "n_papers": v["n_papers"],
            "total_s": round(v["total_s"], 4),
            "total_e": round(v["total_e"], 4),
            "total_s_e": round(total_se, 4),
            "avg_s": round(avg_s, 4),
            "avg_e": round(avg_e, 4),
            "purity_safety": round(purity, 4) if purity is not None else "",
            "n_mixed_papers": v["n_mixed"],
            "venues_sample": "; ".join(sorted(list(v["venues"]))[:5]),
            "paper_ids": ";".join(map(str, v["paper_ids"]))[:2000],
        })

    return rows


def save_csv(rows, out_path):
    if not rows:
        print("No author rows to write.")
        return
    keys = [
        "author",
        "n_papers",
        "total_s",
        "total_e",
        "total_s_e",
        "avg_s",
        "avg_e",
        "purity_safety",
        "n_mixed_papers",
        "venues_sample",
        "paper_ids",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["total_s_e"], reverse=True):
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-papers", type=int, default=0, help="Minimum #papers to include an author")
    parser.add_argument("--out", type=str, default="author_scores_aggregated.csv")
    args = parser.parse_args()

    print("Loading papers using analyze_aggregated.load_combined_data()...")
    papers = load_combined_data()
    print(f"Loaded {len(papers)} papers")

    rows = aggregate_authors(papers)
    print(f"Aggregated {len(rows)} unique authors")

    # apply min-papers filter
    if args.min_papers > 0:
        rows = [r for r in rows if int(r["n_papers"]) >= args.min_papers]
        print(f"After min_papers >= {args.min_papers}: {len(rows)} authors remain")

    save_csv(rows, args.out)
    print(f"Wrote aggregated CSV to {os.path.abspath(args.out)}")

    # Print top-20 by total_s_e
    top = sorted(rows, key=lambda x: x["total_s_e"], reverse=True)[:20]
    print("\nTop-20 authors by cumulative (S+E):")
    for i, r in enumerate(top, 1):
        print(f"{i:2d}. {r['author']} — papers:{r['n_papers']}, total_s_e:{r['total_s_e']}, purity_safety:{r['purity_safety']}, n_mixed:{r['n_mixed_papers']}")


if __name__ == "__main__":
    main()

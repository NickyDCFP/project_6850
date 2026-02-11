#!/usr/bin/env python3
"""
Stream-parse a DBLP XML dump and write selected records to JSONL.

Usage:
  python3 parse_dblp_xml.py --input dblp.xml --output dblp_filtered.jsonl --type inproceedings --venue nips --year 2020

This script is memory-efficient (iterative parsing) and extracts basic fields:
- id (key attribute)
- type (tag name)
- title
- authors (list)
- year
- venue (booktitle or journal)
- url (if present)

Filters supported: --type, --venue (substring match), --year, --limit
"""
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Optional
import re
import html
import os
from pathlib import Path
import time

DBLP_ENTRY_TAGS = {"article", "inproceedings", "proceedings", "book", "incollection", "phdthesis", "mastersthesis", "www"}


def parse_dtd(dtd_path: str) -> dict:
    """Parse a DTD file for entity declarations of the form:
    <!ENTITY name "&#NNN;"> or <!ENTITY name "&#xHH;"> and return a dict name->codepoint int.
    """
    entity_map = {}
    if not os.path.exists(dtd_path):
        return entity_map
    pattern = re.compile(r'<!ENTITY\s+([A-Za-z0-9]+)\s+"&#(x?)([0-9A-Fa-f]+);"')
    with open(dtd_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                name = m.group(1)
                is_hex = bool(m.group(2))
                num = m.group(3)
                code = int(num, 16) if is_hex else int(num, 10)
                entity_map[name] = code
    return entity_map


# Modify EntityFixingFile to accept external mapping
class EntityFixingFile:
    """Line-buffered wrapper that replaces named entities per-line using a provided map.

    This approach avoids splitting entities across read() calls by processing whole lines
    and serving reads from an internal buffer. It's simpler and reliable because entity
    references almost never span across lines in DBLP XML.
    """
    def __init__(self, f, extra_map=None):
        self.f = f
        self._line_iter = iter(f)
        # merge builtin html entities and extra_map (from DTD) with priority to extra_map
        base_map = html.entities.name2codepoint.copy()
        if extra_map:
            base_map.update(extra_map)
        self._map = base_map
        # pattern matches named entities like &eacute;
        self._pattern = re.compile(r'&([A-Za-z][A-Za-z0-9]+);')
        self._buffer = ''
        self._eof = False

    def _repl(self, m):
        name = m.group(1)
        if name in self._map:
            return f"&#{self._map[name]};"
        return m.group(0)

    def _fill_buffer(self, min_bytes=1):
        """Read lines until buffer has at least min_bytes or EOF reached."""
        while len(self._buffer) < min_bytes and not self._eof:
            try:
                line = next(self._line_iter)
            except StopIteration:
                self._eof = True
                break
            # process the line and append
            processed = self._pattern.sub(self._repl, line)
            self._buffer += processed

    def read(self, size=-1):
        if size == -1:
            # consume everything
            if not self._eof:
                for line in self._line_iter:
                    self._buffer += self._pattern.sub(self._repl, line)
                self._eof = True
            out = self._buffer
            self._buffer = ''
            return out

        # ensure buffer has at least size bytes (or until EOF)
        self._fill_buffer(min_bytes=size)
        out = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return out

    def readline(self, size=-1):
        # if there's a newline in buffer, return up to it
        nl_idx = self._buffer.find('\n')
        if nl_idx != -1:
            # include newline
            end = nl_idx + 1
            out = self._buffer[:end]
            self._buffer = self._buffer[end:]
            return out

        # otherwise, try to read the next line from the underlying iterator
        if not self._eof:
            try:
                line = next(self._line_iter)
                processed = self._pattern.sub(self._repl, line)
                out = self._buffer + processed
                self._buffer = ''
                return out
            except StopIteration:
                self._eof = True
        # EOF: return whatever remains
        out = self._buffer
        self._buffer = ''
        return out

    def __iter__(self):
        # iterate over processed lines
        for line in self.f:
            yield self._pattern.sub(self._repl, line)

    def close(self):
        try:
            return self.f.close()
        finally:
            self._buffer = ''
            self._eof = True


def extract_record(elem: ET.Element) -> dict:
    record = {
        "id": elem.get("key"),
        "type": elem.tag,
        "title": None,
        "authors": [],
        "year": None,
        "venue": None,
        "url": None,
    }

    for child in elem:
        tag = child.tag.lower()
        text = (child.text or "").strip()
        if tag == "author":
            record["authors"].append(text)
        elif tag == "title":
            record["title"] = text
        elif tag == "year":
            record["year"] = text
        elif tag in ("booktitle", "journal"):
            record["venue"] = text
        elif tag == "url":
            record["url"] = text
        # keep other fields out for now

    return record


def matches_filters(record: dict, type_filter: Optional[str], venue_filter: Optional[str], year_filter: Optional[str]) -> bool:
    if type_filter and record.get("type") != type_filter:
        return False
    if year_filter and record.get("year") != year_filter:
        return False
    if venue_filter:
        venue = (record.get("venue") or "").lower()
        if venue_filter.lower() not in venue:
            return False
    return True


def parse_dblp(input_path: str, output_path: str, type_filter: Optional[str], venue_filter: Optional[str], year_filter: Optional[str], limit: Optional[int], entity_map: Optional[dict]=None):
    count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        # Use iterative parsing for memory efficiency
        with open(input_path, "r", encoding="utf-8", errors="replace") as inf:
            wrapped = EntityFixingFile(inf, extra_map=entity_map)
            context = ET.iterparse(wrapped, events=("end",))
            for event, elem in context:
                tag = elem.tag
                if tag in DBLP_ENTRY_TAGS:
                    record = extract_record(elem)
                    if matches_filters(record, type_filter, venue_filter, year_filter):
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                        if limit and count >= limit:
                            break
                    # Important: clear the element to free memory
                    elem.clear()
            # Close parser
            del context
            wrapped.close()
    print(f"Wrote {count} records to {output_path}")


# try to import tqdm for progress, fall back silently
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_dblp_multi(input_path: str, out_dir: str, type_filter: Optional[str], venues: Optional[list], years: Optional[list], entity_map: Optional[dict]=None, flush_interval: int = 1000, show_progress: bool = True, allow_venue_match: bool = False):
    """Single-pass parse that extracts records for multiple venues and years.

    - venues: list of substring tokens (case-insensitive) to match against booktitle/journal or key/url
    - years: list of year strings (e.g., ['2020','2021'])
    Writes files to out_dir/<venue>_<year>.jsonl
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # normalize targets
    venue_tokens = [v.lower() for v in (venues or [])]
    year_tokens = set(years or [])

    # prepare file handles and counters for each target
    writers = {}
    counters = {}

    def writer_for(vtoken, y):
        name = f"{vtoken.replace(' ', '_')}_{y}.jsonl"
        path = out_dir_path / name
        if (vtoken, y) not in writers:
            f = open(path, 'a', encoding='utf-8')
            writers[(vtoken, y)] = f
            counters[(vtoken, y)] = 0
        return writers[(vtoken, y)]

    total_written = 0
    processed = 0

    # open and wrap file
    with open(input_path, 'r', encoding='utf-8', errors='replace') as inf:
        wrapped = EntityFixingFile(inf, extra_map=entity_map)
        context = ET.iterparse(wrapped, events=("end",))

        # optionally wrap with tqdm if available
        iterator = context
        if show_progress and tqdm is not None:
            # we don't know total; use dynamic
            iterator = tqdm(context, desc='Parsing DBLP', unit='records')

        last_flush = time.time()
        try:
            for event, elem in iterator:
                processed += 1
                tag = elem.tag
                if tag in DBLP_ENTRY_TAGS:
                    record = extract_record(elem)
                    # type filter
                    if type_filter and record.get('type') != type_filter:
                        elem.clear(); continue

                    # match year first (fast)
                    ry = record.get('year')
                    if year_tokens and (ry not in year_tokens):
                        elem.clear(); continue

                    # prepare a lowercased venue str to match
                    rv = (record.get('venue') or '').lower()
                    key = (record.get('id') or '').lower()
                    rurl = (record.get('url') or '').lower()

                    # small alias map for common conference path tokens
                    alias_map = {
                        'neurips': ['neurips', 'nips'],
                        'nips': ['nips', 'neurips'],
                        'icml': ['icml'],
                        'iclr': ['iclr'],
                        'aaai': ['aaai'],
                        'fat': ['fat'],
                        'satml': ['satml'],
                        'aies': ['aies'],
                    }

                    def vtoken_matches(vtoken: str) -> bool:
                        # prefer explicit conf/journal path matches in id/url (most robust)
                        path_tokens = alias_map.get(vtoken, [vtoken])
                        for pt in path_tokens:
                            if key.startswith(f'conf/{pt}/'):
                                return True
                            if f'/conf/{pt}/' in rurl or f'/db/conf/{pt}/' in rurl:
                                return True
                            if f'/journals/{pt}/' in rurl or key.startswith(f'journals/{pt}/'):
                                return True

                        # fallback: allow matching against venue field only if user explicitly enables it
                        if allow_venue_match:
                            for pt in path_tokens:
                                if pt in rv:
                                    return True

                        return False

                    matched = False
                    for vtoken in venue_tokens:
                        if vtoken_matches(vtoken):
                            # found a match for the venue token
                            y = record.get('year') or 'unknown'
                            if year_tokens and y not in year_tokens:
                                continue
                            f = writer_for(vtoken, y)
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            counters[(vtoken, y)] += 1
                            total_written += 1
                            matched = True
                    # optional: if no venue tokens given, write to a catch-all per-year
                    if not venue_tokens and (not year_tokens or record.get('year') in year_tokens):
                        y = record.get('year') or 'unknown'
                        f = writer_for('all', y)
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        counters[('all', y)] += 1
                        total_written += 1

                    # clear element to free memory
                    elem.clear()

                # periodic flush based on number of records or time
                if processed % flush_interval == 0:
                    for f in writers.values():
                        f.flush()
                    last_flush = time.time()
        finally:
            # close everything
            del context
            wrapped.close()
            for f in writers.values():
                try:
                    f.flush(); f.close()
                except Exception:
                    pass

    # print summary
    print(f"Processed {processed} records, wrote {total_written} matching records to {out_dir}")
    for k,v in counters.items():
        print(f"{k}: {v}")


# Update main to support multi-target extraction
def main():
    parser = argparse.ArgumentParser(description="Stream-parse DBLP XML to JSONL with simple filters")
    parser.add_argument("--input", required=True, help="Path to dblp.xml (unzipped) file")
    parser.add_argument("--output", help="Path to output JSONL file (used for single-target mode)")
    parser.add_argument("--out-dir", help="Output directory for multi-target mode (writes <venue>_<year>.jsonl)")
    parser.add_argument("--type", help="Filter by record type (e.g., inproceedings, article)")
    parser.add_argument("--venue", help="Filter by venue substring (matches booktitle or journal) (single)")
    parser.add_argument("--venues", help="Comma-separated venue tokens for multi-target extraction")
    parser.add_argument("--year", help="Filter by year (exact match) (single)")
    parser.add_argument("--years", help="Comma-separated years for multi-target extraction, e.g. 2020,2021,2022")
    parser.add_argument("--limit", type=int, help="Maximum number of records to write (single-target mode)")
    parser.add_argument("--dtd", help="Path to dblp.dtd to resolve named entities (optional)")
    parser.add_argument("--flush-interval", type=int, default=1000, help="Flush interval in records")
    parser.add_argument("--no-progress", action='store_true', help="Disable progress bar even if tqdm is installed")
    parser.add_argument("--allow-venue", action='store_true', help="Allow matching on venue substring in addition to strict conf/id-path matching (use with caution)")

    args = parser.parse_args()

    # auto-detect dblp.dtd in current dir if not provided
    dtd_path = args.dtd or ("dblp.dtd" if os.path.exists("dblp.dtd") else None)
    entity_map = parse_dtd(dtd_path) if dtd_path else None

    # multi-target mode
    if args.venues or args.years or args.out_dir:
        venues = [v.strip() for v in args.venues.split(',')] if args.venues else None
        years = [y.strip() for y in args.years.split(',')] if args.years else None
        out_dir = args.out_dir or 'dblp_extracted'
        show_progress = not args.no_progress
        parse_dblp_multi(args.input, out_dir, args.type, venues, years, entity_map, flush_interval=args.flush_interval, show_progress=show_progress, allow_venue_match=args.allow_venue)
        return

    # single-target (backwards compatible)
    parse_dblp(args.input, args.output, args.type, args.venue, args.year, args.limit, entity_map)


if __name__ == "__main__":
    main()

"""
ACL Anthology BibTeX Parser V2
Lightweight parser that fixes LaTeX encoding issues without heavy dependencies.
"""

import re
import json
import csv
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Optional robust LaTeX to text conversion
try:
    from pylatexenc.latex2text import LatexNodes2Text  # type: ignore
    _HAS_PYLATEXENC = True
except Exception:
    _HAS_PYLATEXENC = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ACLPaper:
    """Data class for ACL Anthology paper information."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    venue: str
    year: int
    booktitle: str
    url: str
    doi: str
    pages: str
    month: str
    address: str
    publisher: str
    isbn: str
    source: str = "acl_anthology"
    collected_at: str = None
    safety_score: int = 0
    ethics_score: int = 0
    category: str = "neither"
    
    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now().isoformat()

class ACLAnthologyParserV2:
    """Lightweight parser for ACL Anthology bibtex file with LaTeX encoding fixes."""
    
    def __init__(self, 
                 bib_file_path: str,
                 safety_keywords: List[str],
                 ethics_keywords: List[str],
                 min_year: int = 2020,
                 max_year: int = 2025,
                 output_dir: str = "data",
                 use_pylatexenc: bool = False):
        """Initialize the ACL Anthology parser."""
        self.bib_file_path = bib_file_path
        self.safety_keywords = [k.lower() for k in safety_keywords]
        self.ethics_keywords = [k.lower() for k in ethics_keywords]
        self.min_year = min_year
        self.max_year = max_year
        self.output_dir = output_dir
        self.use_pylatexenc = use_pylatexenc
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'papers_processed': 0,
            'papers_collected': 0,
            'safety_papers': 0,
            'ethics_papers': 0,
        }
    
    def _clean_latex_encoding(self, text: str) -> str:
        """Clean LaTeX encoding from text with robust handling."""
        if not text:
            return ""
        
        # Fast path: if no LaTeX markers, just normalize spaces
        if ('\\' not in text) and ('{' not in text) and ('}' not in text):
            return re.sub(r'\s+', ' ', text).strip()
        
        # Prefer pylatexenc if available
        if self.use_pylatexenc and _HAS_PYLATEXENC and (('\\' in text) or ('{' in text) or ('}' in text)):
            try:
                return LatexNodes2Text().latex_to_text(text)
            except Exception:
                pass
        
        # Fallback: basic replacements
        latex_map = {
            r'\\"a': 'ä', r'\\"A': 'Ä',
            r'\\"e': 'ë', r'\\"E': 'Ë',
            r'\\"i': 'ï', r'\\"I': 'Ï',
            r'\\"o': 'ö', r'\\"O': 'Ö',
            r'\\"u': 'ü', r'\\"U': 'Ü',
            r"\\\'a": 'á', r"\\\'A": 'Á',
            r"\\\'e": 'é', r"\\\'E": 'É',
            r"\\\'i": 'í', r"\\\'I": 'Í',
            r"\\\'o": 'ó', r"\\\'O": 'Ó',
            r"\\\'u": 'ú', r"\\\'U": 'Ú',
            r'\\`a': 'à', r'\\`A': 'À',
            r'\\`e': 'è', r'\\`E': 'È',
            r'\\`i': 'ì', r'\\`I': 'Ì',
            r'\\`o': 'ò', r'\\`O': 'Ò',
            r'\\`u': 'ù', r'\\`U': 'Ù',
            r'\\\^a': 'â', r'\\\^A': 'Â',
            r'\\\^e': 'ê', r'\\\^E': 'Ê',
            r'\\\^i': 'î', r'\\\^I': 'Î',
            r'\\\^o': 'ô', r'\\\^O': 'Ô',
            r'\\\^u': 'û', r'\\\^U': 'Û',
            r'\\~n': 'ñ', r'\\~N': 'Ñ',
            r'\\c\{c\}': 'ç', r'\\c\{C\}': 'Ç',
            r'\\ss': 'ß',
            r'\\&': '&', r'\\%': '%', r'\\#': '#', r'\\$': '$', r'\\_': '_',
            r'\\textasciitilde': '~', r'\\textasciicircum': '^', r'\\textbackslash': '\\',
        }
        for latex, uni in latex_map.items():
            # Use literal replace for speed and to avoid regex escape errors
            text = text.replace(latex, uni)
        
        # Remove LaTeX commands like \emph{...} but keep content
        text = re.sub(r'\\[a-zA-Z]+\s*\{([^}]*)\}', r'\1', text)
        # Remove remaining braces used for capitalization enforcement
        text = text.replace('{', '').replace('}', '')
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_field_value_balanced(self, entry_text: str, field_name: str, clean: bool = True) -> str:
        """Extract field value handling balanced braces and quoted strings."""
        if not entry_text:
            return ""
        # Locate the field assignment
        m = re.search(rf'{field_name}\s*=\s*', entry_text, flags=re.IGNORECASE)
        if not m:
            return ""
        i = m.end()
        n = len(entry_text)
        # Skip whitespace
        while i < n and entry_text[i].isspace():
            i += 1
        if i >= n:
            return ""
        if entry_text[i] == '{':
            # Parse balanced braces
            i += 1
            start = i
            depth = 1
            while i < n and depth > 0:
                ch = entry_text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                elif ch == '\\':
                    # Skip escaped char to avoid mistaking for brace
                    i += 1
                i += 1
            end = i - 1  # position before the closing brace
            raw = entry_text[start:end]
        elif entry_text[i] == '"':
            # Parse quoted string
            i += 1
            start = i
            escaped = False
            while i < n:
                ch = entry_text[i]
                if ch == '"' and not escaped:
                    break
                escaped = (ch == '\\' and not escaped)
                if ch != '\\':
                    escaped = False
                i += 1
            end = i
            raw = entry_text[start:end]
        else:
            # Bare value until comma
            start = i
            while i < n and entry_text[i] not in ',\n\r':
                i += 1
            raw = entry_text[start:i]
        raw = raw.strip()
        if clean:
            return self._clean_latex_encoding(raw)
        # Light cleanup for non-critical fields: drop outer braces, collapse spaces
        raw = raw.replace('{', '').replace('}', '')
        return re.sub(r'\s+', ' ', raw).strip()
    
    def _extract_field_value(self, entry_text: str, field_name: str, clean: bool = True) -> str:
        """Wrapper to extract field value with balanced parsing."""
        return self._extract_field_value_balanced(entry_text, field_name, clean)
    
    def _parse_bib_entry(self, entry_text: str, entry_index: int = 0) -> Optional[Dict[str, str]]:
        """Parse a single bibtex entry into a dictionary."""
        try:
            # Extract entry type and key
            entry_match = re.match(r'@(\w+)\{([^,]+),', entry_text)
            if not entry_match:
                return None
            
            entry_type = entry_match.group(1)
            entry_key = entry_match.group(2)
            
            # Only process @inproceedings entries
            if entry_type != 'inproceedings':
                if entry_index < 20:
                    logger.info(f"Skipping entry {entry_index}: type={entry_type}, key={entry_key}")
                return None
            
            # Extract fields using improved regex
            fields = {}
            
            # Title
            fields['title'] = self._extract_field_value(entry_text, 'title', clean=True)
            
            # Abstract
            fields['abstract'] = self._extract_field_value(entry_text, 'abstract', clean=True)
            
            # Authors
            fields['authors'] = self._extract_field_value(entry_text, 'author', clean=True)
            
            # Booktitle
            fields['booktitle'] = self._extract_field_value(entry_text, 'booktitle', clean=True)
            
            # Year
            year_str = self._extract_field_value(entry_text, 'year', clean=False)
            try:
                fields['year'] = int(year_str) if year_str else 0
            except ValueError:
                fields['year'] = 0
            
            # URL
            fields['url'] = self._extract_field_value(entry_text, 'url', clean=False)
            
            # DOI
            fields['doi'] = self._extract_field_value(entry_text, 'doi', clean=False)
            
            # Pages
            fields['pages'] = self._extract_field_value(entry_text, 'pages', clean=False)
            
            # Month
            fields['month'] = self._extract_field_value(entry_text, 'month', clean=False)
            
            # Address
            fields['address'] = self._extract_field_value(entry_text, 'address', clean=False)
            
            # Publisher
            fields['publisher'] = self._extract_field_value(entry_text, 'publisher', clean=False)
            
            # ISBN
            fields['isbn'] = self._extract_field_value(entry_text, 'isbn', clean=False)
            
            # Add entry key and type
            fields['entry_key'] = entry_key
            fields['entry_type'] = entry_type
            
            return fields
            
        except Exception as e:
            logger.warning(f"Error parsing bibtex entry: {e}")
            return None
    
    def _calculate_relevance_score(self, paper_data: Dict[str, str]) -> Tuple[int, int, int, str]:
        """Calculate relevance score for a paper based on keywords."""
        title = (paper_data.get('title', '') or '').lower()
        abstract = (paper_data.get('abstract', '') or '').lower()
        text_to_search = f"{title} {abstract}"
        
        safety_score = sum(1 for keyword in self.safety_keywords if keyword in text_to_search)
        ethics_score = sum(1 for keyword in self.ethics_keywords if keyword in text_to_search)
        
        total_score = safety_score + ethics_score
        
        # Categorization based on which score is higher
        if safety_score > ethics_score:
            category = 'safety'
        elif ethics_score > safety_score:
            category = 'ethics'
        elif safety_score > 0 and ethics_score > 0 and safety_score == ethics_score:
            category = 'both'
        else:
            category = 'neither'
        
        return safety_score, ethics_score, total_score, category
    
    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list of author names."""
        if not author_string:
            return []
        # Normalize whitespace/newlines
        norm = re.sub(r'\s+', ' ', author_string).strip()
        # Split by 'and' token (case-insensitive, surrounded by whitespace)
        parts = re.split(r'\s+and\s+', norm, flags=re.IGNORECASE)
        authors = [p.strip() for p in parts if p.strip()]
        return authors
    
    def _extract_venue_from_booktitle(self, booktitle: str) -> str:
        """Extract venue name from booktitle."""
        if not booktitle:
            return "Unknown"
        
        # Common venue patterns
        venue_patterns = [
            r'ACL\s+(\d{4})',
            r'EACL\s+(\d{4})',
            r'NAACL\s+(\d{4})',
            r'EMNLP\s+(\d{4})',
            r'Findings of ACL\s+(\d{4})',
            r'CoNLL\s+(\d{4})',
            r'TACL',
            r'CL\s+(\d{4})',
            r'LREC\s+(\d{4})',
        ]
        
        for pattern in venue_patterns:
            match = re.search(pattern, booktitle, re.IGNORECASE)
            if match:
                return match.group(0)
        
        # Fallback: return first part of booktitle
        return booktitle.split(':')[0].strip()
    
    def _extract_paper_id_from_url(self, url: str) -> str:
        """Extract paper ID from ACL Anthology URL."""
        if not url:
            return ""
        
        # Extract ID from URL like https://aclanthology.org/2024.acl-long.1/
        match = re.search(r'aclanthology\.org/([^/]+)', url)
        if match:
            return match.group(1)
        
        return url
    
    def parse_bib_file(self, max_collect: Optional[int] = None) -> List[ACLPaper]:
        """Parse the entire bibtex file and return relevant papers."""
        logger.info(f"Starting to parse {self.bib_file_path}")
        
        collected_papers = []
        
        try:
            with open(self.bib_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into individual entries
            entries = re.split(r'\n@', content)
            
            # Add back the @ symbol to entries (except the first one)
            for i in range(1, len(entries)):
                entries[i] = '@' + entries[i]
            
            logger.info(f"Found {len(entries)} entries in bibtex file")
            logger.info(f"Using {len(self.safety_keywords)} safety keywords and {len(self.ethics_keywords)} ethics keywords")
            logger.info(f"Sample safety keywords: {self.safety_keywords[:5]}")
            logger.info(f"Sample ethics keywords: {self.ethics_keywords[:5]}")
            
            for i, entry_text in enumerate(entries):
                if i % 1000 == 0:
                    logger.info(f"Processing entry {i}/{len(entries)}")
                
                # Parse the entry
                entry_data = self._parse_bib_entry(entry_text, i)
                if not entry_data:
                    continue
                
                self.stats['papers_processed'] += 1
                
                # Check year filter
                year = entry_data.get('year', 0)
                if year < self.min_year or year > self.max_year:
                    continue
                
                # Calculate relevance scores
                safety_score, ethics_score, total_score, category = self._calculate_relevance_score(entry_data)
                
                # Debug: Print first few papers to see what we're getting
                if i < 20:
                    title = entry_data.get('title', '')[:50]
                    logger.info(f"Sample paper {i}: title='{title}...', safety={safety_score}, ethics={ethics_score}, category={category}")
                
                # Only collect relevant papers
                if total_score >= 0 or category in ['safety', 'ethics', 'both']:
                    # Parse authors
                    authors = self._parse_authors(entry_data.get('authors', ''))
                    
                    # Extract venue
                    venue = self._extract_venue_from_booktitle(entry_data.get('booktitle', ''))
                    
                    # Extract paper ID
                    paper_id = self._extract_paper_id_from_url(entry_data.get('url', ''))
                    
                    # Create paper object
                    paper = ACLPaper(
                        paper_id=paper_id,
                        title=entry_data.get('title', ''),
                        abstract=entry_data.get('abstract', ''),
                        authors=authors,
                        venue=venue,
                        year=year,
                        booktitle=entry_data.get('booktitle', ''),
                        url=entry_data.get('url', ''),
                        doi=entry_data.get('doi', ''),
                        pages=entry_data.get('pages', ''),
                        month=entry_data.get('month', ''),
                        address=entry_data.get('address', ''),
                        publisher=entry_data.get('publisher', ''),
                        isbn=entry_data.get('isbn', ''),
                        safety_score=safety_score,
                        ethics_score=ethics_score,
                        category=category
                    )
                    
                    collected_papers.append(paper)
                    self.stats['papers_collected'] += 1
                    
                    # Update category statistics
                    if category == 'safety':
                        self.stats['safety_papers'] += 1
                    elif category == 'ethics':
                        self.stats['ethics_papers'] += 1
                    
                    if len(collected_papers) % 100 == 0:
                        logger.info(f"Collected {len(collected_papers)} papers so far...")
                    if max_collect is not None and len(collected_papers) >= max_collect:
                        logger.info(f"Reached max_collect={max_collect}, stopping early.")
                        break
        
        except Exception as e:
            logger.error(f"Error parsing bibtex file: {e}")
            raise
        
        logger.info(f"Parsing complete! Collected {len(collected_papers)} relevant papers")
        return collected_papers
    
    def save_data(self, papers: List[ACLPaper]):
        """Save collected data in multiple formats."""
        # Save as JSONL
        jsonl_filename = os.path.join(self.output_dir, "acl_anthology_papers_v2_full.jsonl")
        with open(jsonl_filename, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(asdict(paper), ensure_ascii=False) + '\n')
        
        # Save as CSV
        #csv_filename = os.path.join(self.output_dir, "acl_anthology_papers_v2.csv")
        # if papers:
        #     with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        #         writer = csv.DictWriter(f, fieldnames=list(asdict(papers[0]).keys()))
        #         writer.writeheader()
        #         for paper in papers:
        #             writer.writerow(asdict(paper))
        
        # Save statistics
        stats_filename = os.path.join(self.output_dir, "acl_parsing_stats_v2_full.json")
        with open(stats_filename, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Data saved to {self.output_dir}")
    
    def run_parser(self):
        """Run the main parsing process."""
        logger.info("="*50)
        logger.info("Starting ACL Anthology BibTeX Parser V2")
        logger.info("="*50)
        
        # Parse the bibtex file
        papers = self.parse_bib_file()
        
        # Save the data
        self.save_data(papers)
        
        # Print summary
        logger.info("="*50)
        logger.info("Parsing completed!")
        logger.info(f"Total papers processed: {self.stats['papers_processed']}")
        logger.info(f"Total papers collected: {self.stats['papers_collected']}")
        logger.info(f"Safety papers: {self.stats['safety_papers']}")
        logger.info(f"Ethics papers: {self.stats['ethics_papers']}")
        logger.info("="*50)

def main():
    """Main execution function."""
    # Import keywords from the keywords module
    import sys
    from keywords import keywords
    
    # Initialize parser
    parser = ACLAnthologyParserV2(
        bib_file_path="anthology+abstracts.bib",
        safety_keywords=keywords["safety"],
        ethics_keywords=keywords["ethics"],
        min_year=2020,
        max_year=2025,
        output_dir="data"
    )
    
    try:
        parser.run_parser()
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise

if __name__ == "__main__":
    main()

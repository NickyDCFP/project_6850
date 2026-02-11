"""
DBLP collector for conference streams (NeurIPS/NIPS, ICLR, SaTML, etc.).
Pulls titles, authors, years, links from DBLP stream exports.
Attempts to enrich abstracts via Crossref (by DOI) or arXiv (by ee link).
Scores papers with safety/ethics keywords and saves JSONL/CSV.
"""

import os
import re
import sys
import csv
import json
import time
import logging
import asyncio
import aiohttp
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Local keywords
sys.path.append(os.path.dirname(__file__))
from keywords import keywords as KEYWORDS

# Optional HTML parsing
try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VenuePaper:
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    venue: str
    year: int
    url: str
    doi: str
    ee: str
    source: str = "dblp"
    collected_at: str = datetime.now().isoformat()
    safety_score: int = 0
    ethics_score: int = 0
    category: str = "neither"
    extras: Dict = field(default_factory=dict)


class DBLPScraper:
    """Scrape DBLP conference streams and enrich metadata."""

    def __init__(self,
                 venues: List[str],  # e.g., ['nips', 'iclr', 'satml']
                 min_year: int = 2020,
                 max_year: int = 2025,
                 output_dir: str = "data",
                 per_venue_limit: Optional[int] = None,
                 show_progress: bool = True,
                 debug: bool = False):
        self.venues = venues
        self.min_year = min_year
        self.max_year = max_year
        self.output_dir = output_dir
        self.per_venue_limit = per_venue_limit
        self.show_progress = show_progress and _HAS_TQDM
        self.debug = debug
        os.makedirs(output_dir, exist_ok=True)

        self.safety_keywords = [k.lower() for k in KEYWORDS["safety"]]
        self.ethics_keywords = [k.lower() for k in KEYWORDS["ethics"]]

        if self.debug:
            logger.setLevel(logging.DEBUG)

    # ------------------------ DBLP fetching ------------------------
    def _fetch_publ_api_page(self, venue: str, f: int = 0, h: int = 1000) -> Optional[Dict]:
        """Fetch one page from DBLP publ API with offset f and page size h.
        Docs: https://dblp.org/search/publ/api
        """
        q = f"stream:conf/{venue}:*"
        url = "https://dblp.org/search/publ/api"
        params = {"q": q, "h": str(h), "f": str(f), "format": "json"}
        try:
            r = requests.get(url, params=params, timeout=30, verify=False)  # Disable SSL verification temporarily
            if r.ok:
                return r.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
        return None

    def _fetch_stream_xml(self, venue: str) -> Optional[str]:
        urls = [
            f"https://dblp.org/db/conf/{venue}/index.xml",  # Updated base URL
            f"https://dblp.org/streams/conf/{venue}.xml",  # Fallback URL
            f"https://dblp.org/streams/conf/{venue}?format=xml"  # Another fallback
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=30, verify=False)
                if r.ok and 'xml' in r.headers.get('content-type', '').lower():
                    return r.text
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch XML from {url}: {e}")
        return None

    def _parse_publ_api(self, data: Dict, venue: str) -> List[Dict]:
        """Parse DBLP publ API JSON into entry dicts with a uniform schema."""
        entries: List[Dict] = []
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
        for hit in hits:
            info = hit.get('info', {})
            # authors can be dict or list in DBLP JSON
            authors = []
            auth = info.get('authors', {})
            if isinstance(auth, dict) and 'author' in auth:
                a = auth['author']
                if isinstance(a, list):
                    authors = [x.get('text', '') if isinstance(x, dict) else str(x) for x in a]
                elif isinstance(a, dict):
                    authors = [a.get('text', '')]
                else:
                    authors = [str(a)]

            entries.append({
                'title': info.get('title', '') or '',
                'year': info.get('year', '') or '',
                'author': authors,
                'ee': info.get('ee', '') or '',
                'doi': info.get('doi', '') or '',
                'url': info.get('url', '') or '',
                'key': info.get('key', '') or ''
            })
        logger.info(f"Parsed {len(entries)} entries from publ API for {venue}")
        return entries

    def _parse_xml_stream(self, xml_text: str, venue: str) -> List[Dict]:
        """Parse DBLP XML stream to list of inproceedings dicts."""
        entries: List[Dict] = []
        try:
            root = ET.fromstring(xml_text)
            for node in root.iter():
                if node.tag.endswith('inproceedings'):
                    entry: Dict = {}
                    for child in list(node):
                        tag = child.tag.split('}')[-1]
                        if tag == 'author':
                            entry.setdefault('author', []).append((child.text or '').strip())
                        else:
                            entry[tag] = (child.text or '').strip()
                    entries.append(entry)
        except ET.ParseError:
            logger.warning(f"Failed to parse XML for {venue}")
        logger.info(f"Parsed {len(entries)} inproceedings from XML for {venue}")
        return entries

    # ------------------------ Enrichment ------------------------
    def _get_arxiv_abstract(self, ee: str) -> str:
        try:
            m = re.search(r'arxiv\.org/(?:abs|pdf)/([^/?#]+)', ee)
            if not m:
                return ''
            arx_id = m.group(1).replace('.pdf', '')
            url = f"http://export.arxiv.org/api/query?search_query=id:{arx_id}"
            r = requests.get(url, timeout=20)
            if not r.ok:
                return ''
            # Very simple extraction
            m2 = re.search(r'<summary>([\s\S]*?)</summary>', r.text)
            if m2:
                summary = m2.group(1)
                summary = re.sub(r'\s+', ' ', summary)
                return summary.strip()
        except requests.RequestException:
            return ''
        return ''

    def _is_truncated_snippet(self, text: str) -> bool:
        """Heuristic: does this text look like a truncated meta/snippet (ends with an ellipsis)?"""
        if not text:
            return False
        t = text.strip()
        # common ASCII and unicode ellipsis
        return bool(re.search(r'(\.\.\.|…)$', t))

    def _get_crossref_abstract(self, doi: str) -> str:
        if not doi:
            return ''
        try:
            url = f"https://api.crossref.org/works/{doi}"
            r = requests.get(url, timeout=20, headers={'User-Agent': 'AI-Safety-Ethics-Study/1.0 (mailto:your-email@example.com)'})
            if not r.ok:
                return ''
            data = r.json().get('message', {})
            abstract = data.get('abstract', '') or ''
            # Crossref abstracts may be JATS XML – strip tags
            abstract = re.sub(r'<[^>]+>', ' ', abstract)
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            return abstract
        except requests.RequestException:
            return ''

    def _search_crossref_by_title(self, title: str, year: Optional[int] = None, first_author: str = '') -> Tuple[str, str]:
        """Search Crossref by title (and optionally year/author). Returns (doi, abstract)."""
        if not title:
            return '', ''
        try:
            url = "https://api.crossref.org/works"
            params = {
                'query.title': title,
                'rows': 3,
                'select': 'title,author,DOI,abstract,container-title,issued'
            }
            r = requests.get(url, params=params, timeout=20, headers={'User-Agent': 'AI-Safety-Ethics-Study/1.0 (mailto:your-email@example.com)'})
            if not r.ok:
                return '', ''
            items = r.json().get('message', {}).get('items', [])
            best_doi = ''
            best_abs = ''
            for it in items:
                it_title = (it.get('title', ['']) or [''])[0].lower()
                if title.lower().strip()[:30] not in it_title:
                    continue
                if year:
                    issued = it.get('issued', {}).get('date-parts', [[None]])[0][0]
                    if issued and abs(int(issued) - int(year)) > 1:
                        continue
                best_doi = it.get('DOI', '') or ''
                abs_raw = it.get('abstract', '') or ''
                abs_txt = re.sub(r'<[^>]+>', ' ', abs_raw)
                best_abs = re.sub(r'\s+', ' ', abs_txt).strip()
                break
            return best_doi, best_abs
        except requests.RequestException:
            return '', ''

    def _try_openreview_abstract(self, url: str) -> str:
        """Fetch abstract from OpenReview note if possible."""
        if not url or 'openreview.net' not in url:
            return ''
        try:
            # Extract id parameter
            m = re.search(r'[?&]id=([^&#]+)', url)
            note_id = m.group(1) if m else ''
            if not note_id:
                return ''
            api = f"https://api.openreview.net/notes?id={note_id}"
            r = requests.get(api, timeout=20)
            if not r.ok:
                return ''
            data = r.json()
            # Newer API wraps note in 'notes' list, older single 'note'
            notes = data.get('notes') or [data.get('note')] if data.get('note') else []
            for n in notes:
                content = n.get('content', {})
                abstract = content.get('abstract') or content.get('tl;dr') or ''
                if isinstance(abstract, dict):
                    abstract = abstract.get('value', '')
                abstract = str(abstract)
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                if abstract:
                    return abstract
            return ''
        except requests.RequestException:
            return ''

    def _try_html_abstract(self, url: str) -> str:
        """Best-effort HTML scrape for abstract from conference page (neurips.cc, proceedings.mlr.press, etc.)."""
        if not url:
            return ''
        try:
            r = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            if not r.ok:
                return ''
            html = r.text
            # Quick anti-bot / challenge detection: many landing pages may be Cloudflare/Incapsula
            # which render a 'Just a moment...' page or 'Checking your browser' response. Avoid
            # returning those pages as valid abstracts. Look for common markers in the <title>
            # or body text.
            title_m = re.search(r'<title[^>]*>([^<]+)</title>', html, flags=re.I)
            title_text = title_m.group(1).strip() if title_m else ''
            if title_text and ('just a moment' in title_text.lower() or 'checking your browser' in title_text.lower()):
                # treat as blocked
                return ''
            if re.search(r'checking your browser|just a moment|cf-browser-verification|incap_ses', html, flags=re.I):
                return ''
            if _HAS_BS4:
                soup = BeautifulSoup(html, 'html.parser')
                # Collect candidate texts from meta tags, JSON-LD, and page paragraphs.
                # We'll tag candidates with their source and prefer JSON-LD and paragraph content
                # over generic meta/og descriptions (many publishers put short snippets in meta).
                candidates = []  # list of tuples (text, source)
                # --- Inspect HTML comments for embedded JSON-LD or meta fragments ---
                try:
                    comments = soup.find_all(string=lambda text: isinstance(text, type(soup.Comment)))
                    for cm in comments:
                        c = str(cm).strip()
                        if not c:
                            continue
                        # If the comment contains JSON-LD, try to parse it and extract description/abstract
                        if 'application/ld+json' in c or '"@type"' in c:
                            try:
                                for js in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>', c, flags=re.I):
                                    try:
                                        obj = json.loads(js)
                                        if isinstance(obj, dict):
                                            desc = obj.get('description') or obj.get('abstract') or ''
                                            if desc and isinstance(desc, str):
                                                candidates.append((desc.strip(), 'comment_jsonld'))
                                        elif isinstance(obj, list):
                                            for entry in obj:
                                                if isinstance(entry, dict):
                                                    desc = entry.get('description') or entry.get('abstract') or ''
                                                    if desc and isinstance(desc, str):
                                                        candidates.append((desc.strip(), 'comment_jsonld'))
                                    except Exception:
                                        continue
                            except Exception:
                                pass
                        # Look for common meta tag fragments inside comments: e.g., <meta name="description" content="...">
                        if 'meta' in c.lower() and 'description' in c.lower():
                            m = re.search(r'content=["\']([^"\']{40,})["\']', c)
                            if m:
                                candidates.append((m.group(1).strip(), 'comment_meta'))
                        # Generic heuristic: long comment text might contain an abstract copy
                        if len(c) > 200 and len(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', c)).strip()) > 40:
                            cleaned = re.sub(r'<[^>]+>', ' ', c)
                            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                            candidates.append((cleaned, 'comment'))
                except Exception:
                    pass
                # JSON-LD: look for description/abstract fields (preferred)
                for js in soup.find_all('script', attrs={'type': 'application/ld+json'}):
                    try:
                        obj = json.loads(js.string or '{}')
                        if isinstance(obj, dict):
                            desc = obj.get('description') or obj.get('abstract') or ''
                            if desc and isinstance(desc, str):
                                candidates.append((desc.strip(), 'jsonld'))
                        elif isinstance(obj, list):
                            for entry in obj:
                                if isinstance(entry, dict):
                                    desc = entry.get('description') or entry.get('abstract') or ''
                                    if desc and isinstance(desc, str):
                                        candidates.append((desc.strip(), 'jsonld'))
                    except Exception:
                        continue

                # citation_abstract is common for some publishers (keep but lower priority than paragraphs/jsonld)
                m = soup.find('meta', attrs={'name': 'citation_abstract'})
                if m and m.get('content'):
                    candidates.append((m.get('content').strip(), 'citation_abstract'))

                # Standard meta description / OpenGraph
                m_og = soup.find('meta', attrs={'property': 'og:description'}) or soup.find('meta', attrs={'name': 'og:description'})
                if m_og and m_og.get('content'):
                    candidates.append((m_og.get('content').strip(), 'og'))
                m_meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'name': 'dc.description'}) or soup.find('meta', attrs={'name': 'DC.description'})
                if m_meta and m_meta.get('content'):
                    candidates.append((m_meta.get('content').strip(), 'meta'))

                # Also consider the main visible paragraphs on the page as candidates (preferred)
                try:
                    paras = [(p.get_text(' ', strip=True), 'paragraph') for p in soup.find_all('p') if p.get_text(strip=True)]
                    candidates.extend(paras)
                except Exception:
                    pass

                # For publisher-specific patterns (e.g., proceedings.mlr.press) try to find an explicit abstract element
                host = (url or '').lower()
                if 'proceedings.mlr.press' in host:
                    # MLR pages usually contain the abstract in a div with class/id containing 'abstract' or within the main content area
                    el = soup.find(attrs={'class': lambda x: x and 'abstract' in x.lower()}) or soup.find(attrs={'id': lambda x: x and 'abstract' in x.lower()})
                    if el:
                        txt = el.get_text(' ', strip=True)
                        if txt:
                            candidates.insert(0, (txt, 'mlr_abstract'))

                # Normalize whitespace and choose according to priority: comment_jsonld > jsonld > mlr_abstract > paragraph > citation_abstract > og > meta
                norm_cands = [(re.sub(r'\s+', ' ', c[0]).strip(), c[1]) for c in candidates if isinstance(c[0], str) and c[0].strip()]
                if norm_cands:
                    priority = ['comment_jsonld', 'jsonld', 'mlr_abstract', 'paragraph', 'comment_meta', 'comment', 'citation_abstract', 'og', 'meta']
                    for src in priority:
                        src_cands = [t for t, s in norm_cands if s == src]
                        if src_cands:
                            best = max(src_cands, key=lambda x: len(x))
                            if len(best) > 40:
                                return best
                    # fallback: longest candidate of any source
                    best_any = max((t for t, s in norm_cands), key=lambda x: len(x))
                    if len(best_any) > 40:
                        return best_any
                # ACM DL: https://dl.acm.org
                if 'dl.acm.org' in url:
                    # Try meta tags first
                    meta = soup.find('meta', attrs={'name': 'citation_abstract'})
                    if meta and meta.get('content'):
                        return meta['content'].strip()
                    meta = soup.find('meta', attrs={'name': 'dc.Description'})
                    if meta and meta.get('content'):
                        return meta['content'].strip()
                    # Fallback to div/section with abstract
                    for sel in ['.abstractSection', 'div.abstract', 'section.abstract']:
                        el = soup.select_one(sel)
                        if el:
                            txt = el.get_text(' ', strip=True)
                            if len(txt) > 40:
                                return txt
                # AAAI/aaai.org
                if 'aaai.org' in url:
                    # Look for heading 'Abstract' then next paragraphs
                    h_candidates = soup.find_all(['h3','h4','h2'])
                    for h in h_candidates:
                        if h.get_text(strip=True).lower() == 'abstract':
                            texts = []
                            for sib in h.next_siblings:
                                if getattr(sib, 'name', None) is None:
                                    continue
                                if sib.name in ('h2','h3','h4'):
                                    break
                                if sib.name == 'p':
                                    t = sib.get_text(' ', strip=True)
                                    if t:
                                        texts.append(t)
                                else:
                                    ps = sib.find_all('p') if hasattr(sib, 'find_all') else []
                                    for p in ps:
                                        t = p.get_text(' ', strip=True)
                                        if t:
                                            texts.append(t)
                            abstract = ' '.join(texts).strip()
                            if len(abstract) > 40:
                                return abstract
                    # Generic class/id match
                    for attr in ['class','id']:
                        el = soup.find(attrs={attr: re.compile('abstract', re.I)})
                        if el:
                            txt = el.get_text(' ', strip=True)
                            if len(txt) > 40:
                                return txt
                # NeurIPS specific: pages.nips.cc / neurips.cc have an h4 'Abstract' then paragraph(s)
                if 'neurips.cc' in url or 'nips.cc' in url:
                    # Remove MathJax preview noise
                    for mj in soup.select('.MathJax_Preview, script'):  # strip mathjax script/preview
                        mj.decompose()
                    # Find the Abstract header
                    h4s = soup.find_all('h4')
                    for h in h4s:
                        if h.get_text(strip=True).lower() == 'abstract':
                            # Collect following p siblings until next h4
                            texts = []
                            for sib in h.next_siblings:
                                # Skip whitespace
                                if getattr(sib, 'name', None) is None:
                                    continue
                                if sib.name == 'h4':
                                    break
                                # Dive into container
                                ps = sib.find_all('p') if hasattr(sib, 'find_all') else []
                                if ps:
                                    for p in ps:
                                        t = p.get_text(' ', strip=True)
                                        if t:
                                            texts.append(t)
                                elif sib.name == 'p':
                                    t = sib.get_text(' ', strip=True)
                                    if t:
                                        texts.append(t)
                            abstract = ' '.join(texts).strip()
                            if len(abstract) > 40:
                                return abstract
                    # Fallback: grab the first long paragraph on the page
                    cand_p = max((p.get_text(' ', strip=True) for p in soup.find_all('p')), key=lambda x: len(x), default='')
                    if len(cand_p) > 80:
                        return cand_p
                # Common patterns
                # NeurIPS often has div with class containing 'abstract'
                cand = soup.find(lambda tag: tag.name in ['div','section','p'] and tag.get_text(strip=True) and 'abstract' in tag.get_text(strip=True).lower())
                if cand and cand.name != 'p':
                    # look for following paragraph
                    p = cand.find_next('p')
                    if p:
                        text = p.get_text(' ', strip=True)
                        return text
                # Fallback: any element with class/id containing 'abstract'
                for attr in ['class', 'id']:
                    el = soup.find(attrs={attr: re.compile('abstract', re.I)})
                    if el:
                        text = el.get_text(' ', strip=True)
                        if len(text) > 30:
                            return text
            # Regex fallback: find 'Abstract' header then capture next paragraph
            m = re.search(r'Abstract\s*[:\n]\s*([^<\n]{40,800})', html, flags=re.I)
            if m:
                return re.sub(r'\s+', ' ', m.group(1)).strip()
            return ''
        except requests.RequestException:
            return ''

    def _get_semanticscholar_abstract(self, doi: str) -> str:
        """Query Semantic Scholar Graph API for an abstract using DOI.

        Uses SEMANTIC_SCHOLAR_API_KEY environment variable if available. If no
        key is present the public endpoint is used (rate-limited). Returns an
        empty string on any failure.
        """
        if not doi:
            return ''
        api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
        # Semantic Scholar accepts DOI with /works/ prefix when using the graph v1 API
        base = 'https://api.semanticscholar.org/graph/v1/paper/'
        # encode DOI for URL path
        doi_esc = requests.utils.requote_uri(doi)
        url = f"{base}{doi_esc}?fields=abstract"
        headers = {'User-Agent': 'AI-Safety-Ethics-Study/1.0'}
        if api_key:
            headers['x-api-key'] = api_key
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if not r.ok:
                return ''
            data = r.json()
            abstract = data.get('abstract', '') or ''
            if isinstance(abstract, str):
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                return abstract
        except requests.RequestException:
            return ''
        except ValueError:
            return ''
        return ''

    def _call_semanticscholar(self, doi: str) -> Tuple[str, Optional[Dict]]:
        """Return (abstract, raw_json_or_None) from Semantic Scholar for DOI."""
        if not doi:
            return '', None
        api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
        base = 'https://api.semanticscholar.org/graph/v1/paper/'
        doi_esc = requests.utils.requote_uri(doi)
        url = f"{base}{doi_esc}?fields=abstract,title,doi,year"
        headers = {'User-Agent': 'AI-Safety-Ethics-Study/1.0'}
        if api_key:
            headers['x-api-key'] = api_key
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if not r.ok:
                return '', None
            data = r.json()
            abstract = data.get('abstract', '') or ''
            if isinstance(abstract, str):
                abstract = re.sub(r'\s+', ' ', abstract).strip()
            return abstract, data
        except Exception:
            return '', None

    def _call_openalex(self, doi: str) -> Tuple[str, Optional[Dict]]:
        """Query OpenAlex works by DOI and return (abstract, raw_json)."""
        if not doi:
            return '', None
        url = f"https://api.openalex.org/works/doi:{requests.utils.requote_uri(doi)}"
        try:
            r = requests.get(url, timeout=20, headers={'User-Agent': 'AI-Safety-Ethics-Study/1.0'})
            if not r.ok:
                return '', None
            data = r.json()
            # OpenAlex may provide abstract_inverted_index
            abstract = ''
            if data.get('abstract_inverted_index'):
                inv = data.get('abstract_inverted_index') or {}
                # reconstruct: sort tokens by index order
                tokens = [''] * (max((max(idxs) for idxs in inv.values()), default=-1) + 1)
                for tok, idxs in inv.items():
                    for i in idxs:
                        if 0 <= i < len(tokens):
                            tokens[i] = tok
                abstract = ' '.join(t for t in tokens if t)
                abstract = re.sub(r'\s+', ' ', abstract).strip()
            else:
                abstract = data.get('abstract', '') or ''
                if isinstance(abstract, str):
                    abstract = re.sub(r'\s+', ' ', abstract).strip()
            return abstract, data
        except Exception:
            return '', None

    def _call_crossref(self, doi: str) -> Tuple[str, Optional[Dict]]:
        """Return (abstract, raw_message) from Crossref works endpoint."""
        if not doi:
            return '', None
        try:
            url = f"https://api.crossref.org/works/{doi}"
            r = requests.get(url, timeout=20, headers={'User-Agent': 'AI-Safety-Ethics-Study/1.0 (mailto:your-email@example.com)'})
            if not r.ok:
                return '', None
            msg = r.json().get('message', {})
            abs_raw = msg.get('abstract', '') or ''
            abs_txt = re.sub(r'<[^>]+>', ' ', abs_raw)
            abs_txt = re.sub(r'\s+', ' ', abs_txt).strip()
            return abs_txt, msg
        except Exception:
            return '', None

    def _search_semanticscholar_by_title(self, title: str) -> Tuple[str, Optional[Dict]]:
        """Search Semantic Scholar by title and return (abstract, raw_json) for first hit."""
        if not title:
            return '', None
        base = 'https://api.semanticscholar.org/graph/v1/paper/search'
        params = {'query': title, 'limit': 1, 'fields': 'abstract,doi,title,year'}
        headers = {'User-Agent': 'AI-Safety-Ethics-Study/1.0'}
        api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
        if api_key:
            headers['x-api-key'] = api_key
        try:
            r = requests.get(base, params=params, headers=headers, timeout=20)
            if not r.ok:
                return '', None
            data = r.json()
            # v1 search returns 'data' list
            items = data.get('data') or data.get('results') or []
            if not items:
                return '', data
            it = items[0]
            abstract = it.get('abstract', '') or ''
            if isinstance(abstract, str):
                abstract = re.sub(r'\s+', ' ', abstract).strip()
            return abstract, it
        except Exception:
            return '', None

    def _search_openalex_by_title(self, title: str) -> Tuple[str, Optional[Dict]]:
        """Search OpenAlex works by title (simple search) and return (abstract, raw_json)."""
        if not title:
            return '', None
        url = f"https://api.openalex.org/works?search={requests.utils.requote_uri(title)}&per-page=1"
        try:
            r = requests.get(url, timeout=20, headers={'User-Agent': 'AI-Safety-Ethics-Study/1.0'})
            if not r.ok:
                return '', None
            data = r.json()
            items = data.get('results') or []
            if not items:
                return '', data
            it = items[0]
            # Reconstruct abstract similar to _call_openalex
            abstract = ''
            if it.get('abstract_inverted_index'):
                inv = it.get('abstract_inverted_index') or {}
                tokens = [''] * (max((max(idxs) for idxs in inv.values()), default=-1) + 1)
                for tok, idxs in inv.items():
                    for i in idxs:
                        if 0 <= i < len(tokens):
                            tokens[i] = tok
                abstract = ' '.join(t for t in tokens if t)
                abstract = re.sub(r'\s+', ' ', abstract).strip()
            else:
                abstract = it.get('abstract', '') or ''
                if isinstance(abstract, str):
                    abstract = re.sub(r'\s+', ' ', abstract).strip()
            return abstract, it
        except Exception:
            return '', None

    # ------------------------ Async enrichment methods ------------------------
    async def _async_call_openalex(self, session: aiohttp.ClientSession, doi: str) -> Tuple[str, Optional[Dict]]:
        """Async version of OpenAlex DOI lookup."""
        if not doi:
            return '', None
        url = f"https://api.openalex.org/works/doi:{requests.utils.requote_uri(doi)}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    return '', None
                data = await response.json()
                
                abstract = ''
                if data.get('abstract_inverted_index'):
                    inv = data.get('abstract_inverted_index') or {}
                    tokens = [''] * (max((max(idxs) for idxs in inv.values()), default=-1) + 1)
                    for tok, idxs in inv.items():
                        for i in idxs:
                            if 0 <= i < len(tokens):
                                tokens[i] = tok
                    abstract = ' '.join(t for t in tokens if t)
                    abstract = re.sub(r'\s+', ' ', abstract).strip()
                else:
                    abstract = data.get('abstract', '') or ''
                    if isinstance(abstract, str):
                        abstract = re.sub(r'\s+', ' ', abstract).strip()
                return abstract, data
        except Exception:
            return '', None

    async def _async_search_openalex_by_title(self, session: aiohttp.ClientSession, title: str) -> Tuple[str, Optional[Dict]]:
        """Async version of OpenAlex title search."""
        if not title:
            return '', None
        url = f"https://api.openalex.org/works?search={requests.utils.requote_uri(title)}&per-page=1"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    return '', None
                data = await response.json()
                items = data.get('results') or []
                if not items:
                    return '', data
                it = items[0]
                
                abstract = ''
                if it.get('abstract_inverted_index'):
                    inv = it.get('abstract_inverted_index') or {}
                    tokens = [''] * (max((max(idxs) for idxs in inv.values()), default=-1) + 1)
                    for tok, idxs in inv.items():
                        for i in idxs:
                            if 0 <= i < len(tokens):
                                tokens[i] = tok
                    abstract = ' '.join(t for t in tokens if t)
                    abstract = re.sub(r'\s+', ' ', abstract).strip()
                else:
                    abstract = it.get('abstract', '') or ''
                    if isinstance(abstract, str):
                        abstract = re.sub(r'\s+', ' ', abstract).strip()
                return abstract, it
        except Exception:
            return '', None

    async def _async_enrich_paper(self, session: aiohttp.ClientSession, paper: VenuePaper) -> VenuePaper:
        """Async enrichment for a single paper using OpenAlex and other async sources."""
        paper.extras = paper.extras or {}
        
        # Skip if already has good abstract
        if paper.abstract and len(paper.abstract) > 40:
            return paper
        
        # Try OpenAlex first by DOI
        if paper.doi:
            abs_text, raw_data = await self._async_call_openalex(session, paper.doi)
            if abs_text and len(abs_text) > 40:
                paper.abstract = abs_text
                paper.extras['chosen_source'] = 'openalex_async'
                if raw_data:
                    paper.extras['openalex'] = raw_data
                    # Extract DOI if available
                    if isinstance(raw_data, dict):
                        oa_doi = raw_data.get('doi') or raw_data.get('ids', {}).get('doi') if isinstance(raw_data.get('ids', {}), dict) else None
                        if oa_doi:
                            paper.doi = paper.doi or oa_doi
                            paper.extras.setdefault('discovered', {})
                            paper.extras['discovered']['openalex_doi'] = oa_doi
                        # Update title if OpenAlex has a longer/more complete title
                        oa_title = raw_data.get('title') or raw_data.get('display_name')
                        if oa_title and isinstance(oa_title, str) and len(oa_title.strip()) > len(paper.title.strip()):
                            paper.title = oa_title.strip()
                            paper.extras.setdefault('title_updated_from', 'openalex_doi')
                return paper
        
        # Try OpenAlex title search
        abs_text, raw_data = await self._async_search_openalex_by_title(session, paper.title)
        if abs_text and len(abs_text) > 40:
            paper.abstract = abs_text
            paper.extras['chosen_source'] = 'openalex_title_async'
            if raw_data:
                paper.extras['openalex_title_search'] = raw_data
                # Extract DOI if available
                if isinstance(raw_data, dict):
                    oa_doi = raw_data.get('doi') or raw_data.get('ids', {}).get('doi') if isinstance(raw_data.get('ids', {}), dict) else None
                    if oa_doi:
                        paper.doi = paper.doi or oa_doi
                        paper.extras.setdefault('discovered', {})
                        paper.extras['discovered']['openalex_title_doi'] = oa_doi
                    # Update title if OpenAlex has a longer/more complete title
                    oa_title = raw_data.get('title') or raw_data.get('display_name')
                    if oa_title and isinstance(oa_title, str) and len(oa_title.strip()) > len(paper.title.strip()):
                        paper.title = oa_title.strip()
                        paper.extras.setdefault('title_updated_from', 'openalex_title_search')
            return paper
        
        # If we get here, async OpenAlex didn't work, paper will be handled by sync fallbacks
        return paper

    def _extract_doi_from_ee(self, url: str) -> str:
        """Try to extract a DOI from an `ee`/proceedings URL via meta tags, JSON-LD, or doi.org patterns."""
        if not url:
            return ''
        try:
            r = requests.get(url, timeout=10, headers={'User-Agent': 'AI-Safety-Ethics-Study/1.0'})
            if not r.ok:
                return ''
            html = r.text
            # meta tag citation_doi
            m = re.search(r'<meta[^>]+name=["\']citation_doi["\'][^>]*content=["\']([^"\']+)["\']', html, flags=re.I)
            if m:
                doi = m.group(1).strip()
                if re.match(r'^10\.\d{4,9}/\S+$', doi):
                    return doi
            # JSON-LD: look for "doi" in script type application/ld+json
            for js in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>', html, flags=re.I):
                try:
                    obj = json.loads(js)
                    if isinstance(obj, dict):
                        doi = obj.get('doi') or obj.get('identifier') or ''
                        if isinstance(doi, str) and re.match(r'^10\.\d{4,9}/\S+$', doi):
                            return doi
                except Exception:
                    continue
            # Search for doi.org/ pattern
            m2 = re.search(r'doi\.org/(10\.\d{4,9}/[^\s"\'<>)]+)', html)
            if m2:
                doi = m2.group(1).rstrip('.,;')
                if re.match(r'^10\.\d{4,9}/\S+$', doi):
                    return doi
            # Fallback: any 10.x/ pattern
            m3 = re.search(r'(10\.\d{4,9}/[^\s"\'<>)]+)', html)
            if m3:
                doi = m3.group(1).rstrip('.,;')
                if re.match(r'^10\.\d{4,9}/\S+$', doi):
                    return doi
        except requests.RequestException:
            return ''
        return ''

    # ------------------------ Scoring ------------------------
    def _score_paper(self, title: str, abstract: str) -> Tuple[int, int, str]:
        text = f"{title} {abstract}".lower()
        s = sum(1 for k in self.safety_keywords if k in text)
        e = sum(1 for k in self.ethics_keywords if k in text)
        if s > e:
            cat = 'safety'
        elif e > s:
            cat = 'ethics'
        elif s > 0 and e > 0:
            cat = 'both'
        else:
            cat = 'neither'
        return s, e, cat

    # ------------------------ Main ------------------------
    def _log_diagnostics(self, paper: VenuePaper):
        """Log diagnostics for a paper, including title and abstract issues."""
        if len(paper.title) < 20 or paper.title.endswith("..."):
            logger.warning(f"Truncated title detected: {paper.title} (DOI: {paper.doi})")
        if not paper.abstract or len(paper.abstract) < 40:
            logger.warning(f"Missing or short abstract: {paper.title} (DOI: {paper.doi})")

    def collect_for_venue(self, venue: str, existing_keys: Optional[set] = None, include_rejected: bool = False, filter_mode: str = 'abstract'):
        logger.info(f"Collecting DBLP entries for {venue} ({self.min_year}-{self.max_year})")
        collected: List[VenuePaper] = []

        entries: List[Dict] = []
        # Prefer per-year paging to be sure we collect as many entries as possible
        page_size = 1000
        for y in range(self.min_year, self.max_year + 1):
            offset_y = 0
            page_y = 0
            logger.info(f"{venue}: paging publ API for year {y}")
            while True:
                params = {"q": f"stream:conf/{venue}:* AND year:{y}", "h": str(page_size), "f": str(offset_y), "format": "json"}
                try:
                    r = requests.get("https://dblp.org/search/publ/api", params=params, timeout=30)
                    if not r.ok:
                        break
                    data = r.json()
                except requests.RequestException:
                    break

                page_hits = self._parse_publ_api(data, venue)
                # extra safety: filter by year field
                page_hits = [h for h in page_hits if str(h.get('year') or '') == str(y)]
                for h in page_hits:
                    entries.append(h)

                logger.info(f"{venue}: year {y} page {page_y+1} -> {len(page_hits)} hits")
                if not page_hits or len(page_hits) < page_size:
                    break
                offset_y += page_size
                page_y += 1
                time.sleep(0.15)

        # If we didn't get anything via the JSON API at all, fall back to the XML stream
        if not entries:
            xml_text = self._fetch_stream_xml(venue)
            if xml_text:
                entries = self._parse_xml_stream(xml_text, venue)

        count_added = 0
        pbar = tqdm(total=len(entries), desc=f"{venue} {self.min_year}-{self.max_year}", unit="paper") if self.show_progress else None
        for ent in entries:
            # Extract fields robustly from possible JSON or XML dicts
            title = (ent.get('title') or '').strip()
            year_str = str(ent.get('year') or '').strip()
            year = int(year_str) if year_str.isdigit() else 0
            # Skip odd proceedings-level or index entries (e.g., 'Conference on ...' or year-only), and placeholders like 'X18'
            tlow = title.lower() if title else ''
            if tlow.startswith('conference on') or tlow.startswith('proceedings') or tlow.strip() == str(year):
                if self.debug:
                    logger.debug(f"Skipping proceedings/index entry: {title} ({ent.get('key')})")
                if pbar: pbar.update(1)
                continue

            # Skip placeholder / malformed titles (very short or patterns like 'X18')
            if len(title.strip()) <= 2 or re.match(r'^x\d{2}$', tlow):
                if self.debug:
                    logger.debug(f"Skipping placeholder/short title: {title} ({ent.get('key')})")
                if pbar: pbar.update(1)
                continue

            if not title or not (self.min_year <= year <= self.max_year):
                if pbar: pbar.update(1)
                continue

            # Early duplicate check: if DBLP provides a stable key we can skip enrichment/network calls
            early_pid = ent.get('key') or ent.get('@key') or None
            if existing_keys is not None and early_pid is not None and str(early_pid) in existing_keys:
                # skip without doing heavy enrichment or scoring
                if pbar: pbar.update(1)
                continue

            # Authors
            authors = ent.get('author') or ent.get('authors') or []
            if isinstance(authors, str):
                authors_list = [a.strip() for a in re.split(r'\s+and\s+', authors)]
            else:
                authors_list = [str(a).strip() for a in authors if str(a).strip()]

            # Links (prefer EE for scraping; URL often points to DBLP record)
            ee = ent.get('ee') or ''
            url = ee or (ent.get('url') or '')
            doi = ent.get('doi') or ''


            pid = ent.get('key') or ent.get('@key') or f"{venue}:{year}:{len(collected)}"
            paper = VenuePaper(
                paper_id=str(pid),
                title=title,
                abstract=ent.get('abstract', ''),
                authors=authors_list,
                venue=venue.upper(),
                year=year,
                url=url,
                doi=doi,
                ee=ee,
            )

            # Log diagnostics for the paper
            self._log_diagnostics(paper)

            # carry over any provenance/extras discovered during initial enrichment
            try:
                paper.extras = ent.get('extras', {}) or {}
            except Exception:
                paper.extras = {}

            collected.append(paper)
            count_added += 1
            if self.per_venue_limit is not None and count_added >= self.per_venue_limit:
                if pbar: pbar.update(1)
                break
            if pbar: pbar.update(1)

        if pbar: pbar.close()
        logger.info(f"Initial collect complete: {len(collected)} papers for {venue}")
        num_w_abstract = 0
        for paper in collected:
            if paper.abstract and len(paper.abstract) > 40:
                num_w_abstract += 1
        logger.info(f"{num_w_abstract}/{len(collected)} papers have abstracts before initial enrichment")

        # ------------------ Async post-enrichment pass ------------------
        logger.info(f"Starting async post-enrichment pass for {venue}: checking missing abstracts")
        
        async def enrich_papers():
            # Create async session for enrichment
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Create tasks for papers that need enrichment
                tasks = []
                papers_to_enrich = []
                
                for p in collected:
                    if p.abstract and len(p.abstract) > 40:
                        continue  # Already has good abstract
                    
                    papers_to_enrich.append(p)
                    task = asyncio.create_task(self._async_enrich_paper(session, p))
                    tasks.append(task)
                
                if tasks:
                    logger.info(f"Enriching {len(tasks)} papers asynchronously...")
                    enriched_papers = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Update the collected papers with enriched data
                    for i, result in enumerate(enriched_papers):
                        if isinstance(result, Exception):
                            logger.warning(f"Error enriching paper {papers_to_enrich[i].paper_id}: {result}")
                        else:
                            # Paper is already updated in place
                            pass
                
                # Now do sync fallbacks for any that still don't have abstracts
                for p in collected:
                    if p.abstract and len(p.abstract) > 40:
                        continue
                    
                    # Try OpenReview
                    or_abs = self._try_openreview_abstract(p.ee or p.url)
                    if or_abs:
                        p.abstract = or_abs
                        p.extras = p.extras or {}
                        p.extras.setdefault('openreview', {'note': 'fetched via openreview'})
                        continue

                    # arXiv
                    if 'arxiv.org' in (p.ee or ''):
                        ar_abs = self._get_arxiv_abstract(p.ee)
                        if ar_abs:
                            p.abstract = ar_abs
                            p.extras = p.extras or {}
                            p.extras.setdefault('arxiv', {'note': 'fetched via arxiv API'})
                            continue

                    # HTML scrape
                    html_abs = self._try_html_abstract(p.ee or p.url)
                    if html_abs and len(html_abs) >= 80:
                        if self._is_truncated_snippet(html_abs):
                            p.extras = p.extras or {}
                            p.extras.setdefault('html_scrape_rejected', {'reason': 'truncated_snippet', 'sample': html_abs[:300]})
                        else:
                            p.abstract = html_abs
                            p.extras = p.extras or {}
                            p.extras.setdefault('html_scrape', {'source': p.ee or p.url, 'sample': html_abs[:300]})
                    
                    # Crossref title search as final fallback
                    if not p.abstract:
                        cross_doi, cross_abs = self._search_crossref_by_title(p.title, p.year, p.authors[0] if p.authors else '')
                        if cross_doi:
                            p.doi = p.doi or cross_doi
                        if cross_abs and len(cross_abs) > 40:
                            p.abstract = cross_abs
                            p.extras = p.extras or {}
                            p.extras.setdefault('crossref_title_search', {'doi': cross_doi})

        # Run the async enrichment
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(enrich_papers())
        finally:
            loop.close()

        # Re-score and filter to only keep papers with any ethics or safety score
        final_papers: List[VenuePaper] = []
        rejected_papers: List[VenuePaper] = []
        for p in collected:
            # Only score when we have a substantive abstract; otherwise mark as no_abstract
            if p.abstract and len(p.abstract or '') > 40:
                s_score, e_score, cat = self._score_paper('', p.abstract or '')
            else:
                s_score, e_score, cat = 0, 0, 'no_abstract'
            p.safety_score = s_score
            p.ethics_score = e_score
            p.category = cat

            # Filtering modes
            keep = False
            if filter_mode == 'score':
                keep = (s_score > 0 or e_score > 0)
            elif filter_mode == 'abstract':
                keep = bool(p.abstract and len(p.abstract) > 40)
            elif filter_mode == 'either':
                keep = (s_score > 0 or e_score > 0) or (p.abstract and len(p.abstract) > 40)
            else:
                # unknown mode: default to score
                keep = (s_score > 0 or e_score > 0)

            if keep:
                final_papers.append(p)
            else:
                rejected_papers.append(p)

        logger.info(f"Post-enrichment complete: {len(final_papers)} papers kept for {venue} after scoring filter")
        if include_rejected:
            return {'kept': final_papers, 'rejected': rejected_papers}
        return final_papers

    async def _fetch_publ_api_page_async(self, session: aiohttp.ClientSession, venue: str, year: int, offset: int = 0, page_size: int = 1000):
        """Async fetch one page from DBLP publ API with offset and page size."""
        q = f"stream:conf/{venue}:{year}"
        url = "https://dblp.org/search/publ/api"
        params = {"q": q, "h": str(page_size), "f": str(offset), "format": "json"}
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to fetch {url} with status {response.status}")
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
        return None

    async def collect_for_venue_async(self, venue: str, year: int):
        logger.info(f"Collecting DBLP entries for {venue} in {year}")
        collected = []
        offset = 0
        page_size = 1000

        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.sleep(5)  # Respectful delay
                data = await self._fetch_publ_api_page_async(session, venue, year, offset, page_size)
                if not data or not data.get('result', {}).get('hits', {}).get('hit', []):
                    break

                hits = data['result']['hits']['hit']
                for hit in hits:
                    abstract = hit.get('info', {}).get('abstract', '')
                    if len(abstract) > 12:
                        collected.append({
                            "title": hit.get('info', {}).get('title', ''),
                            "abstract": abstract,
                            "authors": hit.get('info', {}).get('authors', {}).get('author', []),
                            "year": year,
                            "venue": venue,
                            "url": hit.get('info', {}).get('url', ''),
                        })

                offset += page_size

        logger.info(f"Collected {len(collected)} papers for {venue} in {year}")
        return collected

    async def run_async(self, venues: list, years: list):
        tasks = []
        for venue in venues:
            for year in years:
                tasks.append(self.collect_for_venue_async(venue, year))

        results = await asyncio.gather(*tasks)
        all_collected = [paper for result in results for paper in result]
        logger.info(f"Total papers collected: {len(all_collected)}")
        return all_collected

    def run(self, venues: list, years: list):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_async(venues, years))
        finally:
            loop.close()

# ...existing code...
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DBLP Conference Scraper with Safe Resumption")
    parser.add_argument("--venues", nargs="+", default=["nips", "iclr", "icml", "satml", "aies", "fat"],
                       help="Venues to scrape (default: all major conferences)")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume mode: only scrape missing venues (SAFE)")
    parser.add_argument("--limit", type=int, help="Limit papers per venue")
    parser.add_argument("--min-year", type=int, default=2020, help="Minimum year")
    parser.add_argument("--max-year", type=int, default=2025, help="Maximum year")
    parser.add_argument("--rescan-no-limit", action="store_true", help="Rescan all venues with no per-venue limit but skip already-saved papers")
    
    args = parser.parse_args()
    
    print("🔄 DBLP Conference Scraper")
    print("=" * 40)
    print(f"Target venues: {', '.join(v.upper() for v in args.venues)}")
    print(f"Year range: {args.min_year}-{args.max_year}")
    if args.limit:
        print(f"Limit per venue: {args.limit:,}")
    
    scraper = DBLPScraper(
        venues=args.venues, 
        min_year=args.min_year, 
        max_year=args.max_year, 
        output_dir="data",
        per_venue_limit=args.limit,
        show_progress=True
    )
    
    if args.resume:
        print("🛡️  SAFE RESUME MODE (appends to existing data)")
        papers = scraper.run_safe_resume()
    elif args.rescan_no_limit:
        print("🔁 RESCAN (no per-venue limit) — skipping papers already present in data/")
        papers = scraper.run_rescan_all()
    else:
        print("⚠️  FULL SCRAPE MODE (overwrites existing data)")
        confirm = input("This will overwrite existing data. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Cancelled.")
            return
        papers = scraper.run()
    
    print(f"\n✅ Scraping complete: {len(papers):,} papers processed")


if __name__ == "__main__":
    main()



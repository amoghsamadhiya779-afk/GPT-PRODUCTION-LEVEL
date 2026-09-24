# app/search.py
"""Scraper utility to retrieve search results from web providers.

Supports caching and fallback chains:
- In-memory TTL cache (15 minutes).
- Serper.dev (if WEB_SEARCH_API_KEY is present).
- DuckDuckGo HTML scraper (fallback).
"""

import logging
import urllib.request
import urllib.parse
import re
import html
import os
import time
import json
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

# TTL Cache configuration. Bounded: every distinct prompt is a distinct key,
# so an unbounded dict grows without limit on a public endpoint.
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes
CACHE_MAX_ENTRIES = 256
_search_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

# Search queries are derived from user prompts (up to 4000 chars); providers
# don't need more than this, and there's no reason to ship whole prompts out.
MAX_QUERY_CHARS = 256
MAX_TITLE_CHARS = 300
MAX_SNIPPET_CHARS = 1000

def _get_from_cache(query: str):
    with _cache_lock:
        if query in _search_cache:
            entry = _search_cache[query]
            if time.time() - entry['timestamp'] < CACHE_TTL_SECONDS:
                _search_cache.move_to_end(query)
                return entry['results']
            else:
                del _search_cache[query]
    return None

def _set_in_cache(query: str, results: list[dict]):
    if not results:
        return  # Do not cache empty results
    with _cache_lock:
        _search_cache[query] = {
            'timestamp': time.time(),
            'results': results
        }
        _search_cache.move_to_end(query)
        while len(_search_cache) > CACHE_MAX_ENTRIES:
            _search_cache.popitem(last=False)


def sanitize_link(href: str) -> str:
    """Return an absolute http(s) URL, or "" if `href` isn't one.

    Result links come from third parties and end up in the UI's <a href>, so
    anything else (javascript:, data:, ...) must never get through. Also
    unwraps DuckDuckGo's redirect links (//duckduckgo.com/l/?uddg=<target>).
    """
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
            return sanitize_link(target) if target else ""  # target is shorter, so this terminates
    except ValueError:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return href

def serper_search(query: str, max_results: int = 3) -> list[dict]:
    """Search using Serper.dev API."""
    api_key = os.environ.get("WEB_SEARCH_API_KEY")
    if not api_key:
        return []
        
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    data = json.dumps({"q": query, "num": max_results}).encode("utf-8")
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
        results = []
        for item in res_data.get("organic", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
            
        if results:
            logger.info("Successfully retrieved %d search results using Serper", len(results))
        return results
    except Exception as e:
        logger.error("Serper search failed: %s", e)
        return []

def duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    """Search DuckDuckGo HTML and return a list of snippets."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read().decode("utf-8", errors="ignore")
            
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            results = []
            
            result_divs = soup.find_all("div", class_="result")
            for div in result_divs:
                if len(results) >= max_results:
                    break
                    
                a_title = div.find("a", class_="result__url")
                if not a_title:
                    continue
                title = a_title.get_text(strip=True)
                link = a_title.get("href", "")
                
                a_snippet = div.find("a", class_="result__snippet")
                snippet = a_snippet.get_text(strip=True) if a_snippet else ""
                
                if title and snippet:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "link": link
                    })
            
            if results:
                logger.info("Successfully retrieved %d search results using BeautifulSoup", len(results))
                return results
                
        except ImportError:
            logger.warning("BeautifulSoup not found. Falling back to regex parser.")
            
        results = []
        url_matches = re.finditer(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_content, re.DOTALL)
        snippet_matches = re.finditer(r'<a class="result__snippet"[^>]*>(.*?)</a>', html_content, re.DOTALL)
        
        for url_match, snippet_match in zip(url_matches, snippet_matches):
            if len(results) >= max_results:
                break
                
            raw_url = url_match.group(1)
            raw_title = url_match.group(2)
            raw_snippet = snippet_match.group(1)
            
            title = re.sub(r'<[^>]+>', '', raw_title)
            title = html.unescape(title).strip()
            
            snippet = re.sub(r'<[^>]+>', '', raw_snippet)
            snippet = html.unescape(snippet).strip()
            
            link = html.unescape(raw_url)

            if title and snippet:
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link
                })
                
        if results:
            logger.info("Successfully retrieved %d search results using Regex fallback", len(results))
        return results
        
    except Exception as e:
        logger.error("DuckDuckGo search failed: %s", e)
        return []

def clean_and_rank_results(query: str, results: list, max_results: int = 3) -> list:
    """Sanitize, rank, de-duplicate and truncate raw provider results
    (ranking lives in app/retrieval.py)."""
    from app.retrieval import get_dense_reranker, select_sources

    prepared = [
        {
            "title": re.sub(r"\s+", " ", r.get("title", "")).strip()[:MAX_TITLE_CHARS],
            "snippet": re.sub(r"\s+", " ", r.get("snippet", "")).strip()[:MAX_SNIPPET_CHARS],
            "link": sanitize_link(r.get("link", "")),
        }
        for r in results
    ]
    return select_sources(query, prepared, max_results, reranker=get_dense_reranker())


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Retrieve search results using Serper -> DDG fallback with caching."""
    query = re.sub(r'\s+', ' ', query).strip()[:MAX_QUERY_CHARS]
    if not query:
        return []
    cached = _get_from_cache(query)
    if cached is not None:
        logger.info("Returning cached web search results (query_len=%d)", len(query))
        return cached

    # Log the length, not the text: queries are user prompts and may be sensitive.
    logger.info("Performing web search (query_len=%d)", len(query))
    
    results = []
    # Try Serper first
    if os.environ.get("WEB_SEARCH_API_KEY"):
        results = serper_search(query, max_results * 2)
            
    # Fallback to DuckDuckGo
    if not results:
        results = duckduckgo_search(query, max_results * 2)
        
    final_results = clean_and_rank_results(query, results, max_results)
    
    if final_results:
        _set_in_cache(query, final_results)
        
    return final_results

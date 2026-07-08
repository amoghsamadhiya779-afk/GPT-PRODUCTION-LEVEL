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

logger = logging.getLogger(__name__)

# TTL Cache configuration
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes
_search_cache = {}
_cache_lock = threading.Lock()

def _get_from_cache(query: str):
    with _cache_lock:
        if query in _search_cache:
            entry = _search_cache[query]
            if time.time() - entry['timestamp'] < CACHE_TTL_SECONDS:
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
            
            link = raw_url
            if "//uddg=" in link:
                try:
                    query_part = link.split("//uddg=")[-1]
                    link = urllib.parse.unquote(query_part)
                except Exception:
                    pass
            
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

import difflib

def clean_and_rank_results(query: str, results: list, max_results: int = 3) -> list:
    if not results:
        return []
        
    query_words = set(re.findall(r'\w+', query.lower()))
    
    cleaned = []
    for r in results:
        snip = re.sub(r'(?:\.{3,}|…)', '', r['snippet'])
        snip = re.sub(r'\s+', ' ', snip).strip()
        
        title = re.sub(r'\s+', ' ', r['title']).strip()
        link = r['link']
        
        snip_words = set(re.findall(r'\w+', snip.lower()))
        overlap = len(query_words.intersection(snip_words))
        
        if snip and title:
            cleaned.append({
                "title": title,
                "snippet": snip,
                "link": link,
                "overlap": overlap
            })
            
    cleaned.sort(key=lambda x: x["overlap"], reverse=True)
    
    final = []
    for c in cleaned:
        is_dup = False
        for f in final:
            ratio = difflib.SequenceMatcher(None, c['snippet'], f['snippet']).ratio()
            if ratio > 0.8:
                is_dup = True
                break
        if not is_dup:
            final.append({"title": c['title'], "snippet": c['snippet'], "link": c['link']})
            if len(final) == max_results:
                break
                
    return final

def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Retrieve search results using Serper -> DDG fallback with caching."""
    cached = _get_from_cache(query)
    if cached is not None:
        logger.info("Returning cached web search results for: '%s'", query)
        return cached

    logger.info("Performing web search query: '%s'", query)
    
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

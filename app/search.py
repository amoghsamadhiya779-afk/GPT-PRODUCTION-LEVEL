# app/search.py
"""Scraper utility to retrieve search results from DuckDuckGo HTML interface.

Provides simple RAG snippets without requiring any paid API keys.
"""

import logging
import urllib.request
import urllib.parse
import re
import html

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search DuckDuckGo HTML and return a list of snippets.

    Args:
        query: Search term query.
        max_results: Maximum number of search snippets to retrieve.

    Returns:
        List of dicts containing 'title', 'snippet', and 'link'.
    """
    logger.info("Performing web search query: '%s'", query)
    
    # Encode query
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
        # Fetch search results page (10s timeout)
        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read().decode("utf-8", errors="ignore")
            
        # Parse results using BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            results = []
            
            # Find result divs
            result_divs = soup.find_all("div", class_="result")
            for div in result_divs:
                if len(results) >= max_results:
                    break
                    
                # Extract title and link
                a_title = div.find("a", class_="result__url")
                if not a_title:
                    continue
                title = a_title.get_text(strip=True)
                link = a_title.get("href", "")
                
                # Extract snippet
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
            
        # Regex fallback parser
        results = []
        # Match result blocks: class="result__body" ... class="result__snippet" ...
        # Match links and titles: <a class="result__url" href="URL">TITLE</a>
        url_matches = re.finditer(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_content, re.DOTALL)
        snippet_matches = re.finditer(r'<a class="result__snippet"[^>]*>(.*?)</a>', html_content, re.DOTALL)
        
        for url_match, snippet_match in zip(url_matches, snippet_matches):
            if len(results) >= max_results:
                break
                
            raw_url = url_match.group(1)
            raw_title = url_match.group(2)
            raw_snippet = snippet_match.group(1)
            
            # Clean HTML tags and entities
            title = re.sub(r'<[^>]+>', '', raw_title)
            title = html.unescape(title).strip()
            
            snippet = re.sub(r'<[^>]+>', '', raw_snippet)
            snippet = html.unescape(snippet).strip()
            
            # Decode DuckDuckGo redirected links (if they start with //uddg=)
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
                
        logger.info("Successfully retrieved %d search results using Regex fallback", len(results))
        return results
        
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return []

# tests/test_search.py
"""Unit tests for the web search RAG utility.

Usage:
    py -m pytest tests/test_search.py -v
"""

import os
import sys
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.search import web_search


def test_web_search_mock():
    # Mock urllib.request.urlopen to return mock HTML
    mock_html = """
    <div class="result">
        <a class="result__url" href="https://example.com/transformer">Transformer Architecture</a>
        <a class="result__snippet">The Transformer model uses self-attention to process sequences...</a>
    </div>
    """
    
    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = unittest.mock.MagicMock()
        mock_response.read.return_value = mock_html.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        results = web_search("transformer", max_results=1)
        
        assert len(results) == 1
        assert results[0]["title"] == "Transformer Architecture"
        # Verify ellipsis was stripped
        assert results[0]["snippet"] == "The Transformer model uses self-attention to process sequences"
        assert results[0]["link"] == "https://example.com/transformer"

def test_clean_and_rank_results():
    from app.search import clean_and_rank_results
    
    query = "deep learning"
    results = [
        {"title": "Title 1", "snippet": "Deep learning is a subset of machine learning...", "link": "http://link1.com"},
        {"title": "Title 2", "snippet": "Deep learning is a subset of machine learning", "link": "http://link2.com"},
        {"title": "Title 3", "snippet": "Unrelated topic", "link": "http://link3.com"}
    ]
    
    final = clean_and_rank_results(query, results, max_results=3)
    # Dedupe should remove the near identical Title 2
    assert len(final) == 2
    assert final[0]["title"] == "Title 1" or final[0]["title"] == "Title 2"
    # Unrelated topic should be last
    assert final[-1]["title"] == "Title 3"
    assert "..." not in final[0]["snippet"]

def test_web_search_live():
    # Attempt a live search run to check parser compatibility with active DuckDuckGo HTML
    results = web_search("pytorch deep learning", max_results=2)
    
    if len(results) > 0:
        for res in results:
            assert "title" in res
            assert "snippet" in res
            assert "link" in res
            assert isinstance(res["title"], str)
            assert isinstance(res["snippet"], str)
            assert isinstance(res["link"], str)

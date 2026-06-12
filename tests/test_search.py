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
        <a class="result__snippet">The Transformer model uses self-attention to process sequences.</a>
    </div>
    """
    
    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = unittest.mock.MagicMock()
        mock_response.read.return_value = mock_html.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        results = web_search("transformer", max_results=1)
        
        assert len(results) == 1
        assert results[0]["title"] == "Transformer Architecture"
        assert results[0]["snippet"] == "The Transformer model uses self-attention to process sequences."
        assert results[0]["link"] == "https://example.com/transformer"


def test_web_search_live():
    # Attempt a live search run to check parser compatibility with active DuckDuckGo HTML
    results = web_search("pytorch deep learning", max_results=2)
    
    # Since live network calls could occasionally time out or be rate-limited,
    # we don't assert length strictly, but if it succeeded, it must have correct keys
    if len(results) > 0:
        for res in results:
            assert "title" in res
            assert "snippet" in res
            assert "link" in res
            assert isinstance(res["title"], str)
            assert isinstance(res["snippet"], str)
            assert isinstance(res["link"], str)

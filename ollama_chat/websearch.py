# ollama_chat/websearch.py
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"

def _search_html(query: str) -> str:
    """Return the raw HTML of DuckDuckGo results (no API key needed)."""
    resp = requests.post(DUCKDUCKGO_URL, data={"q": query})
    resp.raise_for_status()
    return resp.text


def search(query: str, max_results: int = 3) -> List[Tuple[str, str]]:
    """
    Perform a quick DuckDuckGo search and return a list of (title, snippet).
    """
    html = _search_html(query)
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for a in soup.select("a.result__a")[:max_results]:
        title = a.get_text(strip=True)
        # The snippet is the sibling <a class="result__snippet">
        snippet_tag = a.find_parent("div", class_="result").find("a", class_="result__snippet")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append((title, snippet))
    return results

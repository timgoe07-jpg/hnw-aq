from typing import List, Dict, Any, Optional
import requests

from config import Config


GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


def google_linkedin_search(query: str, location: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Use Google Custom Search to find LinkedIn profile URLs."""
    if not (Config.GOOGLE_SEARCH_API_KEY and Config.GOOGLE_SEARCH_CX):
        return []
    q = f"site:linkedin.com/in {query}"
    if location:
        q = f"{q} {location}"
    params = {
        "key": Config.GOOGLE_SEARCH_API_KEY,
        "cx": Config.GOOGLE_SEARCH_CX,
        "q": q,
        "num": max(1, min(limit, 10)),
        "gl": "au",
    }
    try:
        resp = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    items = data.get("items", []) or []
    results: List[Dict[str, Any]] = []
    for item in items:
        title = item.get("title") or ""
        link = item.get("link")
        snippet = item.get("snippet")
        # Extract rough name from title before separators
        name = title.split(" - ")[0].split("|")[0].strip() if title else link
        results.append(
            {
                "full_name": name,
                "profile_url": link,
                "headline": title,
                "about": snippet,
                "location": location,
            }
        )
    return results[:limit]

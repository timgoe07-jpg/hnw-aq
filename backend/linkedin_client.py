"""
LinkedIn search wrapper using SaleLeads Fresh LinkedIn Scraper API (RapidAPI).
Reference: https://saleleads.ai/blog/linkedin-api-search-people
"""
from typing import List, Dict, Any, Optional
import logging
import requests

from config import Config

logger = logging.getLogger(__name__)


SEARCH_URL = "https://fresh-linkedin-scraper-api.p.rapidapi.com/api/v1/search/people"


def _headers() -> Dict[str, str]:
    if not Config.FRESH_LINKEDIN_API_KEY:
        raise ValueError("FRESH_LINKEDIN_API_KEY is not configured")
    return {
        "x-rapidapi-key": Config.FRESH_LINKEDIN_API_KEY,
        "x-rapidapi-host": Config.FRESH_LINKEDIN_API_HOST or "fresh-linkedin-scraper-api.p.rapidapi.com",
    }


def _build_params(query: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"keyword": query, "page": 1, "limit": 10}
    if not filters:
        return params
    if filters.get("location"):
        params["location"] = filters["location"]
    if filters.get("industry"):
        params["industry"] = filters["industry"]
    if filters.get("company"):
        params["company"] = filters["company"]
    if filters.get("school"):
        params["school"] = filters["school"]
    if filters.get("page"):
        params["page"] = filters["page"]
    if filters.get("limit"):
        params["limit"] = min(int(filters["limit"]), 10)
    return params


def _map_result(entry: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    avatar = entry.get("avatar") or []
    avatar_entry = avatar[0] if isinstance(avatar, list) and avatar else {}
    avatar_url = avatar_entry.get("url") if isinstance(avatar_entry, dict) else None
    return {
        "id": entry.get("id") or entry.get("public_identifier"),
        "full_name": entry.get("full_name") or entry.get("name") or "Unknown",
        "headline": entry.get("title"),
        "current_title": entry.get("title"),
        "current_company": entry.get("company"),
        "profile_url": entry.get("url"),
        "location": entry.get("location"),
        "industry": entry.get("industry") or (filters or {}).get("industry"),
        "photo_url": avatar_url,
        "is_premium": entry.get("is_premium"),
    }


def search_profiles(
    query: str, filters: Optional[Dict[str, Any]] = None, access_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not query:
        raise ValueError("Search query is required")
    params = _build_params(query, filters)
    logger.info(
        "SaleLeads RapidAPI search: query=%s location=%s industry=%s page=%s limit=%s",
        query,
        params.get("location"),
        params.get("industry"),
        params.get("page"),
        params.get("limit"),
    )
    try:
        resp = requests.get(SEARCH_URL, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"LinkedIn search failed: {exc}")

    results: List[Dict[str, Any]] = []
    for entry in data.get("data", []):
        results.append(_map_result(entry, filters))
    return results


def get_profile_details(profile_id: str, access_token: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": profile_id,
        "full_name": "",
        "headline": "",
        "current_title": "",
        "current_company": "",
        "location": "",
        "industry": "",
        "about": "",
        "positions": [],
        "photo_url": None,
    }

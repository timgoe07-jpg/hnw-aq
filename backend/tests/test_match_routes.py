import pytest
pytest.importorskip("flask")
import api.match_routes as match_routes


def test_search_with_fallbacks_relaxes_location_filter(monkeypatch):
    calls = []

    def fake_search(query, filters=None, access_token=None):
        calls.append({"query": query, "filters": filters or {}})
        if filters and filters.get("location"):
            return []
        return [{"id": "ok"}]

    monkeypatch.setattr(match_routes, "search_profiles", fake_search)

    results, used_query, attempts = match_routes._search_with_fallbacks(
        ["retiree income", "founder investor", "SMSF trustee"],
        {"location": "Australia", "industry": None},
        max_results=5,
        free_text="",
    )

    assert results and results[0]["id"] == "ok"
    assert used_query == "retiree income founder investor SMSF trustee"
    assert calls[0]["filters"].get("location") == "Australia"
    assert calls[1]["filters"] == {}
    assert attempts[0]["filters"].get("location") == "Australia"
    assert attempts[1]["filters"] == {}


def test_search_with_fallbacks_uses_keyword_fallback(monkeypatch):
    call_order = []

    def fake_search(query, filters=None, access_token=None):
        call_order.append(query)
        if query == "SMSF trustee":
            return [{"id": "winner"}]
        return []

    monkeypatch.setattr(match_routes, "search_profiles", fake_search)

    keywords = [
        "retiree income",
        "founder investor",
        "SMSF trustee",
        "club treasurer",
        "family office CIO",
        "Australia",
    ]
    results, used_query, attempts = match_routes._search_with_fallbacks(
        keywords,
        filters={},
        max_results=3,
        free_text="",
    )

    assert results and results[0]["id"] == "winner"
    assert used_query == "SMSF trustee"
    # Ensure we tried earlier combinations before landing on the fallback keyword.
    assert call_order[:3] == [
        "retiree income founder investor SMSF trustee club treasurer family office CIO Australia",
        "retiree income founder investor SMSF trustee",
        "retiree income",
    ]
    assert attempts[-1]["query"] == "SMSF trustee"


def test_search_with_fallbacks_injects_australia(monkeypatch):
    calls = []

    def fake_search(query, filters=None, access_token=None):
        calls.append({"query": query, "filters": filters or {}})
        return []

    monkeypatch.setattr(match_routes, "search_profiles", fake_search)

    keywords = ["family office"]
    match_routes._search_with_fallbacks(
        keywords,
        filters={},
        max_results=3,
        free_text="",
    )

    # First filter should include Australia
    assert calls
    assert calls[0]["filters"].get("location") == "Australia"

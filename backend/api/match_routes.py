from typing import List, Dict, Any, Optional, Tuple
import json
import copy
from flask import Blueprint, jsonify, request

from personas.loader import load_personas
from personas.models import Profile
from case_studies.loader import load_case_studies
from personas.matcher import match_profiles_to_personas
from linkedin_client import search_profiles
from ai_matcher import ai_rank_profiles
from ai_prospect import generate_search_plan, chat_about_prospect
from config import Config
from background import JobManager
from google_search import google_linkedin_search
import matching_engine

bp = Blueprint("match", __name__, url_prefix="/api/match")
job_manager = JobManager(max_workers=2)
DEFAULT_QUERY = "private credit investor australia"
FALLBACK_QUERIES = [
    "private credit investor australia",
    "high net worth investor australia",
    "family office australia",
    "smsf trustee australia",
    "club treasurer australia",
    "yield income investor australia",
    "income retiree australia",
    "investment manager australia",
    "private credit fund australia",
    "wealth management australia",
]
MAX_LINKEDIN_ATTEMPTS = 12
INCLUDE_KEYWORDS = [
    "private credit",
    "investment",
    "investor",
    "capital",
    "wealth",
    "family office",
    "smsf",
    "trustee",
    "treasurer",
    "pension",
    "retiree",
    "advisor",
    "bank",
    "portfolio",
    "fund",
    "cio",
    "director",
]
EXCLUDE_KEYWORDS = [
    "student",
    "intern",
    "trainee",
    "assistant",
    "brand",
    "game",
    "gaming",
    "library",
    "librarian",
    "developer",
    "engineer",
    "tester",
    "software",
    "marketing",
    "sales",
]


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    deduped = []
    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_query_candidates(keywords: List[str], free_text: str, location: Optional[str]) -> List[str]:
    cleaned = [kw.strip() for kw in keywords if kw and str(kw).strip()]
    base = " ".join(cleaned[:6]).strip()
    candidates: List[str] = []

    def _shorten(q: str) -> List[str]:
        if len(q) <= 70:
            return [q]
        parts = q.split()
        return [" ".join(parts[:5]), " ".join(parts[:3])]

    if base:
        candidates.extend(_shorten(base))
    if len(cleaned) > 3:
        short_base = " ".join(cleaned[:3])
        candidates.extend(_shorten(short_base))
    if free_text:
        candidates.extend(_shorten(free_text.strip()))
    # Push user-provided location into queries to tighten relevance
    if location:
        candidates.extend([f"{location} {c}" for c in cleaned[:4] if c])
        candidates.append(f"{base} {location}".strip())
    for kw in cleaned[:5]:
        candidates.extend(_shorten(kw))
        if "australia" not in kw.lower():
            candidates.append(f"{kw} australia")
    persona_anchors = [
        "retiree income australia",
        "yield investor australia",
        "family office australia",
        "smsf trustee australia",
        "club treasurer australia",
        "high net worth investor australia",
        "private credit investor australia",
    ]
    candidates.extend(persona_anchors)
    candidates.extend(FALLBACK_QUERIES)
    return [q for q in _dedupe_preserve_order(candidates) if q]


def _build_filter_variants(location: Optional[str], industry: Optional[str]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = [{}]
    base: Dict[str, Any] = {}
    if location:
        base["location"] = location
    else:
        base["location"] = "Australia"
    if industry:
        base["industry"] = industry
    if base:
        variants.insert(0, base)
        if base.get("industry") and not base.get("location"):
            variants.append({"industry": base["industry"]})
        if base.get("location"):
            variants.append({k: v for k, v in base.items() if k != "location"})
    return _dedupe_preserve_order(variants)


def _profile_text_from_result(res: Dict[str, Any]) -> str:
    parts = [
        res.get("full_name"),
        res.get("headline"),
        res.get("current_title"),
        res.get("current_company"),
        res.get("industry"),
        res.get("about"),
        res.get("summary"),
    ]
    if res.get("positions"):
        for pos in res.get("positions", []):
            parts.extend([pos.get("title"), pos.get("company"), pos.get("description")])
    return " ".join([p for p in parts if p]).lower()


def _is_relevant_result(res: Dict[str, Any], location_hint: Optional[str], keywords: List[str]) -> bool:
    text = _profile_text_from_result(res)
    if not text:
        return False
    if any(bad in text for bad in EXCLUDE_KEYWORDS):
        return False
    # Require some overlap with include keywords or search keywords to avoid generic/irrelevant profiles
    include_terms = set([kw.lower() for kw in INCLUDE_KEYWORDS + keywords if kw])
    hits = sum(1 for term in include_terms if term and term in text)
    if hits == 0:
        return False
    if location_hint:
        # Prefer candidates that mention target location
        if location_hint.lower() not in (res.get("location") or "").lower() and location_hint.lower() not in text:
            return False
    return True


def _search_with_fallbacks(
    keywords: List[str],
    filters: Optional[Dict[str, Any]],
    max_results: int,
    free_text: str = "",
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:

    location = (filters or {}).get("location")
    queries = _build_query_candidates(keywords, free_text, location)

    if not queries:
        queries = [DEFAULT_QUERY]


    filter_variants = _build_filter_variants(
        (filters or {}).get("location"),
        (filters or {}).get("industry"),
    )
    if {} not in filter_variants:
        filter_variants.append({})
    attempts: List[Dict[str, Any]] = []
    chosen_query = queries[0]

    for query in queries:
        chosen_query = query
        for filt in filter_variants:
            if len(attempts) >= MAX_LINKEDIN_ATTEMPTS:
                chosen_query = query
                return [], chosen_query, attempts
            attempts.append({"query": query, "filters": {k: v for k, v in filt.items() if v}})
            results = search_profiles(query, filt or None)
            if results:
                chosen_query = query
                return results[:max_results], chosen_query, attempts

    return [], chosen_query, attempts


def _google_backfill(
    keywords: List[str], location: Optional[str], max_results: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Use Google CSE to find LinkedIn URLs, then hydrate via LinkedIn API."""
    queries = []
    base = " ".join(keywords[:4]) if keywords else DEFAULT_QUERY
    if base:
        queries.append(base)
    queries.extend([kw for kw in keywords[:6] if kw])
    queries.extend(
        [
            "retiree income australia",
            "yield investor australia",
            "family office australia",
            "smsf trustee australia",
            "private credit investor australia",
            "club treasurer australia",
            "high net worth investor australia",
        ]
    )
    seen_urls = set()
    aggregated: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []

    for q in _dedupe_preserve_order(queries):
        google_results = google_linkedin_search(q, location, limit=max_results * 3)
        attempts.append({"query": q, "filters": {"location": location}, "source": "google"})
        if not google_results:
            continue
        for item in google_results:
            url = item.get("profile_url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            name = item.get("full_name") or q
            enriched: List[Dict[str, Any]] = []
            try:
                enriched = search_profiles(name, {"location": location} if location else None)
            except Exception:
                enriched = []
            if enriched:
                aggregated.extend(enriched)
            else:
                aggregated.append(
                    {
                        "full_name": name,
                        "headline": item.get("headline"),
                        "profile_url": url,
                        "location": location,
                        "about": item.get("about"),
                    }
                )
            if len(aggregated) >= max_results:
                break
        if len(aggregated) >= max_results:
            break

    return aggregated[:max_results], attempts


def _build_profiles(linkedin_results: List[Dict[str, Any]], keywords: List[str], location_hint: Optional[str]) -> List[Profile]:
    profiles: List[Profile] = []
    for res in linkedin_results:
        # Skip profiles that don't meet relevance filters
        if not _is_relevant_result(res, location_hint or res.get("location"), keywords):
            continue
        profiles.append(
            Profile.from_dict(
                {
                    "full_name": res.get("full_name") or res.get("name") or (res.get("profile_url") or "Unknown"),
                    "headline": res.get("headline"),
                    "current_title": res.get("current_title"),
                    "current_company": res.get("current_company"),
                    "profile_url": res.get("profile_url"),
                    "location": res.get("location"),
                    "industry": res.get("industry"),
                    "about_summary": res.get("about") or res.get("summary"),
                    "experience": res.get("positions", []),
                    "photo_url": res.get("photo_url"),
                    "is_premium": res.get("is_premium"),
                }
            )
        )
    return profiles


def _to_engine_persona(p) -> matching_engine.Persona:
    return matching_engine.Persona(
        id=p.id,
        name=p.name,
        slug=p.id,
        description=p.raw_text or p.primary_goal or "",
        goals=[p.primary_goal] if p.primary_goal else [],
        concerns=[p.key_concern] if p.key_concern else [],
        structure=[p.wealth_range_or_structure] if p.wealth_range_or_structure else [],
        investment_behaviour=[p.investment_behaviour] if p.investment_behaviour else [],
        why_private_credit_appeals=p.why_private_credit_appeals or [],
    )


def _to_engine_case(cs) -> matching_engine.CaseStudy:
    return matching_engine.CaseStudy(
        id=cs.id,
        name=cs.title,
        persona_slug=cs.persona_id,
        problem=cs.problem,
        why_it_matters=cs.why_it_matters,
        solution=cs.solution,
        outcome=cs.outcome,
        angle=cs.why_it_matters,
    )


def _to_engine_profile(profile: Profile, idx: int) -> matching_engine.Profile:
    profile_id = profile.profile_url or f"profile-{idx}-{profile.full_name}"
    industries = [profile.industry] if profile.industry else []
    notes = profile.about_summary or ""
    if profile.experience:
        notes = notes + " " + " ".join([exp.description or "" for exp in profile.experience])
    return matching_engine.Profile(
        id=profile_id,
        full_name=profile.full_name,
        headline=profile.headline or "",
        summary=profile.about_summary or "",
        location=profile.location,
        headline_role=profile.current_title,
        current_company=profile.current_company,
        past_companies=[exp.company for exp in profile.experience],
        industries=industries,
        skills=[],
        extra_notes=notes,
    )


def _aggregate_engine_matches(
    engine_results: Dict[str, List[matching_engine.MatchScore]],
    profile_lookup: Dict[str, Profile],
    persona_lookup: Dict[str, str],
) -> List[Any]:
    by_profile: Dict[str, List[PersonaMatchScore]] = {}
    for persona_slug, scores in engine_results.items():
        persona_name = persona_lookup.get(persona_slug, persona_slug)
        for score in scores:
            pm = PersonaMatchScore(
                persona_id=persona_slug,
                persona_name=persona_name,
                score=round(score.combined_score * 100, 2),
                reason=score.explanation,
            )
            by_profile.setdefault(score.profile_id, []).append(pm)

    results: List[ProfileMatchResult] = []
    for pid, matches in by_profile.items():
        profile = profile_lookup.get(pid)
        if not profile:
            continue
        matches.sort(key=lambda m: m.score, reverse=True)
        results.append(ProfileMatchResult(profile=profile, top_persona_matches=matches[:3]))
    return results


def _fallback_profiles_from_personas(personas: List[Profile], limit: int = 3) -> List[Profile]:
    samples: List[Profile] = []
    for persona in personas[:limit]:
        samples.append(
            Profile.from_dict(
                {
                    "full_name": f"{persona.short_label or persona.name} Example",
                    "headline": persona.primary_goal,
                    "current_title": persona.name,
                    "current_company": "Persona-aligned sample",
                    "location": "Australia",
                    "industry": "Investment",
                    "about_summary": persona.key_concern,
                    "experience": [{"title": "Sample experience", "company": "Sample Co"}],
                    "profile_url": None,
                }
            )
        )
    return samples


def _execute_match(payload: Dict[str, Any]) -> Dict[str, Any]:
    persona_ids = payload.get("personaIds") or []
    free_text_query = payload.get("freeTextQuery") or ""
    location = payload.get("locationFilter")
    industry = payload.get("industryFilter")
    max_results = min(max(int(payload.get("maxResults") or 5), 1), Config.MAX_LINKEDIN_RESULTS)
    use_ai_ranking = bool(payload.get("useAiRanking"))

    personas = load_personas()
    if persona_ids:
        personas = [p for p in personas if p.id in persona_ids]
    case_studies = load_case_studies()

    search_plan = generate_search_plan(personas, case_studies, free_text_query)
    keywords = search_plan.get("keywords") or []
    linkedin_results, final_query, attempts = _search_with_fallbacks(
        keywords,
        {"location": location, "industry": industry},
        max_results,
        free_text_query,
    )

    used_google = False
    google_attempts: List[Dict[str, Any]] = []
    if not linkedin_results:
        google_results, google_attempts = _google_backfill(keywords, location or "Australia", max_results)
        attempts.extend(google_attempts)
        if google_results:
            linkedin_results = google_results
            used_google = True

    if not linkedin_results:
        return {
            "profiles": [],
            "search_plan": {
                "keywords": keywords,
                "explanation": f"{search_plan.get('explanation')} | No LinkedIn or Google results.",
                "raw_query": final_query,
                "attempted_queries": attempts,
            },
            "metadata": {"used_ai_ranking": False, "synthetic_profiles": False},
        }

    profiles = _build_profiles(linkedin_results, keywords, location or "")
    # If LinkedIn returned entries but none passed relevance filters, try Google backfill once.
    if not profiles and not used_google:
        google_results, google_attempts = _google_backfill(keywords, location or "Australia", max_results)
        attempts.extend(google_attempts)
        if google_results:
            profiles = _build_profiles(google_results, keywords, location or "")
            used_google = True
    metadata = {"used_ai_ranking": False}
    min_score = 0  # allow low-scoring real profiles to surface
    try:
        engine_personas = [_to_engine_persona(p) for p in personas]
        engine_cases = [_to_engine_case(cs) for cs in case_studies]
        engine_profiles = [_to_engine_profile(p, idx) for idx, p in enumerate(profiles)]
        profile_lookup = {ep.id: profiles[idx] for idx, ep in enumerate(engine_profiles)}
        persona_lookup = {p.id: p.name for p in personas}
        engine_results = matching_engine.match_profiles(
            engine_personas, engine_cases, engine_profiles, top_k_per_persona=5
        )
        matches = _aggregate_engine_matches(engine_results, profile_lookup, persona_lookup)
        metadata["used_ai_ranking"] = True
        metadata["used_matching_engine"] = True
    except Exception:
        metadata["used_matching_engine"] = False
        if use_ai_ranking:
            try:
                matches = ai_rank_profiles(profiles, personas)
                metadata["used_ai_ranking"] = True
            except Exception:
                matches = match_profiles_to_personas(profiles, personas, top_n=3, min_score=min_score)
                metadata["used_ai_ranking"] = False
        else:
            matches = match_profiles_to_personas(profiles, personas, top_n=3, min_score=min_score)

    if not matches:
        return {
            "profiles": [],
            "search_plan": {
                "keywords": keywords,
                "explanation": f"{search_plan.get('explanation')} | No usable matches after filtering.",
                "raw_query": final_query,
                "attempted_queries": attempts,
            },
            "metadata": {"used_ai_ranking": metadata.get("used_ai_ranking"), "synthetic_profiles": False},
        }
    else:
        search_plan_note = search_plan.get("explanation")

    return {
        "profiles": [m.to_dict() for m in matches],
        "search_plan": {
            "keywords": keywords,
            "explanation": search_plan_note,
            "raw_query": final_query,
            "attempted_queries": attempts,
        },
        "metadata": metadata,
    }


@bp.route("/suggest-profiles", methods=["POST"])
def suggest_profiles():
    payload = request.get_json(force=True)
    try:
        response = _execute_match(payload)
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@bp.route("/suggest-profiles/async", methods=["POST"])
def suggest_profiles_async():
    payload = request.get_json(force=True)
    job_id = job_manager.submit(_execute_match, copy.deepcopy(payload))
    return jsonify({"job_id": job_id})


@bp.route("/suggest-profiles/job/<job_id>", methods=["GET"])
def suggest_profiles_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@bp.route("/prospect-chat", methods=["POST"])
def prospect_chat():
    payload = request.get_json(force=True)
    profile_data = payload.get("profile")
    question = (payload.get("question") or "").strip()
    raw_history = payload.get("history") or []
    history = []
    if isinstance(raw_history, list):
        for msg in raw_history[-6:]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            content = msg.get("content")
            if not content:
                continue
            history.append({"role": role, "content": content})
    if not profile_data:
        return jsonify({"error": "Profile payload missing"}), 400
    if not question:
        return jsonify({"error": "Question is required"}), 400

    profile = Profile.from_dict(profile_data)
    personas = load_personas()
    case_studies = load_case_studies()
    answer = chat_about_prospect(profile, personas, case_studies, question, history)
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return jsonify({"answer": answer, "history": updated_history})

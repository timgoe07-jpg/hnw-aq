"""
Matching engine for personas, case studies, and LinkedIn profiles.

This module is pure Python (no Flask endpoints). It can be imported by a Flask
route (e.g., /api/matches) to score LinkedIn profiles—already fetched via the
official API—against your personas and case studies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import os
import logging
import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

logger = logging.getLogger(__name__)

# -----------------------------
# Data models
# -----------------------------


@dataclass
class Persona:
    id: str
    name: str
    slug: str
    description: str
    goals: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    structure: List[str] = field(default_factory=list)
    investment_behaviour: List[str] = field(default_factory=list)
    why_private_credit_appeals: List[str] = field(default_factory=list)


@dataclass
class CaseStudy:
    id: str
    name: str
    persona_slug: str
    problem: str
    why_it_matters: str
    solution: str
    outcome: str
    angle: str


@dataclass
class Profile:
    id: str
    full_name: str
    headline: str
    summary: Optional[str] = None
    location: Optional[str] = None
    headline_role: Optional[str] = None
    current_company: Optional[str] = None
    past_companies: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    extra_notes: Optional[str] = None


@dataclass
class MatchScore:
    profile_id: str
    persona_slug: str
    persona_similarity: float
    case_study_id: Optional[str]
    case_study_similarity: Optional[float]
    combined_score: float
    explanation: str


# -----------------------------
# Text builders
# -----------------------------


def build_profile_text(profile: Profile) -> str:
    """Combine profile fields into a readable blob for embedding/keyword scoring."""
    parts = [
        profile.full_name,
        profile.headline,
        profile.summary,
        profile.location,
        profile.headline_role,
        profile.current_company,
        ", ".join(profile.past_companies or []),
        ", ".join(profile.industries or []),
        ", ".join(profile.skills or []),
        profile.extra_notes,
    ]
    return " | ".join([p for p in parts if p])


def build_persona_text(persona: Persona) -> str:
    parts = [
        persona.name,
        persona.description,
        "; ".join(persona.goals),
        "; ".join(persona.concerns),
        "; ".join(persona.structure),
        "; ".join(persona.investment_behaviour),
        "; ".join(persona.why_private_credit_appeals),
    ]
    return " | ".join([p for p in parts if p])


def build_case_study_text(case_study: CaseStudy) -> str:
    parts = [
        case_study.problem,
        case_study.why_it_matters,
        case_study.solution,
        case_study.outcome,
        case_study.angle,
    ]
    return " | ".join([p for p in parts if p])


# -----------------------------
# Embeddings & similarity
# -----------------------------


def _get_openai_client() -> Optional["OpenAI"]:
    if not OpenAI:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover
        logger.warning("OpenAI client init failed: %s", exc)
        return None


def _embeddings_enabled() -> bool:
    flag = os.getenv("MATCHING_ENGINE_USE_EMBEDDINGS", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def get_embedding(text: str) -> Optional[List[float]]:
    """Fetch an embedding; return None on failure."""
    if not _embeddings_enabled():
        return None
    client = _get_openai_client()
    if not client:
        return None
    try:
        resp = client.embeddings.create(model="text-embedding-3-large", input=text)
        return resp.data[0].embedding  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        logger.warning("Embedding failed: %s", exc)
        return None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -----------------------------
# Scoring helpers
# -----------------------------


PERSONA_KEYWORD_BOOSTS: Dict[str, List[Tuple[str, float]]] = {
    "retirees-chasing-yield": [
        ("retiree", 0.05),
        ("retirement", 0.05),
        ("pension", 0.04),
        ("income", 0.03),
        ("yield", 0.03),
    ],
    "self-directed-hnw": [
        ("founder", 0.05),
        ("entrepreneur", 0.04),
        ("investor", 0.03),
        ("angel", 0.04),
    ],
    "smsf-trustee": [
        ("smsf", 0.1),
        ("superannuation", 0.05),
        ("trustee", 0.04),
        ("compliance", 0.03),
    ],
    "community-club-treasurer": [
        ("treasurer", 0.1),
        ("club", 0.04),
        ("committee", 0.03),
        ("fiduciary", 0.03),
    ],
    "multi-generational-family-office": [
        ("family office", 0.1),
        ("cio", 0.05),
        ("investment director", 0.04),
        ("allocator", 0.03),
        ("portfolio", 0.03),
    ],
}


def _keyword_boost(text: str, persona_slug: str) -> float:
    text_lower = text.lower()
    boosts = 0.0
    for kw, bonus in PERSONA_KEYWORD_BOOSTS.get(persona_slug, []):
        if kw in text_lower:
            boosts += bonus
    return boosts


def _keyword_fallback_similarity(text_a: str, text_b: str) -> float:
    """Simple Jaccard over token sets as a fallback when embeddings are unavailable."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union else 0.0


def score_profile_against_persona(profile: Profile, persona: Persona) -> float:
    profile_text = build_profile_text(profile)
    persona_text = build_persona_text(persona)
    emb_p = get_embedding(profile_text)
    emb_per = get_embedding(persona_text)
    if emb_p and emb_per:
        sim = cosine_similarity(emb_p, emb_per)
    else:
        sim = _keyword_fallback_similarity(profile_text, persona_text)
    sim += _keyword_boost(profile_text, persona.slug)
    return min(sim, 1.0)


def score_profile_against_case_studies(
    profile: Profile, case_studies: List[CaseStudy], persona_slug: str
) -> Tuple[Optional[CaseStudy], float]:
    relevant = [cs for cs in case_studies if cs.persona_slug == persona_slug]
    if not relevant:
        return None, 0.0
    profile_text = build_profile_text(profile)
    profile_emb = get_embedding(profile_text)
    best_cs: Optional[CaseStudy] = None
    best_score = 0.0
    for cs in relevant:
        cs_text = build_case_study_text(cs)
        cs_emb = get_embedding(cs_text) if profile_emb else None
        if profile_emb and cs_emb:
            sim = cosine_similarity(profile_emb, cs_emb)
        else:
            sim = _keyword_fallback_similarity(profile_text, cs_text)
        if sim > best_score:
            best_score = sim
            best_cs = cs
    return best_cs, best_score


def _explanation(profile: Profile, persona: Persona, cs: Optional[CaseStudy], sim: float, cs_sim: float) -> str:
    bits = []
    if profile.headline:
        bits.append(f"Headline: {profile.headline}")
    if profile.summary:
        bits.append(f"Summary mentions: {profile.summary[:120]}...")
    if profile.skills:
        bits.append(f"Skills: {', '.join(profile.skills[:5])}")
    if cs:
        bits.append(f"Aligns with case study {cs.name}")
    bits.append(f"Similarity: persona {sim:.2f}" + (f", case {cs_sim:.2f}" if cs else ""))
    return "; ".join(bits)


# -----------------------------
# Public API
# -----------------------------


def match_profiles(
    personas: List[Persona],
    case_studies: List[CaseStudy],
    profiles: List[Profile],
    top_k_per_persona: int = 20,
) -> Dict[str, List[MatchScore]]:
    results: Dict[str, List[MatchScore]] = {}
    for persona in personas:
        persona_matches: List[MatchScore] = []
        for profile in profiles:
            persona_sim = score_profile_against_persona(profile, persona)
            best_cs, cs_sim = score_profile_against_case_studies(profile, case_studies, persona.slug)
            combined = 0.7 * persona_sim + (0.3 * cs_sim if best_cs else 0.0)
            persona_matches.append(
                MatchScore(
                    profile_id=profile.id,
                    persona_slug=persona.slug,
                    persona_similarity=round(persona_sim, 4),
                    case_study_id=best_cs.id if best_cs else None,
                    case_study_similarity=round(cs_sim, 4) if best_cs else None,
                    combined_score=round(combined, 4),
                    explanation=_explanation(profile, persona, best_cs, persona_sim, cs_sim),
                )
            )
        persona_matches.sort(key=lambda m: m.combined_score, reverse=True)
        results[persona.slug] = persona_matches[:top_k_per_persona]
    return results


def search_best_matches(
    personas: List[Persona],
    case_studies: List[CaseStudy],
    profiles: List[Profile],
    min_score: float = 0.4,
) -> List[MatchScore]:
    flattened: List[MatchScore] = []
    per_persona = match_profiles(personas, case_studies, profiles, top_k_per_persona=len(profiles))
    for matches in per_persona.values():
        for m in matches:
            if m.combined_score >= min_score:
                flattened.append(m)
    flattened.sort(key=lambda m: m.combined_score, reverse=True)
    return flattened


# -----------------------------
# Example usage (for local testing only)
# -----------------------------


if __name__ == "__main__":  # pragma: no cover
    # Dummy data to illustrate usage. In production, load real personas/case studies/profiles.
    personas = [
        Persona(
            id="1",
            name="Retirees chasing yield",
            slug="retirees-chasing-yield",
            description="Self-funded retirees seeking dependable income",
            goals=["Income replacement", "Capital preservation"],
            concerns=["Inflation", "Volatility"],
            structure=["SMSF", "Personal account"],
            investment_behaviour=["Conservative", "Income focused"],
            why_private_credit_appeals=["Steady yield", "Secured lending"],
        ),
        Persona(
            id="2",
            name="Self directed HNW",
            slug="self-directed-hnw",
            description="Entrepreneurial HNW seeking alternatives",
            goals=["Diversification", "Non-mainstream assets"],
            concerns=["Idle cash", "Public market volatility"],
            structure=["Family trust"],
            investment_behaviour=["Opportunistic", "Co-investment friendly"],
            why_private_credit_appeals=["Exclusive deals", "Co-investments"],
        ),
    ]

    case_studies = [
        CaseStudy(
            id="cs1",
            name="Margaret",
            persona_slug="retirees-chasing-yield",
            problem="Low term deposit rates",
            why_it_matters="Cannot fund lifestyle",
            solution="Allocated to secured private credit",
            outcome="8-10% yield with monthly distributions",
            angle="Income and capital preservation",
        ),
        CaseStudy(
            id="cs2",
            name="James",
            persona_slug="self-directed-hnw",
            problem="Idle cash after business sale",
            why_it_matters="Needs diversification and yield",
            solution="Access to curated private credit deals",
            outcome="Deployed across secured loans with reporting",
            angle="Entrepreneurial alignment",
        ),
    ]

    profiles = [
        Profile(
            id="p1",
            full_name="Alice Warren",
            headline="Retired CFO | Seeking steady income",
            summary="Self-funded retiree managing SMSF; interested in secured income products.",
            location="Sydney, Australia",
            headline_role="Retired",
            current_company=None,
            past_companies=["ABC Corp"],
            industries=["Finance"],
            skills=["Treasury", "Risk management", "Portfolio"],
        ),
        Profile(
            id="p2",
            full_name="Brian Lee",
            headline="Founder & Investor",
            summary="Exited tech founder now investing in private deals; open to private credit and co-investments.",
            location="Melbourne, Australia",
            headline_role="Founder",
            current_company="Family Office",
            past_companies=["TechCo"],
            industries=["Technology", "Investment"],
            skills=["Venture investing", "Portfolio construction", "Due diligence"],
        ),
    ]

    matches = match_profiles(personas, case_studies, profiles, top_k_per_persona=5)
    for slug, scores in matches.items():
        print(f"\nPersona: {slug}")
        for m in scores:
            print(f" - {m.profile_id}: combined={m.combined_score} persona={m.persona_similarity} cs={m.case_study_similarity}")
            print(f"   Explanation: {m.explanation}")

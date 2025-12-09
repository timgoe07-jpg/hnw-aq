from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .models import Profile, Persona, ProfileMatchResult, PersonaMatchScore
from case_studies.loader import load_case_studies

# Cache case study text by persona id to enrich matching context.
_CASE_TEXT_CACHE: Dict[str, str] = {}
_NEGATIVE_KEYWORDS = [
    "software engineer",
    "developer",
    "devops",
    "data scientist",
    "it support",
    "mobile app",
    "outsourcing",
    "offshore",
    "consultant",
    "freelance",
    "web app",
    "hanoi",
    "vietnam",
]
MIN_RELEVANCE_SCORE = 5.0


def _persona_to_text(persona: Persona) -> str:
    if not _CASE_TEXT_CACHE:
        for cs in load_case_studies():
            text = " ".join(
                [cs.title, cs.problem, cs.why_it_matters, cs.solution, cs.outcome]
            )
            _CASE_TEXT_CACHE[cs.persona_id] = (_CASE_TEXT_CACHE.get(cs.persona_id, "") + " " + text).strip()

    case_text = _CASE_TEXT_CACHE.get(persona.id, "")
    parts = [
        persona.name,
        persona.short_label,
        persona.age_range,
        persona.wealth_range_or_structure,
        persona.primary_goal,
        persona.key_concern,
        persona.investment_behaviour,
        " ".join(persona.why_private_credit_appeals or []),
        persona.raw_text,
        case_text,
        # Bias matching toward Australian finance/wealth context
        "Australia finance investment wealth trustee club treasurer family office SMSF",
    ]
    return " ".join([p for p in parts if p])


def _profile_to_text(profile: Profile) -> str:
    parts = [
        profile.full_name,
        profile.headline,
        profile.current_title,
        profile.current_company,
        profile.location,
        profile.industry,
        profile.about_summary,
    ]
    for role in profile.experience:
        parts.extend([role.title, role.company, role.description or ""])
    return " ".join([p for p in parts if p])


def _persona_heuristic_boost(profile: Profile, persona: Persona) -> float:
    """Add small boosts for clear persona signals that TF-IDF might dilute."""
    text = _profile_to_text(profile).lower()
    persona_id = persona.id
    boost = 0.0
    if "private credit" in text:
        boost += 6
    if "wealth" in text or "family office" in text:
        boost += 4
    if "finance" in text or "bank" in text or "treasury" in text:
        boost += 3
    keyword_rules: Dict[str, List[tuple]] = {
        "retirees-chasing-yield": [
            ("retired", 6),
            ("retiree", 6),
            ("pension", 5),
            ("income", 3),
            ("yield", 3),
            ("fixed income", 4),
        ],
        "self-directed-hnw": [
            ("founder", 6),
            ("angel investor", 6),
            ("investor", 3),
            ("entrepreneur", 5),
            ("private investor", 5),
        ],
        "smsf-trustee": [
            ("smsf", 10),
            ("self managed super", 8),
            ("trustee", 4),
            ("superannuation", 4),
            ("accountant", 2),
        ],
        "community-club-treasurer": [
            ("treasurer", 10),
            ("club", 4),
            ("association", 3),
            ("committee", 4),
            ("not-for-profit", 5),
            ("nfp", 5),
        ],
        "multi-generational-family-office": [
            ("family office", 10),
            ("cio", 6),
            ("chief investment officer", 8),
            ("investment director", 6),
            ("portfolio manager", 4),
            ("wealth", 2),
        ],
    }
    for keyword, weight in keyword_rules.get(persona_id, []):
        if keyword in text:
            boost += weight

    if profile.location and "australia" in profile.location.lower():
        boost += 2
    return boost


def _persona_penalty(profile: Profile, persona: Persona) -> float:
    """Apply penalties for clearly non-aligned roles/locations to avoid noisy matches."""
    text = _profile_to_text(profile).lower()
    penalty = 0.0
    if profile.location:
        loc = profile.location.lower()
        if "australia" not in loc:
            penalty += 20
            if "vietnam" in loc or "hanoi" in loc:
                penalty += 8
    for kw in _NEGATIVE_KEYWORDS:
        if kw in text:
            penalty += 8
            break
    if _persona_heuristic_boost(profile, persona) == 0:
        penalty += 5
    return penalty


def match_profiles_to_personas(
    profiles: List[Profile],
    personas: List[Persona],
    top_n: int = 3,
    min_score: Optional[float] = MIN_RELEVANCE_SCORE,
) -> List[ProfileMatchResult]:
    if not profiles or not personas:
        return []
    threshold = MIN_RELEVANCE_SCORE if min_score is None else min_score
    persona_texts = [_persona_to_text(p) for p in personas]
    profile_texts = [_profile_to_text(p) for p in profiles]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(persona_texts + profile_texts)
    persona_vectors = matrix[: len(personas)]
    profile_vectors = matrix[len(personas) :]

    cosine_similarities = linear_kernel(profile_vectors, persona_vectors)

    results: List[ProfileMatchResult] = []
    for i, profile in enumerate(profiles):
        sims = cosine_similarities[i]
        scores: List[PersonaMatchScore] = []
        for j, persona in enumerate(personas):
            boost = _persona_heuristic_boost(profile, persona)
            penalty = _persona_penalty(profile, persona)
            base = float(sims[j] * 100)
            if base < 0:
                base = 0.0
            scores.append(
                PersonaMatchScore(
                    persona_id=persona.id,
                    persona_name=persona.name,
                    score=max(min(round(base + boost - penalty, 2), 100.0), 0.0),
                )
            )
        scores.sort(key=lambda s: s.score, reverse=True)
        if scores and scores[0].score >= threshold:
            results.append(
                ProfileMatchResult(profile=profile, top_persona_matches=scores[:top_n])
            )
    return results

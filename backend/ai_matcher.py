import json
import os
from typing import Dict, List, Optional, Tuple

from personas.models import Persona, Profile, PersonaMatchScore, ProfileMatchResult
from personas.matcher import match_profiles_to_personas as tfidf_match
from case_studies.loader import load_case_studies
from case_studies.models import CaseStudy
from config import Config

try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:  # pragma: no cover
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None


def build_persona_context(personas: List[Persona], limit: Optional[int] = None) -> str:
    selected = personas[:limit] if limit else personas
    blocks = []
    for p in selected:
        appeals = ", ".join(p.why_private_credit_appeals or [])
        blocks.append(
            f"- id: {p.id}\n"
            f"  name: {p.name}\n"
            f"  goal: {p.primary_goal}\n"
            f"  concern: {p.key_concern}\n"
            f"  behaviour: {p.investment_behaviour}\n"
            f"  appeals: {appeals}"
        )
    return "\n".join(blocks)


def build_case_context(cases: List[CaseStudy], limit: Optional[int] = None) -> str:
    selected = cases[:limit] if limit else cases
    blocks = []
    for cs in selected:
        blocks.append(
            f"- id: {cs.id}\n"
            f"  persona_id: {cs.persona_id}\n"
            f"  title: {cs.title}\n"
            f"  problem: {cs.problem}\n"
            f"  why_it_matters: {cs.why_it_matters}\n"
            f"  solution: {cs.solution}\n"
            f"  outcome: {cs.outcome}"
        )
    return "\n".join(blocks)


def _langchain_available() -> bool:
    return all([ChatOpenAI, ChatPromptTemplate, StrOutputParser, Config.OPENAI_API_KEY])


def _split_profiles(profiles: List[Profile], limit: int) -> Tuple[List[Profile], List[Profile]]:
    return profiles[:limit], profiles[limit:]


def ai_rank_profiles(
    profiles: List[Profile], personas: List[Persona], model: str = "gpt-4o-mini"
) -> List[ProfileMatchResult]:
    """Use LangChain + OpenAI to rank profiles against personas with reasons."""
    if not profiles:
        return []
    if not _langchain_available():
        return tfidf_match(profiles, personas, top_n=3)

    max_ai_profiles = int(os.getenv("MAX_AI_PROFILES") or 6)
    ai_profiles, fallback_profiles = _split_profiles(profiles, max_ai_profiles)
    persona_ctx = build_persona_context(personas)
    case_ctx = build_case_context(load_case_studies())

    llm = ChatOpenAI(model=model, temperature=0.1, openai_api_key=Config.OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a private credit analyst. Score LinkedIn-style profiles against investor personas and case studies.",
            ),
            (
                "human",
                """
Personas:
{persona_context}

Case studies:
{case_context}

Profiles JSON (array):
{profiles_json}

Return JSON with `results` like:
{{
  "results": [
    {{
      "profile_index": 0,
      "matches": [
        {{"persona_id": "...", "persona_name": "...", "score": 0-100, "reason": "..."}}
      ]
    }}
  ]
}}
`profile_index` must map to each array entry (0-based). Limit matches to top 3 per profile.
""",
            ),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser

    results_by_index: Dict[int, List[PersonaMatchScore]] = {}
    if ai_profiles:
        payload = [
            {"profile_index": idx, "profile": p.to_dict()} for idx, p in enumerate(ai_profiles)
        ]
        try:
            raw = chain.invoke(
                {
                    "persona_context": persona_ctx,
                    "case_context": case_ctx,
                    "profiles_json": json.dumps(payload, ensure_ascii=False),
                }
            )
            data = json.loads(raw)
            for entry in data.get("results", []):
                idx = entry.get("profile_index")
                if idx is None:
                    continue
                matches_raw = entry.get("matches", [])
                scores: List[PersonaMatchScore] = []
                for m in matches_raw:
                    pid = m.get("persona_id") or ""
                    name = m.get("persona_name") or pid
                    try:
                        score_val = float(m.get("score", 0))
                    except Exception:
                        score_val = 0.0
                    scores.append(
                        PersonaMatchScore(
                            persona_id=pid,
                            persona_name=name,
                            score=round(score_val, 2),
                            reason=m.get("reason"),
                        )
                    )
                scores.sort(key=lambda s: s.score, reverse=True)
                results_by_index[int(idx)] = scores[:3]
        except Exception:
            fallback_profiles.extend(ai_profiles)
            results_by_index = {}

    final_results: List[ProfileMatchResult] = []
    for idx, profile in enumerate(ai_profiles):
        matches = results_by_index.get(idx)
        if not matches:
            matches = tfidf_match([profile], personas, top_n=3)[0].top_persona_matches
        final_results.append(ProfileMatchResult(profile=profile, top_persona_matches=matches))

    if fallback_profiles:
        final_results.extend(tfidf_match(fallback_profiles, personas, top_n=3))

    return final_results

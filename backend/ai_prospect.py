import json
from typing import List, Dict, Any, Optional

from personas.models import Persona, Profile
from case_studies.models import CaseStudy
from ai_matcher import build_persona_context, build_case_context
from config import Config

try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:  # pragma: no cover
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None


def _llm_available() -> bool:
    return all([Config.OPENAI_API_KEY, ChatOpenAI, ChatPromptTemplate, StrOutputParser])


def _default_keywords(personas: List[Persona], free_text: str) -> List[str]:
    base = [
        "private credit investor",
        "high net worth investor",
        "family office australia",
        "smsf trustee",
        "club treasurer",
        "income retiree",
        "yield investor",
    ]
    for persona in personas[:3]:
        base.append(persona.short_label or persona.name)
        if persona.primary_goal:
            first_word = persona.primary_goal.split(" ")[0]
            base.append(first_word)
    if free_text:
        base.append(free_text)
    return [kw for kw in {kw.strip(): kw for kw in base if kw}.values()]


def generate_search_plan(
    personas: List[Persona],
    case_studies: List[CaseStudy],
    free_text: str = "",
) -> Dict[str, Any]:
    """Use LangChain to craft focused keywords, fallback to heuristics."""
    if not personas:
        return {"keywords": ["high net worth private credit"], "explanation": "Default keywords"}

    if not _llm_available():
        keywords = _default_keywords(personas, free_text)
        return {
            "keywords": keywords,
            "explanation": "Heuristic keywords derived from personas (LangChain unavailable).",
        }

    persona_ctx = build_persona_context(personas, limit=3)
    case_ctx = build_case_context(case_studies, limit=3)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=Config.OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert LinkedIn sourcer for Capspace Private Credit finding high net worth prospects.",
            ),
            (
                "human",
                """
Personas:
{persona_context}

Case studies:
{case_context}

Optional user query: {free_text}

Return JSON:
{{
  "keywords": ["keyword1", "keyword2", ... up to 5],
  "explanation": "why these keywords map to personas/case studies"
}}
""",
            ),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    try:
        raw = chain.invoke(
            {
                "persona_context": persona_ctx,
                "case_context": case_ctx,
                "free_text": free_text or "None",
            }
        )
        data = json.loads(raw)
    except Exception:
        keywords = _default_keywords(personas, free_text)
        return {
            "keywords": keywords,
            "explanation": "Fallback keywords (unable to parse LangChain output).",
        }

    keywords = [kw.strip() for kw in data.get("keywords", []) if kw and kw.strip()]
    if not keywords:
        keywords = _default_keywords(personas, free_text)
    explanation = data.get("explanation") or "AI-selected keywords aligned to personas and case studies."
    return {"keywords": keywords, "explanation": explanation}


def chat_about_prospect(
    profile: Profile,
    personas: List[Persona],
    case_studies: List[CaseStudy],
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    if not question:
        return "Please provide a question for the AI coach."
    if not _llm_available():
        return "AI chat is unavailable (missing OpenAI/LangChain configuration)."

    persona_ctx = build_persona_context(personas)
    case_ctx = build_case_context(case_studies)
    history_text = ""
    if history:
        trimmed = history[-6:]
        history_text = "\n".join([f"{msg.get('role','user')}: {msg.get('content','')}" for msg in trimmed])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=Config.OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Capspace's persona strategist. Reference personas and case studies when advising on outreach.",
            ),
            (
                "human",
                """
Persona context:
{persona_context}

Case studies:
{case_context}

Profile JSON:
{profile_json}

Prior conversation:
{history}

Question:
{question}

Respond in 3-5 sentences, referencing the most relevant personas/case studies explicitly.
""",
            ),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    profile_json = json.dumps(profile.to_dict(), ensure_ascii=False)
    try:
        response = chain.invoke(
            {
                "persona_context": persona_ctx,
                "case_context": case_ctx,
                "profile_json": profile_json,
                "history": history_text or "None",
                "question": question,
            }
        )
        return response.strip()
    except Exception as exc:  # pragma: no cover
        return f"AI chat failed: {exc}"

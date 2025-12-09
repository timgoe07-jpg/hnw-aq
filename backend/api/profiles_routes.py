import csv
import io
from typing import List
from flask import Blueprint, jsonify, request

from personas.loader import load_personas
from personas.models import Profile
from personas.matcher import match_profiles_to_personas
from ai_matcher import ai_rank_profiles
from config import Config
import json

# Optional OpenAI import for AI explanations
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None

bp = Blueprint("profiles", __name__, url_prefix="/api")


def _parse_profiles_from_json(payload) -> List[Profile]:
    profiles = []
    for item in payload.get("profiles", []):
        profiles.append(Profile.from_dict(item))
    return profiles


def _parse_profiles_from_csv(file_storage) -> List[Profile]:
    text = file_storage.stream.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    profiles: List[Profile] = []
    for row in reader:
        profiles.append(
            Profile.from_dict(
                {
                    "full_name": row.get("name") or row.get("full_name"),
                    "current_title": row.get("title") or row.get("current_title"),
                    "current_company": row.get("company") or row.get("current_company"),
                    "profile_url": row.get("profile_url"),
                    "about_summary": row.get("summary") or row.get("about_summary"),
                    "industry": row.get("industry"),
                    "headline": row.get("headline"),
                }
            )
        )
    return profiles


@bp.route("/profiles/match", methods=["POST"])
def match_profiles():
    profiles: List[Profile] = []
    if request.files:
        csv_file = next(iter(request.files.values()))
        profiles = _parse_profiles_from_csv(csv_file)
    else:
        payload = request.get_json(silent=True) or {}
        profiles = _parse_profiles_from_json(payload)

    if not profiles:
        return jsonify({"error": "No profiles provided"}), 400

    personas = load_personas()
    results = match_profiles_to_personas(profiles, personas)
    return jsonify({"results": [r.to_dict() for r in results]})


@bp.route("/profiles/ai-match", methods=["POST"])
def ai_match_profiles():
    profiles: List[Profile] = []
    payload = request.get_json(silent=True) or {}
    profiles = _parse_profiles_from_json(payload)
    if not profiles:
        return jsonify({"error": "No profiles provided"}), 400
    personas = load_personas()
    results = ai_rank_profiles(profiles, personas)
    return jsonify({"results": [r.to_dict() for r in results]})


@bp.route("/profiles/ai-explain", methods=["POST"])
def ai_explain_profile():
    payload = request.get_json(force=True) or {}
    profile_data = payload.get("profile")
    if not profile_data:
        return jsonify({"error": "No profile provided"}), 400
    profile = Profile.from_dict(profile_data)
    personas = load_personas()
    if not Config.OPENAI_API_KEY or OpenAI is None:
        return jsonify(
            {
                "explanation": "AI explanation unavailable (missing OpenAI API key).",
                "profile": profile.to_dict(),
            }
        )
    persona_summaries = "\n".join(
        [f"- {p.id}: {p.name} | goal: {p.primary_goal}; concerns: {p.key_concern}" for p in personas]
    )
    profile_text = json.dumps(profile.to_dict(), ensure_ascii=False)
    prompt = f"""
You are an analyst. Given personas and a LinkedIn-like profile JSON, explain why this profile is or isn't a strong fit for the top personas.

Personas:
{persona_summaries}

Profile JSON:
{profile_text}

Return a concise paragraph (3-5 sentences) describing the fit and which personas it aligns with.
"""
    try:
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        completion = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=300,
        )
        explanation = completion.output_text
        return jsonify({"explanation": explanation, "profile": profile.to_dict()})
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"AI explanation failed: {exc}"}), 502

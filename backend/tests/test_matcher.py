from personas.loader import load_personas
from personas.models import Profile, ExperienceRole
from personas.matcher import match_profiles_to_personas


def test_retiree_profile_matches_retiree_persona():
    personas = load_personas()
    profile = Profile(
        full_name="Mary Thompson",
        headline="Retired CFO looking for reliable income",
        current_title="Retired",
        current_company="",
        industry="Finance",
        about_summary="Self-funded retiree with $2m seeking steady yield; dislikes low term deposits",
        experience=[ExperienceRole(title="CFO", company="ABC Ltd", description="Treasury and risk")],
    )
    matches = match_profiles_to_personas([profile], personas)
    assert matches
    top = matches[0].top_persona_matches[0]
    assert top.persona_id == "retirees-chasing-yield"


def test_smsf_profile_matches_smsf_persona():
    personas = load_personas()
    profile = Profile(
        full_name="Sarah Lee",
        headline="SMSF trustee and accountant",
        current_title="SMSF Trustee",
        current_company="Family Trust",
        location="Sydney, Australia",
        industry="Accounting",
        about_summary="Self managed super fund trustee focused on compliance and audit-ready reporting.",
        experience=[
            ExperienceRole(title="Treasurer", company="Local Association", description="Committee oversight"),
        ],
    )
    matches = match_profiles_to_personas([profile], personas)
    assert matches
    top = matches[0].top_persona_matches[0]
    assert top.persona_id == "smsf-trustee"


def test_family_office_keywords_boost_family_office_persona():
    personas = load_personas()
    profile = Profile(
        full_name="James Carter",
        headline="CIO at multi-family office",
        current_title="Chief Investment Officer",
        current_company="Carter Family Office",
        location="Melbourne, Australia",
        industry="Investment Management",
        about_summary="Leads allocation and due diligence for a multi-generational family office.",
    )
    matches = match_profiles_to_personas([profile], personas)
    assert matches
    top = matches[0].top_persona_matches[0]
    assert top.persona_id == "multi-generational-family-office"


def test_unrelated_software_engineer_scores_near_zero():
    personas = load_personas()
    profile = Profile(
        full_name="Random Engineer",
        headline="Software Engineer",
        current_title="Software Engineer",
        current_company="Tech Co",
        location="Hanoi, Vietnam",
        industry="Software",
        about_summary="Building web apps and devops tooling.",
    )
    matches = match_profiles_to_personas([profile], personas)
    assert matches == []

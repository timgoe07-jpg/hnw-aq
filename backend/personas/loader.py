import logging
import re
from typing import List, Optional
from pathlib import Path

from .models import Persona
from config import Config

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None

# Hardcoded fallback derived from the provided PDF so the app still works if pypdf is unavailable
HARDCODED_PERSONAS = [
    {
        "id": "retirees-chasing-yield",
        "name": "Retirees chasing yield",
        "short_label": "Yield chaser retiree",
        "age_range": "65-75",
        "wealth_range_or_structure": "Self-funded retiree with a paid-off home and $1-2m+ in investable assets",
        "primary_goal": "Income replacement in retirement without eroding capital",
        "key_concern": "Bank term deposit rates too low to fund lifestyle; worried about inflation reducing spending power",
        "investment_behaviour": "Conservative but willing to step up risk for dependable yield; prefers products with regular income distribution",
        "why_private_credit_appeals": [
            "Steady yield of 8-10% vs. 1-3% in bank cash/deposits",
            "Regular monthly or quarterly distributions fit cash flow needs",
            "Comforted by secured lending against tangible collateral (e.g., property or business assets)",
        ],
        "raw_text": "INVESTOR PERSONA 1: THE YIELD CHASER RETIREE | Age: 65-75 | Status: Self-funded retiree with a paid-off home and $1-2m+ in investable assets | Primary Goal: Income replacement in retirement without eroding capital | Key Concern: Bank term deposit rates too low to fund lifestyle; worried about inflation reducing spending power | Investment Behaviour: Conservative but willing to step up risk in exchange for dependable yield, prefers products with regular income distribution | Why Private Credit Appeals: Steady yield of 8-10% vs. 1-3% in bank cash/deposits; Regular monthly or quarterly distributions fit cash flow needs; Comforted by secured lending against tangible collateral.",
    },
    {
        "id": "self-directed-hnw",
        "name": "Self directed HNW",
        "short_label": "Entrepreneurial HNW",
        "age_range": "40-60",
        "wealth_range_or_structure": "Net Worth: $10m+ from sale of a business, property development or ongoing entrepreneurial ventures",
        "primary_goal": "Capital diversification and exposure outside equities/property",
        "key_concern": "Doesn't want idle cash; wary of volatility in listed markets",
        "investment_behaviour": "Opportunistic, values access to non-mainstream assets; likes solutions that match entrepreneurial mindset",
        "why_private_credit_appeals": [
            "Attracted to private pooled deals that feel exclusive and tangible",
            "Sees credit opportunities as similar to how they themselves lent/invested in their business growth",
            "Interested in co-investments and larger allocation strategic partnerships",
        ],
        "raw_text": "INVESTOR PERSONA 2: THE SELF DIRECTED HIGH NET WORTH | Age: 40-60 | Net Worth: $10m+ from sale of a business, property development or ongoing entrepreneurial ventures | Primary Goal: Capital diversification and exposure outside equities/property | Key Concern: Doesn't want idle cash; wary of volatility in listed markets | Investment Behaviour: Opportunistic, values access to non-mainstream assets, likes solutions that match entrepreneurial mindset | Why Private Credit Appeals: Attracted to private pooled deals that feel exclusive and tangible; Sees credit opportunities as similar to their own growth; Interested in co-investments and larger allocation strategic partnerships.",
    },
    {
        "id": "smsf-trustee",
        "name": "SMSF Investor / SMSF Trustee",
        "short_label": "SMSF Trustee",
        "age_range": "45-65",
        "wealth_range_or_structure": "Family Self-Managed Super Fund of $2-5m",
        "primary_goal": "Long-term capital security with above-inflation returns",
        "key_concern": "Compliance and ensuring SMSF trustees meet fiduciary duties",
        "investment_behaviour": "Cautious; works via advisers or accountants; likes products substantiated with proper reporting",
        "why_private_credit_appeals": [
            "Diversifies SMSF assets away from equities/property",
            "Consistent income supports pension-phase obligations",
            "Detailed quarterly reporting provides audit trail for SMSF compliance",
        ],
        "raw_text": "INVESTOR PERSONA 3: THE SMSF TRUSTEE | Age: 45-65 | Structure: Family Self-Managed Super Fund of $2-5m | Primary Goal: Long-term capital security with above-inflation returns | Key Concern: Compliance and ensuring SMSF trustees meet fiduciary duties | Investment Behaviour: Cautious; works via financial advisers or accountants; likes products that can be substantiated with proper reporting | Why Private Credit Appeals: Diversifies SMSF assets away from equities/property; Consistent income supports pension-phase obligations; Detailed quarterly reporting provides audit trail.",
    },
    {
        "id": "community-club-treasurer",
        "name": "Community & Club Treasurer",
        "short_label": "Club Treasurer",
        "age_range": "N/A (entity)",
        "wealth_range_or_structure": "Fund Base: $2-15m cash reserves earmarked for future long-term infrastructure (clubhouse/practice facilities etc.)",
        "primary_goal": "Preserve capital until major project requires deployment, while generating meaningful income in the interim",
        "key_concern": "Must demonstrate fiduciary prudence and transparency to members/stakeholders",
        "investment_behaviour": "Conservative; prefers lower-risk secured opportunities with liquidity options",
        "why_private_credit_appeals": [
            "Generates higher returns vs traditional bank deposits to fund future projects",
            "Emphasis on secured lending provides confidence that capital is preserved",
            "Regular reporting supports accountability to committees/boards",
        ],
        "raw_text": "INVESTOR PERSONA 4: COMMUNITY & CLUB TREASURER | Entity: Local sports club, professional association, charity, or not-for-profit | Fund Base: $2-15m cash reserves earmarked for future long-term infrastructure | Primary Goal: Preserve capital until a major project requires deployment while generating income | Key Concern: Must demonstrate fiduciary prudence and transparency | Investment Behaviour: Conservative, prefers secured opportunities with liquidity | Why Private Credit Appeals: Generates higher returns vs bank deposits; Secured lending provides confidence; Regular reporting supports accountability.",
    },
    {
        "id": "multi-generational-family-office",
        "name": "Multi-generational Family Office",
        "short_label": "Family Office",
        "age_range": "Multi-generational",
        "wealth_range_or_structure": "Family office with $50m+ in FUM, overseeing multiple entities and generations",
        "primary_goal": "Consistent risk-adjusted returns with diversification",
        "key_concern": "Balancing yield with capital stability across generations",
        "investment_behaviour": "Sophisticated allocator; conducts thorough due diligence; seeks to place $5-10m per manager across diversified mandates",
        "why_private_credit_appeals": [
            "Fits into alternative assets sleeve with attractive yield vs. volatility in equities",
            "Provides regular income to fund family distributions and philanthropy",
            "Interest in direct engagement with fund managers and potential bespoke mandates",
        ],
        "raw_text": "INVESTOR PERSONA 5: THE MULTI-GENERATIONAL FAMILY OFFICE | Structure: Family office with $50m+ in FUM | Primary Goal: Consistent risk-adjusted returns with diversification | Key Concern: Balancing yield with capital stability across generations | Investment Behaviour: Sophisticated allocator, thorough due diligence, seeks to place $5-10m per manager across diversified mandates | Why Private Credit Appeals: Fits alternative assets sleeve with attractive yield; Provides regular income to fund distributions and philanthropy; Interested in direct engagement and bespoke mandates.",
    },
]


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_sections(text: str, headings: List[str]) -> dict:
    sections = {}
    lower_text = text.lower()
    for heading in headings:
        idx = lower_text.find(heading.lower())
        if idx == -1:
            continue
        next_indices = [
            lower_text.find(h.lower())
            for h in headings
            if h != heading and lower_text.find(h.lower()) > idx
        ]
        end = min(next_indices) if next_indices else len(text)
        sections[heading] = _clean_text(text[idx:end])
    return sections


def _from_fallback(sections: Optional[dict] = None) -> List[Persona]:
    personas = []
    for persona_data in HARDCODED_PERSONAS:
        raw = persona_data.get("raw_text", "")
        if sections:
            for heading, section_text in sections.items():
                if persona_data["name"].lower() in heading.lower() or persona_data["short_label"].lower() in section_text.lower():
                    raw = section_text
                    break
        data = dict(persona_data)
        data["raw_text"] = raw
        personas.append(Persona(**data))
    return personas


def load_personas(pdf_path: Optional[str] = None) -> List[Persona]:
    pdf_path = pdf_path or Config.PDF_PATH
    text = ""
    if PdfReader and Path(pdf_path).exists():
        try:
            reader = PdfReader(pdf_path)
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages)
        except Exception as exc:  # pragma: no cover
            logger.warning("Falling back to embedded personas: %s", exc)
    if text:
        headings = [
            "INVESTOR PERSONA 1: THE YIELD CHASER RETIREE",
            "INVESTOR PERSONA 2: THE SELF DIRECTED HIGH NET WORTH",
            "INVESTOR PERSONA 3: THE SMSF TRUSTEE",
            "INVESTOR PERSONA 4: COMMUNITY & CLUB TREASURER",
            "INVESTOR PERSONA 5: THE MULTI-GENERATIONAL FAMILY OFFICE",
        ]
        sections = _extract_sections(text, headings)
        personas = _from_fallback(sections)
    else:
        personas = _from_fallback()
    return personas

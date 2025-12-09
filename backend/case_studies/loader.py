from typing import List
from .models import CaseStudy

# Hardcoded case studies extracted from the provided PDF for offline use.
HARDCODED_CASE_STUDIES = [
    CaseStudy(
        id="cs-retiree-yield",
        title="Case Study 1: Margaret, Yield-Chasing Retiree",
        persona_id="retirees-chasing-yield",
        persona_name="Retirees chasing yield",
        problem=(
            "Margaret is a 68-year-old self-funded retiree with $1.5m in investable assets and a paid-off home. "
            "Term deposit rates are too low to fund her lifestyle and inflation is eroding her purchasing power."
        ),
        why_it_matters=(
            "She needs dependable income without eroding capital, and wants confidence that her money is secured against real assets."
        ),
        solution=(
            "Allocates $500k into Capspace Private Credit Fund, selecting monthly distribution share class to align with her cash flow needs."
        ),
        outcome=(
            "Receives regular distributions that meaningfully outpace term deposits while maintaining capital stability via secured lending."
        ),
        capspace_angle=(
            "Capspace provides steady yield (8-10% p.a.) with secured, transparent lending—matching a retiree’s need for income plus capital preservation."
        ),
    ),
    CaseStudy(
        id="cs-james-entrepreneur",
        title="Case Study 2: James, the Entrepreneurial High Net Worth",
        persona_id="self-directed-hnw",
        persona_name="Self directed HNW",
        problem=(
            "James recently sold his technology company for $20m. He already has exposure to equities and "
            "commercial property, but much of his portfolio feels locked up or volatile. He has $5m in cash reserves waiting to be allocated."
        ),
        why_it_matters=(
            "He wants to stay entrepreneurial with his wealth—investments should feel strategic, opportunistic, and meaningful, not just sitting in the bank."
        ),
        solution=(
            "James invests $3m into Capspace Private Credit Fund because it provides direct access to private business "
            "lending opportunities, at scale, with institutional-quality oversight."
        ),
        outcome=(
            "James feels engaged in an asset class that reflects how he once grew his own business—with access to tangible credit opportunities "
            "and the ability to generate consistent yield."
        ),
        capspace_angle=(
            "Capspace appeals to high net worth investors by offering yield, diversification, and pooled lending assets that feel private and entrepreneurial."
        ),
    ),
    CaseStudy(
        id="cs-richard-anna-smsf",
        title="Case Study 3: Richard & Anna, SMSF Trustees",
        persona_id="smsf-trustee",
        persona_name="SMSF Investor / SMSF Trustee",
        problem=(
            "Richard and Anna manage their $3m self-managed superannuation fund concentrated in Australian equities and investment properties. "
            "Market volatility and compliance obligations make them wary of overexposure and audit risk."
        ),
        why_it_matters=(
            "They want diversification to protect long-term wealth and ensure stable income during pension phase, while satisfying SMSF reporting requirements."
        ),
        solution=(
            "They direct 20% of their SMSF portfolio into Capspace Private Credit Fund, which provides detailed and transparent reporting, "
            "monthly distributions, and visibility over assets backing the loan facilities."
        ),
        outcome=(
            "Their SMSF achieves a more balanced risk/return profile, with capitalised monthly yields supporting retirement objectives and simplified audit processes."
        ),
        capspace_angle=(
            "Capspace helps SMSF trustees gain exposure to private markets with transparency, structured reporting, and capital stability."
        ),
    ),
    CaseStudy(
        id="cs-emerald-golf-club",
        title="Case Study 4: The Emerald Golf Club",
        persona_id="community-club-treasurer",
        persona_name="Community & Club Treasurer",
        problem=(
            "Emerald Golf Club holds $10m in cash raised for a future clubhouse redevelopment. Funds sit in low-yield term deposits; "
            "the committee wants to preserve capital while maximising income for future projects."
        ),
        why_it_matters=(
            "Members expect prudent management of club funds. Idle cash means lost opportunities; the committee needs to balance liquidity with growth."
        ),
        solution=(
            "The club allocates $5m into Capspace Private Credit Fund, generating strong yield (8%+ p.a.) while preserving capital through secured SME loans."
        ),
        outcome=(
            "After three years, the club has generated over $1m in income, accelerating financing for the new clubhouse without drawing down reserves."
        ),
        capspace_angle=(
            "Capspace balances fiduciary responsibility, transparency, and higher-yield outcomes compared to leaving money in the bank."
        ),
    ),
    CaseStudy(
        id="cs-hamilton-family-office",
        title="Case Study 5: The Hamilton Family Office",
        persona_id="multi-generational-family-office",
        persona_name="Multi-generational Family Office",
        problem=(
            "The Hamilton Family Office manages $100m across three generations and seeks durable, consistent return streams. "
            "Public markets feel unpredictable; they want alternatives with stability."
        ),
        why_it_matters=(
            "The family requires investments that align with governance—high yield, transparent reporting, and diversification beyond property and listed equities."
        ),
        solution=(
            "Capspace Private Credit Fund becomes part of their alternatives sleeve. With a $10m allocation, they secure steady annual yield, "
            "detailed reporting, and access to manager insights on private debt markets."
        ),
        outcome=(
            "They create a stable income stream to support philanthropy and generational distributions while maintaining a well-diversified portfolio."
        ),
        capspace_angle=(
            "Capspace delivers institutional-quality private credit exposure, attractive yield, and deep reporting for long-term, multi-generational wealth goals."
        ),
    ),
]


def load_case_studies() -> List[CaseStudy]:
    # For now, return the embedded set; a PDF parser could be added similarly to personas if needed.
    return HARDCODED_CASE_STUDIES

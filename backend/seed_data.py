import random
from typing import List

from models import Investor, Loan, SessionLocal, init_db

random.seed(42)


def _generate_investors(count: int) -> List[Investor]:
    investors: List[Investor] = []
    risk_choices = ["low", "medium", "high"]
    for i in range(count):
        name = f"Investor {i + 1}"
        age = random.randint(28, 72)
        aum = round(random.lognormvariate(13, 0.4), 2)  # skew toward larger AUM
        risk_tolerance = random.choices(risk_choices, weights=[0.3, 0.5, 0.2])[0]
        engagement_score = round(random.gauss(65, 15), 1)
        engagement_score = max(15, min(engagement_score, 100))
        email_open_rate = round(random.uniform(0.12, 0.95), 2)
        call_frequency = round(random.uniform(0.5, 10), 1)
        inactivity_days = random.randint(0, 120)
        redemption_intent = random.random() < 0.15
        distribution_yield = round(random.uniform(0.04, 0.12), 3)
        meetings_last_quarter = random.randint(0, 6)

        churn_prob = 0.08
        churn_prob += max(0, (55 - engagement_score) / 140)
        churn_prob += max(0, (0.35 - email_open_rate) * 0.8)
        churn_prob += 0.1 if risk_tolerance == "low" else 0
        churn_prob += 0.05 if call_frequency < 2 else 0
        churn_prob += 0.08 if inactivity_days > 60 else 0
        churn_prob += 0.1 if redemption_intent else 0
        churn_prob = max(0, min(churn_prob, 0.9))
        churn_label = 1 if random.random() < churn_prob else 0

        investors.append(
            Investor(
                name=name,
                age=age,
                aum=aum,
                risk_tolerance=risk_tolerance,
                engagement_score=engagement_score,
                email_open_rate=email_open_rate,
                call_frequency=call_frequency,
                inactivity_days=inactivity_days,
                redemption_intent=redemption_intent,
                distribution_yield=distribution_yield,
                meetings_last_quarter=meetings_last_quarter,
                churn_label=churn_label,
                churn_risk_score=None,
            )
        )
    return investors


def _generate_loans(investors: List[Investor], count: int) -> List[Loan]:
    sectors = ["property", "infrastructure", "healthcare", "technology", "hospitality", "energy", "logistics"]
    loans: List[Loan] = []
    for _ in range(count):
        investor = random.choice(investors)
        amount = round(random.uniform(150_000, 3_500_000), 2)
        ltv_ratio = round(random.uniform(0.3, 0.95), 2)
        term_months = random.choice([12, 18, 24, 36, 48, 60, 72, 84])
        sector = random.choice(sectors)
        arrears_flag = random.random() < 0.2
        dscr = round(random.uniform(0.8, 2.2), 2)
        covenants_flag = random.random() < 0.12
        collateral_score = round(random.uniform(0.4, 0.95), 2)

        sector_risk = {
            "hospitality": 0.08,
            "property": 0.06,
            "logistics": 0.04,
            "technology": 0.03,
            "infrastructure": 0.025,
            "energy": 0.05,
            "healthcare": 0.02,
        }.get(sector, 0.03)

        default_prob = 0.04 + sector_risk
        default_prob += max(0, ltv_ratio - 0.6) * 0.75
        default_prob += 0.15 if arrears_flag else 0
        default_prob += 0.08 if amount > 2_000_000 else 0
        default_prob += 0.03 if term_months > 60 else 0
        default_prob += 0.12 if dscr < 1.0 else 0
        default_prob += 0.1 if covenants_flag else 0
        default_prob += max(0, 0.6 - collateral_score) * 0.3
        default_prob = max(0, min(default_prob, 0.95))
        default_label = 1 if random.random() < default_prob else 0

        loans.append(
            Loan(
                investor_id=investor.id,
                amount=amount,
                ltv_ratio=ltv_ratio,
                term_months=term_months,
                sector=sector,
                arrears_flag=arrears_flag,
                dscr=dscr,
                covenants_flag=covenants_flag,
                collateral_score=collateral_score,
                default_label=default_label,
                default_risk_score=None,
            )
        )
    return loans


def seed():
    init_db()
    session = SessionLocal()
    try:
        session.query(Loan).delete()
        session.query(Investor).delete()
        session.commit()

        investors = _generate_investors(80)
        session.add_all(investors)
        session.commit()

        loans = _generate_loans(investors, 150)
        session.add_all(loans)
        session.commit()
        print(f"Seeded {len(investors)} investors and {len(loans)} loans.")
    finally:
        session.close()


if __name__ == "__main__":
    seed()

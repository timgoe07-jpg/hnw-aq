from typing import List, Dict, Any
from datetime import datetime, timezone

from models import Investor, Loan, RiskAlert
from ml_models import risk_bucket


# Investor playbook rules keyed off 6m horizon by default
INVESTOR_PLAYBOOK_RULES = [
    {
        "id": "call_low_engagement_high_risk",
        "label": "Call low-engagement, high-risk investors",
        "condition": lambda inv: (inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)) == "High"
        and inv.engagement_score < 30,
        "action": "Schedule RM call within 3 days to re-engage and understand drivers of disengagement.",
    },
    {
        "id": "monitor_high_aum_medium_risk",
        "label": "Monitor high-AUM medium-risk investors",
        "condition": lambda inv: (inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)) == "Medium"
        and inv.aum > 5_000_000,
        "action": "Add to weekly review; propose liquidity check-in and proactive portfolio review.",
    },
    {
        "id": "email_campaign_low_touch",
        "label": "Email campaign for low-touch investors",
        "condition": lambda inv: inv.email_open_rate < 0.25 and inv.call_frequency < 2,
        "action": "Trigger targeted nurture emails and follow-up task for the RM team.",
    },
]


LOAN_PLAYBOOK_RULES = [
    {
        "id": "tighten_covenants_high_default",
        "label": "Tighten covenants on high-risk loans",
        "condition": lambda loan: (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High"
        and loan.dscr < 1.2,
        "action": "Review covenants and consider tightening LVR or adding additional security.",
    },
    {
        "id": "arrears_escalation",
        "label": "Escalate arrears for high-risk loans",
        "condition": lambda loan: (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High"
        and loan.arrears_flag,
        "action": "Trigger arrears management workflow and notify credit team.",
    },
    {
        "id": "sector_focus_property",
        "label": "Review property sector high-risk loans",
        "condition": lambda loan: loan.sector.lower() == "property"
        and (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High",
        "action": "Prioritise property loans for closer monitoring and updated collateral checks.",
    },
]


def _days_open(alert: RiskAlert | None) -> int:
    if not alert or not alert.first_high_at:
        return 0
    try:
        start = datetime.fromisoformat(alert.first_high_at)
    except Exception:
        return 0
    return max(0, (datetime.now(timezone.utc) - start).days)


def evaluate_playbooks(investors: List[Investor], loans: List[Loan], alerts: Dict[tuple, RiskAlert] | None = None) -> Dict[str, List[Dict[str, Any]]]:
    alerts = alerts or {}
    investor_actions: List[Dict[str, Any]] = []
    for rule in INVESTOR_PLAYBOOK_RULES:
        matched = [inv for inv in investors if rule["condition"](inv)]
        if matched:
            sla_breach = 0
            total_days = 0
            counted = 0
            for inv in matched:
                alert = alerts.get(("investor", inv.id))
                if alert:
                    days = _days_open(alert)
                    total_days += days
                    counted += 1
                    if days > (alert.sla_days or 5):
                        sla_breach += 1
            investor_actions.append(
                {
                    "rule_id": rule["id"],
                    "label": rule["label"],
                    "action": rule["action"],
                    "count": len(matched),
                    "entity_ids": [inv.id for inv in matched[:10]],
                    "sla_breach": sla_breach,
                    "avg_days_open": (total_days / counted) if counted else 0.0,
                    "examples": [{"id": inv.id, "name": inv.name, "bucket": inv.churn_risk_bucket_6m} for inv in matched[:5]],
                }
            )

    loan_actions: List[Dict[str, Any]] = []
    for rule in LOAN_PLAYBOOK_RULES:
        matched = [loan for loan in loans if rule["condition"](loan)]
        if matched:
            sla_breach = 0
            total_days = 0
            counted = 0
            for loan in matched:
                alert = alerts.get(("loan", loan.id))
                if alert:
                    days = _days_open(alert)
                    total_days += days
                    counted += 1
                    if days > (alert.sla_days or 5):
                        sla_breach += 1
            loan_actions.append(
                {
                    "rule_id": rule["id"],
                    "label": rule["label"],
                    "action": rule["action"],
                    "count": len(matched),
                    "entity_ids": [loan.id for loan in matched[:10]],
                    "sla_breach": sla_breach,
                    "avg_days_open": (total_days / counted) if counted else 0.0,
                    "examples": [{"id": loan.id, "sector": loan.sector, "bucket": loan.default_risk_bucket_6m} for loan in matched[:5]],
                }
            )

    return {"investor_playbooks": investor_actions, "loan_playbooks": loan_actions}

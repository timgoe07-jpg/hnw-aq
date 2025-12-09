import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify

from models import SessionLocal, Investor, Loan
from ml_models import risk_bucket

bp_monitoring = Blueprint("monitoring", __name__, url_prefix="/api/monitoring")
BASE_DIR = Path(__file__).resolve().parent.parent


def _bucket_proportions(items, score_attr: str, bucket_attr: str):
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for item in items:
        bucket = getattr(item, bucket_attr, None)
        if not bucket:
            bucket = risk_bucket(getattr(item, score_attr, 0) or 0)
        counts[bucket] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _psi(expected: dict, actual: dict) -> float:
    """Simple population stability index between expected and actual bucket proportions."""
    psi = 0.0
    for key in expected.keys():
        exp = expected.get(key, 1e-6)
        act = actual.get(key, 1e-6)
        if exp == 0 or act == 0:
            continue
        psi += (act - exp) * (np.log(act / exp))
    return float(psi)


@bp_monitoring.route("/drift", methods=["GET"])
def drift():
    metrics_path = BASE_DIR / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    baseline = metrics.get("baseline_distributions", {})
    baseline_feat = metrics.get("baseline_feature_stats", {})
    ci = metrics.get("bootstrap_ci", {})

    with SessionLocal() as session:
        investors = session.query(Investor).all()
        loans = session.query(Loan).all()

    reports = []
    # Only 6m for now
    base_churn = baseline.get("investor_churn_6m", {}).get("bucket_proportions", {"Low": 1, "Medium": 0, "High": 0})
    curr_churn = _bucket_proportions(investors, "churn_risk_score_6m", "churn_risk_bucket_6m")
    psi_churn = _psi(base_churn, curr_churn)
    status_churn = "Stable" if psi_churn < 0.1 else "Moderate" if psi_churn < 0.25 else "High"

    base_default = baseline.get("loan_default_6m", {}).get("bucket_proportions", {"Low": 1, "Medium": 0, "High": 0})
    curr_default = _bucket_proportions(loans, "default_risk_score_6m", "default_risk_bucket_6m")
    psi_default = _psi(base_default, curr_default)
    status_default = "Stable" if psi_default < 0.1 else "Moderate" if psi_default < 0.25 else "High"

    reports.append(
        {"model": "investor_churn_6m", "bucket_psi": psi_churn, "status": status_churn, "current": curr_churn}
    )
    reports.append(
        {"model": "loan_default_6m", "bucket_psi": psi_default, "status": status_default, "current": curr_default}
    )

    # Feature-level drift using z-score shifts vs baseline mean/std
    feature_drifts = []
    if baseline_feat.get("investor"):
        inv_df = pd.DataFrame([{
            "churn_risk_score_6m": getattr(i, "churn_risk_score_6m", i.churn_risk_score or 0),
            "engagement_score": i.engagement_score,
            "email_open_rate": i.email_open_rate,
            "call_frequency": i.call_frequency,
            "inactivity_days": i.inactivity_days,
            "distribution_yield": i.distribution_yield,
            "meetings_last_quarter": i.meetings_last_quarter,
            "aum": i.aum,
        } for i in investors])
        for feat, stats in baseline_feat["investor"].items():
            if feat not in inv_df.columns:
                continue
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1e-6) or 1e-6
            curr_mean = float(inv_df[feat].mean()) if not inv_df.empty else 0.0
            delta_std = abs(curr_mean - mean) / std
            feature_drifts.append({"name": f"investor::{feat}", "delta_std": float(delta_std)})

    if baseline_feat.get("loan"):
        loan_df = pd.DataFrame([{
            "default_risk_score_6m": getattr(l, "default_risk_score_6m", l.default_risk_score or 0),
            "amount": l.amount,
            "ltv_ratio": l.ltv_ratio,
            "term_months": l.term_months,
            "dscr": l.dscr,
            "collateral_score": l.collateral_score,
        } for l in loans])
        for feat, stats in baseline_feat["loan"].items():
            if feat not in loan_df.columns:
                continue
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1e-6) or 1e-6
            curr_mean = float(loan_df[feat].mean()) if not loan_df.empty else 0.0
            delta_std = abs(curr_mean - mean) / std
            feature_drifts.append({"name": f"loan::{feat}", "delta_std": float(delta_std)})

    feature_drifts = sorted(feature_drifts, key=lambda x: x["delta_std"], reverse=True)[:10]

    return jsonify({"buckets": reports, "feature_drifts": feature_drifts, "ci": ci})

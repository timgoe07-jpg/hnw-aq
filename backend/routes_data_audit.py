from flask import Blueprint, jsonify
from models import SessionLocal, Investor, Loan

bp_data_audit = Blueprint("data_audit", __name__, url_prefix="/api/data")


@bp_data_audit.route("/audit", methods=["GET"])
def audit():
    """Lightweight data readiness audit: missingness, categorical encoding, scaling hints."""
    with SessionLocal() as session:
        investors = session.query(Investor).all()
        loans = session.query(Loan).all()

    def missing_pct(objs, field):
        vals = [getattr(o, field, None) for o in objs]
        missing = sum(1 for v in vals if v is None)
        return (missing / max(len(vals), 1)) * 100

    investor_fields = [
        "age",
        "aum",
        "risk_tolerance",
        "engagement_score",
        "email_open_rate",
        "call_frequency",
        "inactivity_days",
        "redemption_intent",
        "distribution_yield",
        "meetings_last_quarter",
    ]
    loan_fields = [
        "amount",
        "ltv_ratio",
        "term_months",
        "sector",
        "arrears_flag",
        "dscr",
        "covenants_flag",
        "collateral_score",
    ]

    investor_missing = {f: round(missing_pct(investors, f), 2) for f in investor_fields}
    loan_missing = {f: round(missing_pct(loans, f), 2) for f in loan_fields}

    # Missingness by simple segments for quick diagnostics
    def missing_by_segment(objs, field, seg_attr):
        buckets = {}
        for obj in objs:
            seg = getattr(obj, seg_attr, "unknown")
            buckets.setdefault(seg, {"total": 0, "missing": 0})
            buckets[seg]["total"] += 1
            if getattr(obj, field, None) is None:
                buckets[seg]["missing"] += 1
        return {k: round((v["missing"] / v["total"] * 100) if v["total"] else 0, 2) for k, v in buckets.items()}

    investor_missing_by_tol = {f: missing_by_segment(investors, f, "risk_tolerance") for f in investor_fields}
    loan_missing_by_sector = {f: missing_by_segment(loans, f, "sector") for f in loan_fields}

    categorical_to_encode = ["risk_tolerance", "sector"]
    scaling_recommended = ["aum", "amount", "dscr", "ltv_ratio"]
    strategy_recommendation = "Use KNNImputer for MAR-like patterns (e.g., DSCR), median for stable numeric features."
    heatmap = {
        "investor": {"rows": list(investor_missing_by_tol.keys()), "cols": list(next(iter(investor_missing_by_tol.values()), {}).keys()), "values": [list(v.values()) for v in investor_missing_by_tol.values()]},
        "loan": {"rows": list(loan_missing_by_sector.keys()), "cols": list(next(iter(loan_missing_by_sector.values()), {}).keys()), "values": [list(v.values()) for v in loan_missing_by_sector.values()]},
    }

    return jsonify(
        {
          "investor_missing_pct": investor_missing,
          "loan_missing_pct": loan_missing,
          "investor_missing_by_tolerance": investor_missing_by_tol,
          "loan_missing_by_sector": loan_missing_by_sector,
          "categorical_to_encode": categorical_to_encode,
          "scaling_recommended": scaling_recommended,
          "strategy_recommendation": strategy_recommendation,
          "heatmap": heatmap,
        }
    )

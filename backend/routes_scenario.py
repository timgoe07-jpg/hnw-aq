from flask import Blueprint, request, jsonify
import pandas as pd

from models import SessionLocal, Investor, Loan, Intervention
from ml_models import (
    CHURN_MODEL_PATHS,
    DEFAULT_MODEL_PATHS,
    HORIZONS,
    investor_to_features,
    loan_to_features,
    risk_bucket,
    load_or_train_horizon,
    train_churn_model,
    train_default_model,
)

bp_scenario = Blueprint("scenario", __name__, url_prefix="/api/scenario")


@bp_scenario.route("/predict", methods=["POST"])
def scenario_predict():
    payload = request.get_json(force=True) or {}
    entity_type = payload.get("entity_type")
    entity_id = payload.get("entity_id")
    overrides = payload.get("overrides", {}) or {}
    horizon = payload.get("horizon", "6m")
    if horizon not in HORIZONS:
        horizon = "6m"

    with SessionLocal() as session:
        if entity_type == "investor":
            inv = session.get(Investor, entity_id)
            if not inv:
                return jsonify({"error": "Investor not found"}), 404
            base_feats = investor_to_features(inv)
            scenario_feats = {**base_feats, **overrides}
            model = load_or_train_horizon(CHURN_MODEL_PATHS, train_churn_model, session, horizon)
            probs_base = model.predict_proba(pd.DataFrame([base_feats]))[:, 1][0]
            probs_scenario = model.predict_proba(pd.DataFrame([scenario_feats]))[:, 1][0]
            delta = float(probs_scenario - probs_base)
            return jsonify(
                {
                    "entity_id": inv.id,
                    "horizon": horizon,
                    "base": {"probability": float(probs_base), "bucket": risk_bucket(probs_base)},
                    "scenario": {"probability": float(probs_scenario), "bucket": risk_bucket(probs_scenario)},
                    "delta": delta,
                }
            )
        elif entity_type == "loan":
            loan = session.get(Loan, entity_id)
            if not loan:
                return jsonify({"error": "Loan not found"}), 404
            base_feats = loan_to_features(loan)
            scenario_feats = {**base_feats, **overrides}
            model = load_or_train_horizon(DEFAULT_MODEL_PATHS, train_default_model, session, horizon)
            probs_base = model.predict_proba(pd.DataFrame([base_feats]))[:, 1][0]
            probs_scenario = model.predict_proba(pd.DataFrame([scenario_feats]))[:, 1][0]
            delta = float(probs_scenario - probs_base)
            return jsonify(
                {
                    "entity_id": loan.id,
                    "horizon": horizon,
                    "base": {"probability": float(probs_base), "bucket": risk_bucket(probs_base)},
                    "scenario": {"probability": float(probs_scenario), "bucket": risk_bucket(probs_scenario)},
                    "delta": delta,
                }
            )
        else:
            return jsonify({"error": "entity_type must be 'investor' or 'loan'"}), 400


@bp_scenario.route("/portfolio", methods=["POST"])
def portfolio_scenario():
    """
    Simple macro scenario: uplift engagement_score for bottom quartile investors by +10 (capped at 100),
    reduce LTV by 0.05 and add 0.1 to DSCR for loans flagged High risk at the selected horizon.
    Returns base vs scenario risk means and bucket counts.
    """
    payload = request.get_json(force=True) or {}
    horizon = payload.get("horizon", "6m")
    if horizon not in HORIZONS:
        horizon = "6m"
    with SessionLocal() as session:
        investors = session.query(Investor).all()
        loans = session.query(Loan).all()
        if not investors or not loans:
            return jsonify({"error": "No data"}), 400

        inv_model = load_or_train_horizon(CHURN_MODEL_PATHS, train_churn_model, session, horizon)
        loan_model = load_or_train_horizon(DEFAULT_MODEL_PATHS, train_default_model, session, horizon)

        inv_df = pd.DataFrame([{"id": i.id, **investor_to_features(i)} for i in investors]).set_index("id")
        loan_df = pd.DataFrame([{"id": l.id, **loan_to_features(l)} for l in loans]).set_index("id")

        # Apply interventions before calculating base probabilities
        interventions = session.query(Intervention).all()
        for iv in interventions:
            if iv.entity_type == "investor" and iv.entity_id in inv_df.index:
                inv_df.loc[iv.entity_id, "engagement_score"] = float(
                    max(0, min(100, inv_df.loc[iv.entity_id, "engagement_score"] + (iv.engagement_delta or 0)))
                )
                inv_df.loc[iv.entity_id, "inactivity_days"] = float(
                    max(0, inv_df.loc[iv.entity_id, "inactivity_days"] + (iv.inactivity_delta or 0))
                )
            elif iv.entity_type == "loan" and iv.entity_id in loan_df.index:
                loan_df.loc[iv.entity_id, "ltv_ratio"] = float(
                    max(0.2, loan_df.loc[iv.entity_id, "ltv_ratio"] + (iv.ltv_delta or 0))
                )
                loan_df.loc[iv.entity_id, "dscr"] = loan_df.loc[iv.entity_id, "dscr"] + (iv.dscr_delta or 0)

        inv_base_probs = inv_model.predict_proba(inv_df)[:, 1]
        loan_base_probs = loan_model.predict_proba(loan_df)[:, 1]

        # Scenario adjustments
        inv_cut = inv_df["engagement_score"].quantile(0.25)
        inv_df_scenario = inv_df.copy()
        inv_df_scenario.loc[inv_df_scenario["engagement_score"] <= inv_cut, "engagement_score"] += 10
        inv_df_scenario["engagement_score"] = inv_df_scenario["engagement_score"].clip(0, 100)

        loan_df_scenario = loan_df.copy()
        high_risk_mask = loan_base_probs >= 0.66
        loan_df_scenario.loc[high_risk_mask, "ltv_ratio"] = (loan_df_scenario.loc[high_risk_mask, "ltv_ratio"] - 0.05).clip(lower=0.2)
        loan_df_scenario.loc[high_risk_mask, "dscr"] = loan_df_scenario.loc[high_risk_mask, "dscr"] + 0.1

        inv_scenario_probs = inv_model.predict_proba(inv_df_scenario)[:, 1]
        loan_scenario_probs = loan_model.predict_proba(loan_df_scenario)[:, 1]

        def bucket_counts(probs):
            buckets = {"Low": 0, "Medium": 0, "High": 0}
            for p in probs:
                buckets[risk_bucket(float(p))] += 1
            return buckets

        response = {
            "horizon": horizon,
            "investors": {
                "base_avg": float(inv_base_probs.mean()),
                "scenario_avg": float(inv_scenario_probs.mean()),
                "base_buckets": bucket_counts(inv_base_probs),
                "scenario_buckets": bucket_counts(inv_scenario_probs),
            },
            "loans": {
                "base_avg": float(loan_base_probs.mean()),
                "scenario_avg": float(loan_scenario_probs.mean()),
                "base_buckets": bucket_counts(loan_base_probs),
                "scenario_buckets": bucket_counts(loan_scenario_probs),
            },
        }
        return jsonify(response)

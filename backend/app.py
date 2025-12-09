import logging
import os
import random
import html
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List

import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from ml_models import (
    CHURN_MODEL_PATHS,
    DEFAULT_MODEL_PATHS,
    HORIZONS,
    MODEL_FAMILIES,
    explain_investor_risk,
    explain_loan_risk,
    investor_to_features,
    loan_to_features,
    load_family_model,
    load_or_train_model,
    load_or_train_horizon,
    knn_neighbors,
    risk_bucket,
    train_churn_model,
    train_default_model,
)
from playbooks import evaluate_playbooks
from models import Investor, Loan, SessionLocal, init_db, DailySnapshot, RiskAlert, Intervention
from config import Config
from api.personas_routes import bp as personas_bp
from api.profiles_routes import bp as profiles_bp
from api.case_studies_routes import bp as case_studies_bp
from api.match_routes import bp as match_bp
from api.auth_routes import bp as auth_bp
import json
from routes_monitoring import bp_monitoring
from routes_scenario import bp_scenario
from routes_data_audit import bp_data_audit

BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_avg(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _parse_iso(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def segment_mini_report(investors: List[Investor], loans: List[Loan], segment_type: str | None, segment_value: str | None) -> str:
    """Lightweight narrative for a specific segment."""
    if not segment_type or not segment_value:
        return ""
    inv_count = len(investors)
    loan_count = len(loans)
    avg_churn = _safe_avg([inv.churn_risk_score_6m or inv.churn_risk_score or 0 for inv in investors])
    avg_default = _safe_avg([loan.default_risk_score_6m or loan.default_risk_score or 0 for loan in loans])
    high_inv = sum(1 for inv in investors if (inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)) == "High")
    high_loan = sum(1 for loan in loans if (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High")
    return (
        f"Segment [{segment_type}={segment_value}]: {inv_count} investors, {loan_count} loans; "
        f"avg churn {avg_churn:.2f}, avg default {avg_default:.2f}; "
        f"high-risk investors {high_inv}, high-risk loans {high_loan}."
    )


def _top_drivers(features: Dict, medians: Dict, problem: str) -> List[Dict]:
    """Compare instance to portfolio medians to highlight top drivers."""
    important = {
        "churn": ["engagement_score", "email_open_rate", "call_frequency", "inactivity_days", "distribution_yield", "meetings_last_quarter", "aum"],
        "default": ["ltv_ratio", "dscr", "amount", "collateral_score", "term_months"],
    }.get(problem, [])
    rows = []
    for feat in important:
        val = features.get(feat, 0)
        med = medians.get(feat, 0) or 1e-6
        diff = val - med
        pct = diff / med if med else 0
        rows.append({"feature": feat, "value": val, "portfolio_median": med, "delta_pct": pct})
    rows = sorted(rows, key=lambda x: abs(x["delta_pct"]), reverse=True)
    return rows[:5]


def _aum_band(aum: float) -> str:
    if aum < 1_000_000:
        return "<1M"
    if aum < 2_000_000:
        return "1-2M"
    if aum < 5_000_000:
        return "2-5M"
    return "5M+"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    app.secret_key = app.config.get("SECRET_KEY", "dev")
    allowed_origins = app.config.get("CORS_ORIGINS", "http://localhost:4200,http://127.0.0.1:4200")
    origins_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origins_list}}, supports_credentials=True)
    init_db()
    app.register_blueprint(personas_bp)
    app.register_blueprint(profiles_bp)
    app.register_blueprint(case_studies_bp)
    app.register_blueprint(match_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(bp_monitoring)
    app.register_blueprint(bp_scenario)
    app.register_blueprint(bp_data_audit)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/metrics", methods=["GET"])
    def metrics():
        metrics_path = BASE_DIR / "models" / "metrics.json"
        if not metrics_path.exists():
            return jsonify({"error": "No metrics available. Run train_models.py first."}), 404
        return jsonify(json.loads(metrics_path.read_text()))

    @app.route("/api/models/metrics", methods=["GET"])
    def models_metrics():
        metrics_path = BASE_DIR / "models" / "metrics.json"
        if not metrics_path.exists():
            return jsonify({"error": "No metrics available. Run train_models.py first."}), 404
        return jsonify(json.loads(metrics_path.read_text()))

    @app.route("/api/models/thresholds", methods=["GET"])
    def model_thresholds():
        """Return threshold grid for a given problem (churn/default), horizon, and family."""
        problem = request.args.get("problem", "churn")
        horizon = request.args.get("horizon", "6m")
        family = request.args.get("family", "ensemble")
        metrics_path = BASE_DIR / "models" / "metrics.json"
        if not metrics_path.exists():
            return jsonify({"error": "No metrics available. Run train_models.py first."}), 404
        metrics = json.loads(metrics_path.read_text())
        thresholds = metrics.get("thresholds", {}).get(problem, {}).get(horizon, [])
        return jsonify({"problem": problem, "horizon": horizon, "family": family, "thresholds": thresholds})

    @app.route("/api/ai/explain", methods=["POST"])
    def ai_explain():
        payload = request.json or {}
        question = payload.get("question") or "Provide a concise risk and engagement summary."
        focus = payload.get("focus")
        history = payload.get("history") or []
        context = payload.get("context") or {}
        with SessionLocal() as session:
            investors = session.query(Investor).all()
            loans = session.query(Loan).all()
            summary = compute_summary(investors, loans) if investors and loans else {}

        metrics_path = BASE_DIR / "models" / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        answer = _ai_explain(question, summary, metrics, focus, history, context)
        return jsonify({"answer": answer, "summary_used": summary, "has_metrics": bool(metrics)})

    @app.route("/api/analytics/overview", methods=["GET"])
    def analytics_overview():
        with SessionLocal() as session:
            investors = session.query(Investor).all()
            loans = session.query(Loan).all()
            if not investors or not loans:
                return jsonify({"error": "No data. Seed first."}), 400

            # Buckets
            churn_buckets = {"Low": 0, "Medium": 0, "High": 0}
            churn_buckets_by_horizon = {h: {"Low": 0, "Medium": 0, "High": 0} for h in HORIZONS}
            for inv in investors:
                bucket6 = inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)
                churn_buckets[bucket6] += 1
                for h in HORIZONS:
                    bucket = getattr(inv, f"churn_risk_bucket_{h}", None) or risk_bucket(getattr(inv, f"churn_risk_score_{h}", 0) or 0)
                    churn_buckets_by_horizon[h][bucket] += 1
            default_buckets = {"Low": 0, "Medium": 0, "High": 0}
            default_buckets_by_horizon = {h: {"Low": 0, "Medium": 0, "High": 0} for h in HORIZONS}
            for loan in loans:
                bucket6 = loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)
                default_buckets[bucket6] += 1
                for h in HORIZONS:
                    bucket = getattr(loan, f"default_risk_bucket_{h}", None) or risk_bucket(getattr(loan, f"default_risk_score_{h}", 0) or 0)
                    default_buckets_by_horizon[h][bucket] += 1

            # Sector exposure
            sector_exposure = {}
            for loan in loans:
                sector_exposure.setdefault(loan.sector, {"exposure": 0.0, "count": 0})
                sector_exposure[loan.sector]["exposure"] += loan.amount
                sector_exposure[loan.sector]["count"] += 1

            # AUM buckets
            aum_buckets = {"<1M": 0, "1-2M": 0, "2-5M": 0, "5M+": 0}
            for inv in investors:
                if inv.aum < 1_000_000:
                    aum_buckets["<1M"] += 1
                elif inv.aum < 2_000_000:
                    aum_buckets["1-2M"] += 1
                elif inv.aum < 5_000_000:
                    aum_buckets["2-5M"] += 1
                else:
                    aum_buckets["5M+"] += 1

            # Engagement distributions
            engagement_scores = [inv.engagement_score for inv in investors]
            email_rates = [inv.email_open_rate for inv in investors]
            yields = [inv.distribution_yield for inv in investors]
            dscrs = [loan.dscr for loan in loans]
            engagement_summary = {
                "avg": sum(engagement_scores) / len(engagement_scores),
                "p25": float(pd.Series(engagement_scores).quantile(0.25)),
                "p50": float(pd.Series(engagement_scores).quantile(0.5)),
                "p75": float(pd.Series(engagement_scores).quantile(0.75)),
            }
            email_summary = {
                "avg": sum(email_rates) / len(email_rates),
                "p25": float(pd.Series(email_rates).quantile(0.25)),
                "p50": float(pd.Series(email_rates).quantile(0.5)),
                "p75": float(pd.Series(email_rates).quantile(0.75)),
            }
            yield_summary = {
                "avg": sum(yields) / len(yields),
                "p25": float(pd.Series(yields).quantile(0.25)),
                "p50": float(pd.Series(yields).quantile(0.5)),
                "p75": float(pd.Series(yields).quantile(0.75)),
            }
            dscr_summary = {
                "avg": sum(dscrs) / len(dscrs),
                "p25": float(pd.Series(dscrs).quantile(0.25)),
                "p50": float(pd.Series(dscrs).quantile(0.5)),
                "p75": float(pd.Series(dscrs).quantile(0.75)),
            }
            engagement_trend = _random_walk(engagement_summary["p50"] / 100.0, 14, noise=0.03)
            dscr_trend = _random_walk(max(0.5, dscr_summary["avg"] / 2), 14, noise=0.05, min_val=0.4, max_val=2.5)
            # Cohort breakdowns
            churn_by_risk_tolerance = {}
            for inv in investors:
                bucket = inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)
                churn_by_risk_tolerance.setdefault(inv.risk_tolerance, {"Low": 0, "Medium": 0, "High": 0})
                churn_by_risk_tolerance[inv.risk_tolerance][bucket] += 1
            default_by_sector = {}
            for loan in loans:
                bucket = loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)
                default_by_sector.setdefault(loan.sector, {"Low": 0, "Medium": 0, "High": 0})
                default_by_sector[loan.sector][bucket] += 1
            churn_by_aum = {}
            for inv in investors:
                band = _aum_band(inv.aum)
                bucket = inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)
                churn_by_aum.setdefault(band, {"Low": 0, "Medium": 0, "High": 0})
                churn_by_aum[band][bucket] += 1

            # Cohort-level risk averages (portfolio view)
            churn_avg_by_risk = {}
            for inv in investors:
                score = inv.churn_risk_score_6m or inv.churn_risk_score or 0.0
                churn_avg_by_risk.setdefault(inv.risk_tolerance, []).append(score)
            churn_avg_by_risk = {k: _safe_avg(v) for k, v in churn_avg_by_risk.items()}

            churn_avg_by_aum = {}
            for inv in investors:
                band = _aum_band(inv.aum)
                score = inv.churn_risk_score_6m or inv.churn_risk_score or 0.0
                churn_avg_by_aum.setdefault(band, []).append(score)
            churn_avg_by_aum = {k: _safe_avg(v) for k, v in churn_avg_by_aum.items()}

            default_avg_by_sector = {}
            for loan in loans:
                score = loan.default_risk_score_6m or loan.default_risk_score or 0.0
                default_avg_by_sector.setdefault(loan.sector, []).append(score)
            default_avg_by_sector = {k: _safe_avg(v) for k, v in default_avg_by_sector.items()}

            segment_scores = {
                "churn": {
                    "risk_tolerance": {k: {"avg": v, "bucket": risk_bucket(v)} for k, v in churn_avg_by_risk.items()},
                    "aum_band": {k: {"avg": v, "bucket": risk_bucket(v)} for k, v in churn_avg_by_aum.items()},
                },
                "default": {
                    "sector": {k: {"avg": v, "bucket": risk_bucket(v)} for k, v in default_avg_by_sector.items()},
                },
            }

            # Heatmap-style matrices (sector vs risk bucket counts; aum band vs tolerance averages)
            sector_bucket_matrix = {}
            for loan in loans:
                bucket = loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)
                sector_bucket_matrix.setdefault(loan.sector, {"Low": 0, "Medium": 0, "High": 0})
                sector_bucket_matrix[loan.sector][bucket] += 1

            aum_tolerance_matrix = {}
            for inv in investors:
                band = _aum_band(inv.aum)
                tol = inv.risk_tolerance
                score = inv.churn_risk_score_6m or inv.churn_risk_score or 0.0
                aum_tolerance_matrix.setdefault(band, {})
                aum_tolerance_matrix[band].setdefault(tol, [])
                aum_tolerance_matrix[band][tol].append(score)
            # collapse to averages
            for band, tol_map in aum_tolerance_matrix.items():
                for tol, scores in tol_map.items():
                    tol_map[tol] = _safe_avg(scores)

            return jsonify(
                {
                    "churn_buckets": churn_buckets,
                    "default_buckets": default_buckets,
                    "churn_buckets_by_horizon": churn_buckets_by_horizon,
                    "default_buckets_by_horizon": default_buckets_by_horizon,
                    "sector_exposure": sector_exposure,
                    "aum_buckets": aum_buckets,
                    "engagement": engagement_summary,
                    "email_open": email_summary,
                    "distribution_yield": yield_summary,
                    "dscr": dscr_summary,
                    "engagement_trend": engagement_trend,
                    "dscr_trend": dscr_trend,
                    "churn_by_risk_tolerance": churn_by_risk_tolerance,
                    "default_by_sector": default_by_sector,
                    "churn_by_aum": churn_by_aum,
                    "cohort_risk": {
                        "churn_avg_by_risk_tolerance": churn_avg_by_risk,
                        "churn_avg_by_aum_band": churn_avg_by_aum,
                        "default_avg_by_sector": default_avg_by_sector,
                        "sector_bucket_matrix": sector_bucket_matrix,
                        "aum_tolerance_matrix": aum_tolerance_matrix,
                    },
                    "segment_scores": segment_scores,
                }
            )

    @app.route("/api/analytics/samples", methods=["GET"])
    def analytics_samples():
        limit = int(request.args.get("limit", 20))
        with SessionLocal() as session:
            investors = session.query(Investor).limit(limit).all()
            loans = session.query(Loan).limit(limit).all()
            return jsonify(
                {
                    "investors": [serialize_investor(inv) for inv in investors],
                    "loans": [serialize_loan(loan) for loan in loans],
                }
            )

    @app.route("/api/analytics/timeline", methods=["GET"])
    def analytics_timeline():
        days = int(request.args.get("days", 30))
        rng = pd.date_range(end=datetime.now(timezone.utc), periods=days, freq="D")
        # Generate synthetic trend lines using recent averages as anchor
        with SessionLocal() as session:
            investors = session.query(Investor).all()
            loans = session.query(Loan).all()
            avg_churn = (
                sum((inv.churn_risk_score or 0) for inv in investors) / len(investors) if investors else 0.2
            )
            avg_default = (
                sum((loan.default_risk_score or 0) for loan in loans) / len(loans) if loans else 0.15
            )
        churn_series = _random_walk(avg_churn, days, noise=0.05)
        default_series = _random_walk(avg_default, days, noise=0.04)
        return jsonify(
            {
                "dates": [d.strftime("%Y-%m-%d") for d in rng],
                "churn": churn_series,
                "default": default_series,
            }
        )

    @app.route("/api/investors", methods=["GET"])
    def list_investors():
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 10))
        with SessionLocal() as session:
            query = session.query(Investor)
            total = query.count()
            investors = (
                query.order_by(Investor.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            items = [serialize_investor(i) for i in investors]
            return jsonify({"items": items, "total": total})

    @app.route("/api/loans", methods=["GET"])
    def list_loans():
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 10))
        with SessionLocal() as session:
            query = session.query(Loan)
            total = query.count()
            loans = (
                query.order_by(Loan.id).offset((page - 1) * page_size).limit(page_size).all()
            )
            items = [serialize_loan(l) for l in loans]
            return jsonify({"items": items, "total": total})

    @app.route("/api/predict/investor_churn", methods=["POST"])
    def predict_investor_churn():
        payload = request.get_json(force=True) or {}
        investor_id = payload.get("investor_id")
        requested_horizon = payload.get("horizon")
        requested_family = payload.get("family", "ensemble")
        extra_families = payload.get("families") or []
        families = [f for f in {requested_family, *extra_families, "ensemble", "adaboost", "knn"} if f in MODEL_FAMILIES]
        with SessionLocal() as session:
            if investor_id:
                investor = session.get(Investor, investor_id)
                if not investor:
                    return jsonify({"error": "Investor not found"}), 404
                features = investor_to_features(investor)
            else:
                missing = [f for f in ["age", "aum", "risk_tolerance", "engagement_score", "email_open_rate", "call_frequency"] if payload.get(f) is None]
                if missing:
                    return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
                features = {
                    "age": int(payload.get("age")),
                    "aum": float(payload.get("aum")),
                    "risk_tolerance": payload.get("risk_tolerance"),
                    "engagement_score": float(payload.get("engagement_score")),
                    "email_open_rate": float(payload.get("email_open_rate")),
                    "call_frequency": float(payload.get("call_frequency")),
                    "inactivity_days": int(payload.get("inactivity_days", 0)),
                    "redemption_intent": bool(payload.get("redemption_intent", False)),
                    "distribution_yield": float(payload.get("distribution_yield", 0.0)),
                    "meetings_last_quarter": int(payload.get("meetings_last_quarter", 0)),
                }

            # Portfolio medians for explainability
            inv_df = pd.DataFrame([investor_to_features(i) for i in session.query(Investor).all()])
            inv_medians = inv_df.median(numeric_only=True).to_dict() if not inv_df.empty else {}

            horizons_to_score = [requested_horizon] if requested_horizon in HORIZONS else HORIZONS
            feature_df = pd.DataFrame([features])
            family_results: Dict[str, Dict[str, Dict[str, float]]] = {}
            for fam in families:
                fam_results = {}
                for horizon in horizons_to_score:
                    model = load_family_model("churn", fam, session, horizon)
                    prob = float(model.predict_proba(feature_df)[0][1])
                    fam_results[horizon] = {"probability": prob, "bucket": risk_bucket(prob)}
                    if fam == "knn":
                        fam_results[horizon]["k"] = 20
                family_results[fam] = fam_results

            primary_family = requested_family if requested_family in family_results else "ensemble"
            primary_family = primary_family if primary_family in family_results else (families[0] if families else "ensemble")
            primary_results = family_results.get(primary_family, {})
            primary = primary_results.get("6m") or (next(iter(primary_results.values())) if primary_results else {"probability": 0.0, "bucket": "Low"})
            explanation = explain_investor_risk(features, primary.get("probability", 0.0))
            neighbors = knn_neighbors(session, "churn", features, horizon="6m", k=10)
            drivers = _top_drivers(features, inv_medians, problem="churn")

            if investor_id:
                store_results = family_results.get("ensemble") or primary_results
                for horizon, res in store_results.items():
                    setattr(investor, f"churn_risk_score_{horizon}", res["probability"])
                    setattr(investor, f"churn_risk_bucket_{horizon}", res["bucket"])
                    if horizon == "6m":
                        investor.churn_risk_score = res["probability"]
                session.commit()

            return jsonify(
                {
                    "churn_probability": primary["probability"],
                    "risk_bucket": primary["bucket"],
                    "horizons": primary_results,
                    "models": family_results,
                    "primary_family": primary_family,
                    "local_knn": family_results.get("knn"),
                    "neighbors": neighbors,
                    "drivers": drivers,
                    "explanation": explanation,
                }
            )

    @app.route("/api/predict/loan_default", methods=["POST"])
    def predict_loan_default():
        payload = request.get_json(force=True) or {}
        loan_id = payload.get("loan_id")
        requested_horizon = payload.get("horizon")
        requested_family = payload.get("family", "ensemble")
        extra_families = payload.get("families") or []
        families = [f for f in {requested_family, *extra_families, "ensemble", "adaboost", "knn"} if f in MODEL_FAMILIES]
        with SessionLocal() as session:
            if loan_id:
                loan = session.get(Loan, loan_id)
                if not loan:
                    return jsonify({"error": "Loan not found"}), 404
                features = loan_to_features(loan)
            else:
                missing = [f for f in ["amount", "ltv_ratio", "term_months", "sector", "arrears_flag"] if payload.get(f) is None]
                if missing:
                    return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
                features = {
                    "amount": float(payload.get("amount")),
                    "ltv_ratio": float(payload.get("ltv_ratio")),
                    "term_months": int(payload.get("term_months")),
                    "sector": payload.get("sector"),
                    "arrears_flag": bool(payload.get("arrears_flag")),
                    "dscr": float(payload.get("dscr", 1.2)),
                    "covenants_flag": bool(payload.get("covenants_flag", False)),
                    "collateral_score": float(payload.get("collateral_score", 0.6)),
                }

            loan_df = pd.DataFrame([loan_to_features(l) for l in session.query(Loan).all()])
            loan_medians = loan_df.median(numeric_only=True).to_dict() if not loan_df.empty else {}

            horizons_to_score = [requested_horizon] if requested_horizon in HORIZONS else HORIZONS
            feature_df = pd.DataFrame([features])
            family_results: Dict[str, Dict[str, Dict[str, float]]] = {}
            for fam in families:
                fam_results = {}
                for horizon in horizons_to_score:
                    model = load_family_model("default", fam, session, horizon)
                    prob = float(model.predict_proba(feature_df)[0][1])
                    fam_results[horizon] = {"probability": prob, "bucket": risk_bucket(prob)}
                    if fam == "knn":
                        fam_results[horizon]["k"] = 20
                family_results[fam] = fam_results

            primary_family = requested_family if requested_family in family_results else "ensemble"
            primary_family = primary_family if primary_family in family_results else (families[0] if families else "ensemble")
            primary_results = family_results.get(primary_family, {})
            primary = primary_results.get("6m") or (next(iter(primary_results.values())) if primary_results else {"probability": 0.0, "bucket": "Low"})
            explanation = explain_loan_risk(features, primary.get("probability", 0.0))
            neighbors = knn_neighbors(session, "default", features, horizon="6m", k=10)
            drivers = _top_drivers(features, loan_medians, problem="default")

            if loan_id:
                store_results = family_results.get("ensemble") or primary_results
                for horizon, res in store_results.items():
                    setattr(loan, f"default_risk_score_{horizon}", res["probability"])
                    setattr(loan, f"default_risk_bucket_{horizon}", res["bucket"])
                    if horizon == "6m":
                        loan.default_risk_score = res["probability"]
                session.commit()

            return jsonify(
                {
                    "default_probability": primary["probability"],
                    "risk_bucket": primary["bucket"],
                    "horizons": primary_results,
                    "models": family_results,
                    "primary_family": primary_family,
                    "local_knn": family_results.get("knn"),
                    "neighbors": neighbors,
                    "drivers": drivers,
                    "explanation": explanation,
                }
            )

    @app.route("/api/predict/batch_refresh", methods=["POST"])
    def batch_refresh():
        with SessionLocal() as session:
            _refresh_scores(session)
            investors = session.query(Investor).count()
            loans = session.query(Loan).count()
            _persist_snapshot(session)

            return jsonify(
                {
                    "investors_updated": investors,
                    "loans_updated": loans,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    @app.route("/api/alerts", methods=["GET"])
    def list_alerts():
        sla_days = int(request.args.get("sla_days", 5))
        now = datetime.now(timezone.utc)
        with SessionLocal() as session:
            active_alerts = session.query(RiskAlert).filter(RiskAlert.resolved_at.is_(None)).all()
            inv_ids = [a.entity_id for a in active_alerts if a.entity_type == "investor"]
            loan_ids = [a.entity_id for a in active_alerts if a.entity_type == "loan"]
            inv_map = {i.id: i for i in session.query(Investor).filter(Investor.id.in_(inv_ids)).all()} if inv_ids else {}
            loan_map = {l.id: l for l in session.query(Loan).filter(Loan.id.in_(loan_ids)).all()} if loan_ids else {}

        def days_open(alert: RiskAlert) -> int:
            start = _parse_iso(alert.first_high_at)
            if not start:
                return 0
            delta = now - start
            return max(0, delta.days)

        investor_payload = []
        loan_payload = []
        for alert in active_alerts:
            if alert.entity_type == "investor":
                inv = inv_map.get(alert.entity_id)
                if not inv:
                    continue
                d_open = days_open(alert)
                investor_payload.append(
                    {
                        "id": inv.id,
                        "name": inv.name,
                        "probability": inv.churn_risk_score_6m or inv.churn_risk_score or 0.0,
                        "bucket": inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0),
                        "days_open": d_open,
                        "sla_breach": d_open > sla_days,
                        "first_high_at": alert.first_high_at,
                        "risk_tolerance": inv.risk_tolerance,
                        "engagement_score": inv.engagement_score,
                        "inactivity_days": inv.inactivity_days,
                        "aum": inv.aum,
                    }
                )
            elif alert.entity_type == "loan":
                loan = loan_map.get(alert.entity_id)
                if not loan:
                    continue
                d_open = days_open(alert)
                loan_payload.append(
                    {
                        "id": loan.id,
                        "investor_id": loan.investor_id,
                        "amount": loan.amount,
                        "sector": loan.sector,
                        "probability": loan.default_risk_score_6m or loan.default_risk_score or 0.0,
                        "bucket": loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0),
                        "days_open": d_open,
                        "sla_breach": d_open > sla_days,
                        "first_high_at": alert.first_high_at,
                        "ltv_ratio": loan.ltv_ratio,
                        "dscr": loan.dscr,
                    }
                )

        summary = {
            "investors_active": len(investor_payload),
            "loans_active": len(loan_payload),
            "investor_breach": sum(1 for a in investor_payload if a["sla_breach"]),
            "loan_breach": sum(1 for a in loan_payload if a["sla_breach"]),
        }
        return jsonify({"investors": investor_payload, "loans": loan_payload, "summary": summary})

    @app.route("/api/eda/summary", methods=["GET"])
    def eda_summary():
        with SessionLocal() as session:
            investors = session.query(Investor).all()
            loans = session.query(Loan).all()
        if not investors and not loans:
            return jsonify({"error": "No data"}), 400
        inv_df = pd.DataFrame([investor_to_features(i) for i in investors]) if investors else pd.DataFrame()
        loan_df = pd.DataFrame([loan_to_features(l) for l in loans]) if loans else pd.DataFrame()
        return jsonify(eda_summary_build(inv_df, loan_df))

    @app.route("/api/interventions", methods=["GET", "POST"])
    def interventions():
        with SessionLocal() as session:
            if request.method == "POST":
                payload = request.get_json(force=True) or {}
                entity_type = payload.get("entity_type")
                entity_id = payload.get("entity_id")
                action_type = payload.get("action_type", "synthetic")
                if entity_type not in {"investor", "loan"} or not entity_id:
                    return jsonify({"error": "entity_type must be investor/loan and entity_id required"}), 400
                iv = Intervention(
                    entity_type=entity_type,
                    entity_id=int(entity_id),
                    action_type=action_type,
                    expected_effect=payload.get("expected_effect"),
                    engagement_delta=payload.get("engagement_delta"),
                    inactivity_delta=payload.get("inactivity_delta"),
                    ltv_delta=payload.get("ltv_delta"),
                    dscr_delta=payload.get("dscr_delta"),
                )
                session.add(iv)
                session.commit()
            records = session.query(Intervention).order_by(Intervention.id.desc()).limit(200).all()
            return jsonify(
                [
                    {
                        "id": r.id,
                        "entity_type": r.entity_type,
                        "entity_id": r.entity_id,
                        "action_type": r.action_type,
                        "expected_effect": r.expected_effect,
                        "engagement_delta": r.engagement_delta,
                        "inactivity_delta": r.inactivity_delta,
                        "ltv_delta": r.ltv_delta,
                        "dscr_delta": r.dscr_delta,
                        "created_at": r.created_at,
                    }
                    for r in records
                ]
            )

    @app.route("/api/report/daily", methods=["POST"])
    def daily_report():
        segment_type = request.args.get("segment_type")
        segment_value = request.args.get("segment_value")
        report_format = request.args.get("format", "full")
        with SessionLocal() as session:
            payload = _build_report_payload(session, segment_type, segment_value, report_format)
            return jsonify(payload)

    @app.route("/api/report/pdf", methods=["POST"])
    def daily_report_pdf():
        segment_type = request.args.get("segment_type")
        segment_value = request.args.get("segment_value")
        report_format = request.args.get("format", "full")
        with SessionLocal() as session:
            payload = _build_report_payload(session, segment_type, segment_value, report_format)
        html_body = _render_report_html(payload["report_markdown"], payload.get("summary_kpis", {}))
        return Response(html_body, mimetype="text/html")

    return app


def eda_summary_build(inv_df: pd.DataFrame, loan_df: pd.DataFrame):
    def stats(df: pd.DataFrame, cols: list[str]):
        out = {}
        for c in cols:
            if c not in df.columns or df[c].empty:
                continue
            out[c] = {
                "mean": float(df[c].mean()),
                "p25": float(df[c].quantile(0.25)),
                "p50": float(df[c].median()),
                "p75": float(df[c].quantile(0.75)),
            }
        return out
    summary = {
        "investor_stats": stats(inv_df, ["engagement_score", "email_open_rate", "aum"]),
        "loan_stats": stats(loan_df, ["ltv_ratio", "dscr", "amount"]),
    }
    # light narration
    lines = []
    es = summary["investor_stats"].get("engagement_score")
    if es:
        lines.append(f"Engagement median {es['p50']:.1f} (p25 {es['p25']:.1f}); tail below {es['p25']:.1f} may need outreach.")
    lt = summary["loan_stats"].get("ltv_ratio")
    if lt:
        lines.append(f"LTV median {lt['p50']:.2f}, p75 {lt['p75']:.2f}; higher tail worth covenant focus.")
    ds = summary["loan_stats"].get("dscr")
    if ds:
        lines.append(f"DSCR median {ds['p50']:.2f}; {ds['p25']:.2f} at lower quartile suggests coverage stress segment.")
    summary["narrative"] = lines
    return summary


def serialize_investor(investor: Investor) -> Dict:
    return {
        "id": investor.id,
        "name": investor.name,
        "age": investor.age,
        "aum": investor.aum,
        "risk_tolerance": investor.risk_tolerance,
        "engagement_score": investor.engagement_score,
        "email_open_rate": investor.email_open_rate,
        "call_frequency": investor.call_frequency,
        "inactivity_days": investor.inactivity_days,
        "redemption_intent": bool(investor.redemption_intent),
        "distribution_yield": investor.distribution_yield,
        "meetings_last_quarter": investor.meetings_last_quarter,
        "churn_risk_score": investor.churn_risk_score,
        "risk_bucket": risk_bucket(investor.churn_risk_score or 0),
        "churn_risk_score_3m": investor.churn_risk_score_3m,
        "churn_risk_score_6m": investor.churn_risk_score_6m,
        "churn_risk_score_12m": investor.churn_risk_score_12m,
        "churn_risk_bucket_3m": investor.churn_risk_bucket_3m,
        "churn_risk_bucket_6m": investor.churn_risk_bucket_6m,
        "churn_risk_bucket_12m": investor.churn_risk_bucket_12m,
    }


def serialize_loan(loan: Loan) -> Dict:
    return {
        "id": loan.id,
        "investor_id": loan.investor_id,
        "amount": loan.amount,
        "ltv_ratio": loan.ltv_ratio,
        "term_months": loan.term_months,
        "sector": loan.sector,
        "arrears_flag": bool(loan.arrears_flag),
        "dscr": loan.dscr,
        "covenants_flag": bool(loan.covenants_flag),
        "collateral_score": loan.collateral_score,
        "default_risk_score": loan.default_risk_score,
        "risk_bucket": risk_bucket(loan.default_risk_score or 0),
        "default_risk_score_3m": loan.default_risk_score_3m,
        "default_risk_score_6m": loan.default_risk_score_6m,
        "default_risk_score_12m": loan.default_risk_score_12m,
        "default_risk_bucket_3m": loan.default_risk_bucket_3m,
        "default_risk_bucket_6m": loan.default_risk_bucket_6m,
        "default_risk_bucket_12m": loan.default_risk_bucket_12m,
    }


def _scores_missing(session) -> bool:
    investor_missing = (
        session.query(Investor).filter(Investor.churn_risk_score.is_(None)).count() > 0
    )
    loan_missing = (
        session.query(Loan).filter(Loan.default_risk_score.is_(None)).count() > 0
    )
    return investor_missing or loan_missing


def _refresh_scores(session):
    churn_models = {h: load_or_train_horizon(CHURN_MODEL_PATHS, train_churn_model, session, h) for h in HORIZONS}
    default_models = {h: load_or_train_horizon(DEFAULT_MODEL_PATHS, train_default_model, session, h) for h in HORIZONS}

    investors = session.query(Investor).all()
    if investors:
        investor_features = [investor_to_features(i) for i in investors]
        features_df = pd.DataFrame(investor_features)
        for horizon, model in churn_models.items():
            probs = model.predict_proba(features_df)[:, 1]
            for inv, prob in zip(investors, probs):
                setattr(inv, f"churn_risk_score_{horizon}", float(prob))
                setattr(inv, f"churn_risk_bucket_{horizon}", risk_bucket(prob))
            # Maintain legacy single score as 6m view
            if horizon == "6m":
                for inv, prob in zip(investors, probs):
                    inv.churn_risk_score = float(prob)

    loans = session.query(Loan).all()
    if loans:
        loan_features = [loan_to_features(l) for l in loans]
        features_df = pd.DataFrame(loan_features)
        for horizon, model in default_models.items():
            probs = model.predict_proba(features_df)[:, 1]
            for loan, prob in zip(loans, probs):
                setattr(loan, f"default_risk_score_{horizon}", float(prob))
                setattr(loan, f"default_risk_bucket_{horizon}", risk_bucket(prob))
            if horizon == "6m":
                for loan, prob in zip(loans, probs):
                    loan.default_risk_score = float(prob)

    _update_alerts(session, investors or [], loans or [], horizon="6m")
    session.commit()


def _update_alerts(session, investors: List[Investor], loans: List[Loan], horizon: str = "6m", sla_days: int = 5):
    """
    Track first high-risk timestamps and keep an active alert list for SLA monitoring.
    Stored in a dedicated risk_alerts table to avoid altering core entities.
    """
    existing = {(a.entity_type, a.entity_id): a for a in session.query(RiskAlert).all()}
    now_iso = datetime.now(timezone.utc).isoformat()

    def upsert(entity_type: str, obj, score_attr: str, bucket_attr: str):
        prob = float(getattr(obj, score_attr, 0) or 0.0)
        bucket = getattr(obj, bucket_attr, None) or risk_bucket(prob)
        key = (entity_type, obj.id)
        alert = existing.get(key)
        if bucket == "High":
            if alert:
                if alert.resolved_at:
                    alert.first_high_at = now_iso
                alert.mark_seen()
                alert.last_bucket = bucket
                alert.last_score = prob
                alert.resolved_at = None
            else:
                alert = RiskAlert(
                    entity_type=entity_type,
                    entity_id=obj.id,
                    first_high_at=now_iso,
                    last_seen_at=now_iso,
                    last_bucket=bucket,
                    last_score=prob,
                    sla_days=sla_days,
                )
                session.add(alert)
                existing[key] = alert
        else:
            if alert and alert.resolved_at is None:
                alert.last_seen_at = now_iso
                alert.last_bucket = bucket
                alert.last_score = prob
                alert.resolved_at = now_iso

    for inv in investors:
        upsert("investor", inv, f"churn_risk_score_{horizon}", f"churn_risk_bucket_{horizon}")
    for loan in loans:
        upsert("loan", loan, f"default_risk_score_{horizon}", f"default_risk_bucket_{horizon}")


def _random_walk(anchor: float, steps: int, noise: float = 0.05, min_val: float = 0.0, max_val: float = 1.0):
    vals = []
    current = anchor
    for _ in range(steps):
        delta = random.uniform(-noise, noise)
        current = max(min_val, min(max_val, current + delta))
        vals.append(current)
    return vals


def _safe_avg(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_summary(investors: List[Investor], loans: List[Loan]) -> Dict:
    investor_buckets = {"Low": 0, "Medium": 0, "High": 0}
    for inv in investors:
        bucket = inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)
        investor_buckets[bucket] += 1
    churn_by_tolerance = {"low": 0, "medium": 0, "high": 0}
    for inv in investors:
        if (inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)) == "High":
            churn_by_tolerance[inv.risk_tolerance] = churn_by_tolerance.get(inv.risk_tolerance, 0) + 1

    loan_buckets = {"Low": 0, "Medium": 0, "High": 0}
    for loan in loans:
        bucket = loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)
        loan_buckets[bucket] += 1
    sector_default = {}
    for loan in loans:
        sector_default.setdefault(loan.sector, {"high": 0, "exposure": 0.0})
        if (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High":
            sector_default[loan.sector]["high"] += 1
            sector_default[loan.sector]["exposure"] += loan.amount

    top_investors = sorted(investors, key=lambda i: (i.churn_risk_score_6m or i.churn_risk_score or 0), reverse=True)[:5]
    top_loans = sorted(loans, key=lambda l: (l.default_risk_score_6m or l.default_risk_score or 0), reverse=True)[:5]

    avg_churn = (
        sum((inv.churn_risk_score_6m or inv.churn_risk_score or 0) for inv in investors) / len(investors)
        if investors
        else 0
    )
    prev_day_avg = max(0, min(1, avg_churn + random.uniform(-0.05, 0.05)))

    high_default_exposure = sum(
        loan.amount for loan in loans if (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High"
    )
    engagement_avg = sum(inv.engagement_score for inv in investors) / len(investors) if investors else 0
    email_open_avg = sum(inv.email_open_rate for inv in investors) / len(investors) if investors else 0
    yield_values = [inv.distribution_yield for inv in investors]
    dscr_values = [loan.dscr for loan in loans]
    distribution_yield = {
        "avg": sum(yield_values) / len(yield_values) if yield_values else 0,
        "p25": float(pd.Series(yield_values).quantile(0.25)) if yield_values else 0,
        "p50": float(pd.Series(yield_values).quantile(0.5)) if yield_values else 0,
        "p75": float(pd.Series(yield_values).quantile(0.75)) if yield_values else 0,
    }
    dscr = {
        "avg": sum(dscr_values) / len(dscr_values) if dscr_values else 0,
        "p25": float(pd.Series(dscr_values).quantile(0.25)) if dscr_values else 0,
        "p50": float(pd.Series(dscr_values).quantile(0.5)) if dscr_values else 0,
        "p75": float(pd.Series(dscr_values).quantile(0.75)) if dscr_values else 0,
    }

    engagement_trend = _random_walk(engagement_avg / 100.0, 14, noise=0.03)
    dscr_trend = _random_walk(max(0.5, dscr["avg"] / 2), 14, noise=0.05, min_val=0.4, max_val=2.5)

    return {
        "investor_buckets": investor_buckets,
        "loan_buckets": loan_buckets,
        "avg_churn_risk": avg_churn,
        "prev_day_avg_churn_risk": prev_day_avg,
        "high_default_exposure": high_default_exposure,
        "churn_by_tolerance": churn_by_tolerance,
        "sector_default": sector_default,
        "engagement_avg": engagement_avg,
        "email_open_avg": email_open_avg,
        "distribution_yield": distribution_yield,
        "dscr": dscr,
        "engagement_trend": engagement_trend,
        "dscr_trend": dscr_trend,
        "top_investors": [
            {
                "id": inv.id,
                "name": inv.name,
                "probability": inv.churn_risk_score or 0,
                "bucket": risk_bucket(inv.churn_risk_score or 0),
            }
            for inv in top_investors
        ],
        "top_loans": [
            {
                "id": loan.id,
                "sector": loan.sector,
                "amount": loan.amount,
                "probability": loan.default_risk_score or 0,
                "bucket": risk_bucket(loan.default_risk_score or 0),
            }
            for loan in top_loans
        ],
    }


def _filter_by_segment(investors: List[Investor], loans: List[Loan], segment_type: str, segment_value: str):
    if segment_type == "sector":
        loans_filtered = [loan for loan in loans if loan.sector.lower() == segment_value.lower()]
        inv_ids = {loan.investor_id for loan in loans_filtered}
        investors_filtered = [inv for inv in investors if inv.id in inv_ids]
        return investors_filtered, loans_filtered
    if segment_type == "risk_tolerance":
        investors_filtered = [inv for inv in investors if inv.risk_tolerance.lower() == segment_value.lower()]
        inv_ids = {inv.id for inv in investors_filtered}
        loans_filtered = [loan for loan in loans if loan.investor_id in inv_ids]
        return investors_filtered, loans_filtered
    if segment_type == "aum_band":
        investors_filtered = [inv for inv in investors if _aum_band(inv.aum) == segment_value]
        inv_ids = {inv.id for inv in investors_filtered}
        loans_filtered = [loan for loan in loans if loan.investor_id in inv_ids]
        return investors_filtered, loans_filtered
    return investors, loans


def _persist_snapshot(session):
    """Create today's snapshot if it doesn't already exist."""
    today = date.today().isoformat()
    existing = session.query(DailySnapshot).filter(DailySnapshot.snapshot_date == today).first()
    if existing:
        return existing

    investors = session.query(Investor).all()
    loans = session.query(Loan).all()

    avg_churn_3m = _safe_avg([inv.churn_risk_score_3m for inv in investors])
    avg_churn_6m = _safe_avg([inv.churn_risk_score_6m for inv in investors])
    avg_churn_12m = _safe_avg([inv.churn_risk_score_12m for inv in investors])

    avg_default_3m = _safe_avg([loan.default_risk_score_3m for loan in loans])
    avg_default_6m = _safe_avg([loan.default_risk_score_6m for loan in loans])
    avg_default_12m = _safe_avg([loan.default_risk_score_12m for loan in loans])

    high_inv_count_6m = sum(
        1 for inv in investors if (inv.churn_risk_bucket_6m or risk_bucket(inv.churn_risk_score_6m or inv.churn_risk_score or 0)) == "High"
    )
    high_loan_count_6m = sum(
        1 for loan in loans if (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High"
    )
    high_exposure_6m = sum(
        loan.amount for loan in loans if (loan.default_risk_bucket_6m or risk_bucket(loan.default_risk_score_6m or loan.default_risk_score or 0)) == "High"
    )

    snap = DailySnapshot(
        snapshot_date=today,
        avg_churn_risk_3m=avg_churn_3m,
        avg_churn_risk_6m=avg_churn_6m,
        avg_churn_risk_12m=avg_churn_12m,
        avg_default_risk_3m=avg_default_3m,
        avg_default_risk_6m=avg_default_6m,
        avg_default_risk_12m=avg_default_12m,
        high_risk_investor_count_6m=high_inv_count_6m,
        high_risk_loan_count_6m=high_loan_count_6m,
        high_risk_exposure_amount_6m=high_exposure_6m,
    )
    session.add(snap)
    session.commit()
    return snap


def _get_snapshots(session):
    today_str = date.today().isoformat()
    today_snap = session.query(DailySnapshot).filter(DailySnapshot.snapshot_date == today_str).first()
    yesterday_snap = (
        session.query(DailySnapshot)
        .filter(DailySnapshot.snapshot_date < today_str)
        .order_by(DailySnapshot.snapshot_date.desc())
        .first()
    )
    week_cut = date.today().fromordinal(date.today().toordinal() - 7).isoformat()
    week_ago_snap = (
        session.query(DailySnapshot)
        .filter(DailySnapshot.snapshot_date <= week_cut)
        .order_by(DailySnapshot.snapshot_date.desc())
        .first()
    )
    return today_snap, yesterday_snap, week_ago_snap


def _snapshot_deltas(today_snap: DailySnapshot | None, yesterday_snap: DailySnapshot | None, week_snap: DailySnapshot | None) -> Dict:
    if not today_snap:
        return {}
    def delta(curr, prev):
        if curr is None or prev is None:
            return None
        return curr - prev

    return {
        "high_risk_investors_6m": {
            "current": today_snap.high_risk_investor_count_6m,
            "delta": delta(today_snap.high_risk_investor_count_6m, getattr(yesterday_snap, "high_risk_investor_count_6m", None)),
            "delta_week": delta(today_snap.high_risk_investor_count_6m, getattr(week_snap, "high_risk_investor_count_6m", None)),
        },
        "high_risk_loans_6m": {
            "current": today_snap.high_risk_loan_count_6m,
            "delta": delta(today_snap.high_risk_loan_count_6m, getattr(yesterday_snap, "high_risk_loan_count_6m", None)),
            "delta_week": delta(today_snap.high_risk_loan_count_6m, getattr(week_snap, "high_risk_loan_count_6m", None)),
        },
        "high_risk_exposure_6m": {
            "current": today_snap.high_risk_exposure_amount_6m,
            "delta": delta(today_snap.high_risk_exposure_amount_6m, getattr(yesterday_snap, "high_risk_exposure_amount_6m", None)),
            "delta_week": delta(today_snap.high_risk_exposure_amount_6m, getattr(week_snap, "high_risk_exposure_amount_6m", None)),
        },
        "avg_churn_6m": {
            "current": today_snap.avg_churn_risk_6m,
            "delta": delta(today_snap.avg_churn_risk_6m, getattr(yesterday_snap, "avg_churn_risk_6m", None)),
            "delta_week": delta(today_snap.avg_churn_risk_6m, getattr(week_snap, "avg_churn_risk_6m", None)),
        },
        "avg_default_6m": {
            "current": today_snap.avg_default_risk_6m,
            "delta": delta(today_snap.avg_default_risk_6m, getattr(yesterday_snap, "avg_default_risk_6m", None)),
            "delta_week": delta(today_snap.avg_default_risk_6m, getattr(week_snap, "avg_default_risk_6m", None)),
        },
    }


def _build_report_payload(session, segment_type: str | None, segment_value: str | None, report_format: str) -> Dict:
    # If scores are missing, refresh them
    if _scores_missing(session):
        logger.info("Scores missing; refreshing before generating report.")
        _refresh_scores(session)

    investors = session.query(Investor).all()
    loans = session.query(Loan).all()
    # Optional segment filtering
    if segment_type and segment_value:
        investors, loans = _filter_by_segment(investors, loans, segment_type, segment_value)

    inv_ids = {inv.id for inv in investors}
    loan_ids = {l.id for l in loans}
    alerts = {
        (a.entity_type, a.entity_id): a
        for a in session.query(RiskAlert).filter(RiskAlert.resolved_at.is_(None)).all()
        if (a.entity_type == "investor" and a.entity_id in inv_ids) or (a.entity_type == "loan" and a.entity_id in loan_ids)
    }

    # Ensure snapshot exists, then get deltas
    today_snap = _persist_snapshot(session)
    _, yesterday_snap, week_snap = _get_snapshots(session)
    deltas = _snapshot_deltas(today_snap, yesterday_snap, week_snap)

    summary_kpis = compute_summary(investors, loans)
    summary_kpis["deltas"] = deltas
    summary_kpis["playbooks"] = evaluate_playbooks(investors, loans, alerts)
    try:
        inv_fields = [
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
        def missing_pct(objs, field):
            vals = [getattr(o, field, None) for o in objs]
            missing = sum(1 for v in vals if v is None)
            return (missing / max(len(vals), 1)) * 100
        summary_kpis["data_audit"] = {
            "investor_missing_pct": {f: round(missing_pct(investors, f), 2) for f in inv_fields},
            "loan_missing_pct": {f: round(missing_pct(loans, f), 2) for f in loan_fields},
        }
    except Exception:
        summary_kpis["data_audit"] = {}
    summary_kpis["segment"] = {"type": segment_type, "value": segment_value}
    summary_kpis["segment_summary"] = segment_mini_report(investors, loans, segment_type, segment_value)
    # Add sensitivity highlights from metrics if available
    metrics_path = BASE_DIR / "models" / "metrics.json"
    sensitivity = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            sensitivity = metrics.get("sensitivity", {})
        except Exception:
            sensitivity = {}
    summary_kpis["sensitivity"] = sensitivity
    # EDA narrative inline
    try:
        inv_df = pd.DataFrame([investor_to_features(i) for i in investors]) if investors else pd.DataFrame()
        loan_df = pd.DataFrame([loan_to_features(l) for l in loans]) if loans else pd.DataFrame()
        summary_kpis["eda_narrative"] = eda_summary_build(inv_df, loan_df)
    except Exception:
        summary_kpis["eda_narrative"] = []
    report_text = build_report(summary_kpis)
    variant_text = _format_report_variant(report_text, summary_kpis, report_format)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_kpis": summary_kpis,
        "report_markdown": variant_text,
        "base_report_markdown": report_text,
        "format": report_format,
    }


def _render_report_html(markdown_text: str, summary_kpis: Dict) -> str:
    """Lightweight HTML wrapper so browsers can print to PDF without extra deps."""
    safe = html.escape(markdown_text).replace("\n", "<br>")
    stats = summary_kpis or {}
    title = "Capspace Daily Report"
    return f"""
    <html>
      <head>
        <meta charset='utf-8' />
        <title>{title}</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #0f172a; }}
          .muted {{ color: #475569; }}
          h1 {{ margin-bottom: 0; }}
          pre {{ background: #f8fafc; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <div class="muted">Generated at {datetime.now(timezone.utc).isoformat()}</div>
        <pre>{safe}</pre>
      </body>
    </html>
    """


def _ai_explain(
    question: str,
    summary: Dict,
    metrics: Dict,
    focus: str | None = None,
    history=None,
    context: Dict | None = None,
) -> str:
    """Provide an explainable AI answer; uses OpenAI if configured, otherwise a heuristic fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    history = history or []
    context = context or {}

    def _shrink(payload: Dict | list, max_len: int = 8000) -> str:
        """Clamp serialized payload to avoid overlong prompts."""
        try:
            txt = json.dumps(payload)
        except Exception:
            txt = str(payload)
        if len(txt) > max_len:
            return txt[: max_len // 2] + "...[truncated]..." + txt[-max_len // 2 :]
        return txt

    focus_key = (focus or "").lower()
    if any(k in focus_key for k in ["dashboard", "report"]):
        data_payload = context or summary
    elif context:
        data_payload = context
    else:
        data_payload = {}
    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            try:
                from langchain.prompts import ChatPromptTemplate  # LangChain <0.2 fallback
            except ImportError:  # pragma: no cover
                from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(
                "You are an explainable AI assistant for a private credit risk lab. Keep replies concise, warm, and conversational (2-4 bullets plus a one-line takeaway). "
                "Focus: {focus}. Question: {question}. "
                "Data payload: {data_payload}. Model metrics: {metrics}. "
                "Chat history: {history}. "
                "If data is empty, ask a short clarifying question or provide a brief, generic next step."
            )
            llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=api_key, max_tokens=400)
            messages = prompt.format_messages(
                question=question,
                data_payload=_shrink(data_payload),
                metrics=_shrink(metrics, max_len=4000),
                focus=focus or "general",
                history=_shrink(history[-5:], max_len=1500),
            )
            result = llm.invoke(messages)
            return getattr(result, "content", str(result))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("AI explanation failed, falling back. %s", exc)

    # Fallback: deterministic explanation
    avg_churn = summary.get("avg_churn_risk", 0)
    avg_default = summary.get("loan_buckets", {}).get("High", 0)
    dist_yield = summary.get("distribution_yield", {}).get("p50", 0)
    dscr_mid = summary.get("dscr", {}).get("p50", 0)
    last_user = history[-1]["content"] if history else question
    return (
        "AI fallback (add OPENAI_API_KEY for live chat).\n"
        f"Hey there — on your question: \"{last_user}\":\n"
        f"- Churn risk ~{avg_churn:.2%}; watch the top cohort and lift touchpoints.\n"
        f"- High-risk loans: {avg_default}; double-check covenants/DSCR.\n"
        f"- Median yield {dist_yield:.2%}, median DSCR {dscr_mid:.2f}; keep yield while nudging coverage up.\n"
        "Takeaway: prioritize outreach to high-churn investors and stress-test loans with high LTV or low DSCR."
    )


def build_report(summary_kpis: Dict) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    inv = summary_kpis["investor_buckets"]
    loans = summary_kpis["loan_buckets"]
    top_investors = summary_kpis["top_investors"]
    top_loans = summary_kpis["top_loans"]
    avg = summary_kpis["avg_churn_risk"]
    prev = summary_kpis["prev_day_avg_churn_risk"]
    high_exposure = summary_kpis["high_default_exposure"]
    churn_by_tol = summary_kpis.get("churn_by_tolerance", {})
    sector_default = summary_kpis.get("sector_default", {})
    engagement_avg = summary_kpis.get("engagement_avg", 0)
    email_open_avg = summary_kpis.get("email_open_avg", 0)
    dist_yield = summary_kpis.get("distribution_yield", {})
    dscr = summary_kpis.get("dscr", {})
    engagement_trend = summary_kpis.get("engagement_trend", [])
    dscr_trend = summary_kpis.get("dscr_trend", [])
    deltas = summary_kpis.get("deltas", {})
    def _arrow(val):
        if val is None:
            return "→"
        return "↑" if val > 0 else "↓" if val < 0 else "→"
    def _fmt(val, currency=False):
        if val is None:
            return "n/a"
        if currency:
            return f"${val:,.0f}"
        if isinstance(val, int):
            return f"{val}"
        return f"{val:.2f}"
    def _delta_text(block: Dict, pct: bool = False, currency: bool = False):
        cur = block.get("current")
        delta = block.get("delta")
        prev_val = cur - delta if (cur is not None and delta is not None) else None
        if cur is None:
            return "n/a"
        if delta is None:
            return _fmt(cur, currency)
        if pct and prev_val not in (None, 0):
            pct_val = (delta / prev_val) * 100
            return f"{_arrow(delta)} {_fmt(cur, currency)} ({pct_val:+.1f}%)"
        return f"{_arrow(delta)} {_fmt(cur, currency)}"
    # Safe numeric helpers
    def num(val, default=0.0):
        return default if val is None else val
    eng_trend_text = (
        f"start {engagement_trend[0]*100:.1f}% → latest {engagement_trend[-1]*100:.1f}% (n={len(engagement_trend)})"
        if engagement_trend else "insufficient data"
    )
    dscr_trend_text = (
        f"start {dscr_trend[0]:.2f} → latest {dscr_trend[-1]:.2f} (n={len(dscr_trend)})"
        if dscr_trend else "insufficient data"
    )

    top_inv_lines = "\n".join(
        [f"- {i['name']} (p={i['probability']:.2f}, {i['bucket']})" for i in top_investors]
    )
    top_loan_lines = "\n".join(
        [
            f"- Loan #{l['id']} in {l['sector']} (${l['amount']:,.0f}) (p={l['probability']:.2f}, {l['bucket']})"
            for l in top_loans
        ]
    )
    sector_lines = "\n".join(
        [
            f"- {sector}: {data['high']} high-risk loans, exposure ${data['exposure']:,.0f}"
            for sector, data in sorted(sector_default.items(), key=lambda kv: kv[1]["exposure"], reverse=True)
        ]
    )

    playbooks = summary_kpis.get("playbooks", {})
    action_lines = []
    for pb in playbooks.get("investor_playbooks", []):
        action_lines.append(f"- {pb['label']} ({pb['count']} investors; SLA breach {pb.get('sla_breach',0)}; avg days open {pb.get('avg_days_open',0):.1f}): {pb['action']}")
    for pb in playbooks.get("loan_playbooks", []):
        action_lines.append(f"- {pb['label']} ({pb['count']} loans; SLA breach {pb.get('sla_breach',0)}; avg days open {pb.get('avg_days_open',0):.1f}): {pb['action']}")
    action_block = "\n".join(action_lines) if action_lines else "- No specific actions generated today."
    hot_sectors = sorted(sector_default.items(), key=lambda kv: kv[1]["high"], reverse=True)
    hotspot_text = ", ".join([f"{name} ({data['high']} loans, ${data['exposure']:,.0f})" for name, data in hot_sectors[:2]]) or "None"
    sens = summary_kpis.get("sensitivity", {})
    sens_lines = []
    if sens.get("churn"):
        top_churn = sorted(sens["churn"].items(), key=lambda kv: max(kv[1].values()), reverse=True)[:2]
        for feat, vals in top_churn:
            sens_lines.append(f"- Churn: {feat} ±0.5σ -> up {vals.get('up_flip_pct',0)*100:.1f}%, down {vals.get('down_flip_pct',0)*100:.1f}% flips")
    if sens.get("default"):
        top_def = sorted(sens["default"].items(), key=lambda kv: max(kv[1].values()), reverse=True)[:2]
        for feat, vals in top_def:
            sens_lines.append(f"- Default: {feat} ±0.5σ -> up {vals.get('up_flip_pct',0)*100:.1f}%, down {vals.get('down_flip_pct',0)*100:.1f}% flips")
    sens_block = "\n".join(sens_lines) if sens_lines else "- Sensitivity data unavailable."
    audit = summary_kpis.get("data_audit", {})
    inv_missing = audit.get("investor_missing_pct", {})
    loan_missing = audit.get("loan_missing_pct", {})
    missing_line = f"Missingness: investors inactivity_days {inv_missing.get('inactivity_days','n/a')}%, loans dscr {loan_missing.get('dscr','n/a')}%" if audit else "Missingness: n/a"

    report = f"""# Daily Risk & Engagement Report
Date: {today}
{summary_kpis.get("segment_summary","")}

## Investor Churn Overview
- Low: {inv.get('Low', 0)} | Medium: {inv.get('Medium', 0)} | High: {inv.get('High', 0)}
- Average churn risk: {avg:.2f} (prev day: {prev:.2f})
- High-risk churn by tolerance: low={churn_by_tol.get('low',0)}, medium={churn_by_tol.get('medium',0)}, high={churn_by_tol.get('high',0)}
- Avg engagement score: {engagement_avg:.1f} | Avg email open rate: {email_open_avg:.2f}
- Median distribution yield: {dist_yield.get('p50', 0):.3f}
- Engagement momentum (synthetic): {eng_trend_text}
- Top at-risk investors:
{top_inv_lines or '- None'}

## Loan Default Overview
- Low: {loans.get('Low', 0)} | Medium: {loans.get('Medium', 0)} | High: {loans.get('High', 0)}
- High-risk exposure: ${high_exposure:,.0f}
- DSCR median: {dscr.get('p50', 0):.2f} | momentum: {dscr_trend_text}
- High-risk by sector:
{sector_lines or '- None'}
- Top at-risk loans:
{top_loan_lines or '- None'}

## Recommended Actions
{action_block}

## Intervention Priorities (sensitivity)
{sens_block}

## Data Quality Snapshot
- {missing_line}
- EDA notes:
{ '\n'.join(['- ' + ln for ln in summary_kpis.get('eda_narrative', [])]) if summary_kpis.get('eda_narrative') else '- n/a' }

## What Changed vs Yesterday (6m horizon)
- High-risk investors: {_delta_text(deltas.get('high_risk_investors_6m', {}), pct=True)}
- High-risk loans: {_delta_text(deltas.get('high_risk_loans_6m', {}), pct=True)}
- High-risk exposure: {_delta_text(deltas.get('high_risk_exposure_6m', {}), pct=True, currency=True)}
- Avg churn risk (6m): {_delta_text(deltas.get('avg_churn_6m', {}))}
- Avg default risk (6m): {_delta_text(deltas.get('avg_default_6m', {}))}
- Hotspots: {hotspot_text}

## What Changed vs Last Week (6m horizon)
- High-risk investors: {_arrow(deltas.get('high_risk_investors_6m', {}).get('delta_week'))} {deltas.get('high_risk_investors_6m', {}).get('current', 'n/a')} (Δ {deltas.get('high_risk_investors_6m', {}).get('delta_week', 'n/a')})
- High-risk loans: {_arrow(deltas.get('high_risk_loans_6m', {}).get('delta_week'))} {deltas.get('high_risk_loans_6m', {}).get('current', 'n/a')} (Δ {deltas.get('high_risk_loans_6m', {}).get('delta_week', 'n/a')})
- High-risk exposure: {_arrow(deltas.get('high_risk_exposure_6m', {}).get('delta_week'))} ${num(deltas.get('high_risk_exposure_6m', {}).get('current')):,.0f} (Δ {num(deltas.get('high_risk_exposure_6m', {}).get('delta_week')):,.0f})
- Avg churn risk (6m): {_arrow(deltas.get('avg_churn_6m', {}).get('delta_week'))} {num(deltas.get('avg_churn_6m', {}).get('current')):.2f} (Δ {num(deltas.get('avg_churn_6m', {}).get('delta_week')):.2f})
- Avg default risk (6m): {_arrow(deltas.get('avg_default_6m', {}).get('delta_week'))} {num(deltas.get('avg_default_6m', {}).get('current')):.2f} (Δ {num(deltas.get('avg_default_6m', {}).get('delta_week')):.2f})
"""
    return report


def _format_report_variant(report_text: str, summary_kpis: Dict, variant: str) -> str:
    variant = (variant or "full").lower()
    if variant == "board":
        return (
            f"# Board Summary (1-pager)\n"
            f"- Churn: {summary_kpis['avg_churn_risk']:.2f} avg; High={summary_kpis['investor_buckets'].get('High',0)}\n"
            f"- Default: High-risk loans={summary_kpis['loan_buckets'].get('High',0)}, exposure ${summary_kpis['high_default_exposure']:,.0f}\n"
            f"- Top themes: engagement avg {summary_kpis.get('engagement_avg',0):.1f}, email open {summary_kpis.get('email_open_avg',0):.2f}, DSCR p50 {summary_kpis.get('dscr',{}).get('p50',0):.2f}\n"
            f"- Actions: {', '.join(pb['label'] for pb in summary_kpis.get('playbooks',{}).get('investor_playbooks',[])[:2]) or 'Monitor'}\n"
        )
    if variant == "rm":
        actions = summary_kpis.get("playbooks", {}).get("investor_playbooks", []) + summary_kpis.get("playbooks", {}).get("loan_playbooks", [])
        lines = [f"- {a['label']}: {a['action']} ({a['count']} items)" for a in actions[:6]]
        return "# RM Call Notes\n" + "\n".join(lines or ["- No urgent actions."])
    if variant == "ic":
        return (
            "# IC Pack Highlights\n"
            f"- Risk: churn {summary_kpis['avg_churn_risk']:.2f}, default high-risk exposure ${summary_kpis['high_default_exposure']:,.0f}\n"
            f"- Concentration: top sectors {', '.join(list(summary_kpis.get('sector_default',{}).keys())[:3])}\n"
            f"- Trends: engagement momentum len={len(summary_kpis.get('engagement_trend',[]))}, DSCR momentum len={len(summary_kpis.get('dscr_trend',[]))}\n"
            f"- Deltas: High-risk inv Δ={summary_kpis.get('deltas',{}).get('high_risk_investors_6m',{}).get('delta','n/a')}, loans Δ={summary_kpis.get('deltas',{}).get('high_risk_loans_6m',{}).get('delta','n/a')}\n"
        )
    return report_text


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

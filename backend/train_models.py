from pathlib import Path
import json

import joblib

from ml_models import (
    CHURN_MODEL_PATHS,
    CHURN_FAMILY_PATHS,
    DEFAULT_MODEL_PATHS,
    DEFAULT_FAMILY_PATHS,
    HORIZONS,
    MODEL_FAMILIES,
    MODEL_FAMILIES_FULL,
    HYPERPARAM_SUMMARY,
    segment_risk_summary,
    single_feature_benchmarks,
    diagnostics_1d_curves,
    compute_sensitivity,
    feature_stats,
    imputation_benchmarks,
    train_churn_family_model,
    train_churn_model,
    train_default_family_model,
    train_default_model,
    bootstrap_metric_ci,
    bootstrap_coefficients,
    model_selection_summary,
    non_linear_patterns,
    correlation_summary,
    imbalance_summary,
    robustness_summary,
    segment_roc_summary,
    feature_importance_summary,
    regression_forecaster,
    scenario_surfaces,
    eda_bundle,
    regression_kpis,
    per_instance_contributions,
    segment_roc_bias,
    voting_importance,
)
from models import SessionLocal, init_db


def main():
    init_db()
    session = SessionLocal()
    try:
        metrics_payload = {
            "churn": {},
            "default": {},
            "baseline_distributions": {},
            "model_families": {"churn": {}, "default": {}},
            "thresholds": {"churn": {}, "default": {}},
            "segment_risk": {},
            "single_feature_benchmarks": {},
            "hyperparameters": HYPERPARAM_SUMMARY,
            "sensitivity": {},
            "baseline_feature_stats": {},
            "missing_data": {},
            "bootstrap_ci": {},
            "cohort_models": {},
            "segment_features": {},
            "diagnostics_1d": {},
            "model_selection": {},
            "non_linear": {},
            "correlation": {},
            "imbalance": {},
            "robustness": {},
            "segment_roc": {},
            "bootstrap_coefficients": {},
        }
        for horizon in HORIZONS:
            # Train and persist each family (ensemble + baselines)
            ensemble_churn_metrics = None
            ensemble_default_metrics = None
            for family in MODEL_FAMILIES_FULL:
                if family == "ensemble":
                    churn_model, churn_metrics = train_churn_model(session, horizon=horizon)
                    default_model, default_metrics = train_default_model(session, horizon=horizon)
                    ensemble_churn_metrics = churn_metrics
                    ensemble_default_metrics = default_metrics
                else:
                    churn_model, churn_metrics = train_churn_family_model(session, horizon=horizon, family=family)
                    default_model, default_metrics = train_default_family_model(session, horizon=horizon, family=family)

                # Persist family models for reuse in inference paths
                joblib.dump(churn_model, CHURN_FAMILY_PATHS.get(family, CHURN_MODEL_PATHS)[horizon])
                joblib.dump(default_model, DEFAULT_FAMILY_PATHS.get(family, DEFAULT_MODEL_PATHS)[horizon])

                metrics_payload["model_families"]["churn"].setdefault(family, {})[horizon] = churn_metrics
                metrics_payload["model_families"]["default"].setdefault(family, {})[horizon] = default_metrics
                if family == "ensemble":
                    metrics_payload["churn"][horizon] = churn_metrics
                    metrics_payload["default"][horizon] = default_metrics

            # Threshold grids (derived from ensemble metrics where available)
            metrics_payload["thresholds"]["churn"][horizon] = (ensemble_churn_metrics or {}).get("thresholds", [])
            metrics_payload["thresholds"]["default"][horizon] = (ensemble_default_metrics or {}).get("thresholds", [])

            # Baseline bucket proportions from training labels
            churn_labels = (ensemble_churn_metrics or {}).get("labels_sample") or []
            default_labels = (ensemble_default_metrics or {}).get("labels_sample") or []
            if churn_labels:
                metrics_payload["baseline_distributions"][f"investor_churn_{horizon}"] = {
                    "bucket_proportions": {
                        "Low": (ensemble_churn_metrics or {}).get("bucket_low", 0),
                        "Medium": (ensemble_churn_metrics or {}).get("bucket_med", 0),
                        "High": (ensemble_churn_metrics or {}).get("bucket_high", 0),
                    }
                }
            if default_labels:
                metrics_payload["baseline_distributions"][f"loan_default_{horizon}"] = {
                    "bucket_proportions": {
                        "Low": (ensemble_default_metrics or {}).get("bucket_low", 0),
                        "Medium": (ensemble_default_metrics or {}).get("bucket_med", 0),
                        "High": (ensemble_default_metrics or {}).get("bucket_high", 0),
                    }
                }

        # Portfolio/cohort summaries and single-feature baselines
        segment_payload = segment_risk_summary(session)
        metrics_payload["segment_risk"] = segment_payload.get("segment_averages", {})
        metrics_payload["segment_heatmaps"] = segment_payload.get("heatmaps", {})
        metrics_payload["cohort_models"] = segment_payload.get("cohort_models", {})
        metrics_payload["segment_features"] = segment_payload.get("feature_stats", {})
        metrics_payload["single_feature_benchmarks"] = single_feature_benchmarks(session)
        metrics_payload["sensitivity"] = compute_sensitivity(session)
        metrics_payload["baseline_feature_stats"] = feature_stats(session)
        metrics_payload["missing_data"] = imputation_benchmarks(session)
        metrics_payload["bootstrap_ci"] = {
            "churn": bootstrap_metric_ci(session, "churn", horizon="6m"),
            "default": bootstrap_metric_ci(session, "default", horizon="6m"),
        }
        metrics_payload["bootstrap_coefficients"] = {
            "churn": bootstrap_coefficients(session, "churn", horizon="6m"),
            "default": bootstrap_coefficients(session, "default", horizon="6m"),
        }
        metrics_payload["diagnostics_1d"] = diagnostics_1d_curves(session)
        metrics_payload["model_selection"] = model_selection_summary(session)
        metrics_payload["non_linear"] = non_linear_patterns(session)
        metrics_payload["correlation"] = correlation_summary(session)
        metrics_payload["imbalance"] = imbalance_summary(session)
        metrics_payload["robustness"] = robustness_summary(session)
        metrics_payload["segment_roc"] = segment_roc_summary(session)
        metrics_payload["eda"] = eda_bundle(session)
        metrics_payload["forecaster"] = regression_forecaster(session)
        metrics_payload["surfaces"] = scenario_surfaces(session)
        metrics_payload["regression_kpi"] = regression_kpis(session)
        metrics_payload["contributions"] = {
            "churn": per_instance_contributions(session, "churn"),
            "default": per_instance_contributions(session, "default"),
        }
        metrics_payload["segment_bias"] = segment_roc_bias(session)
        metrics_payload["voting_importance"] = voting_importance(session)

        model_dir = Path(list(CHURN_MODEL_PATHS.values())[0]).parent
        model_dir.mkdir(exist_ok=True)
        metrics_path = model_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))

        print("Churn model metrics:", metrics_payload["churn"])
        print("Default model metrics:", metrics_payload["default"])
        print(f"Saved metrics to {metrics_path}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

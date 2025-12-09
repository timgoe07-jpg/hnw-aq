from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from models import Investor, Loan
from utils_modeling import cross_val_metrics

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
HORIZONS = ["3m", "6m", "12m"]
CHURN_MODEL_PATHS = {h: MODEL_DIR / f"churn_model_{h}.joblib" for h in HORIZONS}
DEFAULT_MODEL_PATHS = {h: MODEL_DIR / f"default_model_{h}.joblib" for h in HORIZONS}
MODEL_FAMILIES = ["ensemble", "logreg", "rf", "adaboost", "knn", "bagging"]
# Full list used by training script (kept for backward compat)
MODEL_FAMILIES_FULL = MODEL_FAMILIES

# Family-specific model artifacts (reuse legacy paths for ensemble)
CHURN_FAMILY_PATHS = {fam: {h: MODEL_DIR / f"churn_{fam}_{h}.joblib" for h in HORIZONS} for fam in MODEL_FAMILIES}
DEFAULT_FAMILY_PATHS = {fam: {h: MODEL_DIR / f"default_{fam}_{h}.joblib" for h in HORIZONS} for fam in MODEL_FAMILIES}
CHURN_FAMILY_PATHS["ensemble"] = CHURN_MODEL_PATHS
DEFAULT_FAMILY_PATHS["ensemble"] = DEFAULT_MODEL_PATHS

INVESTOR_FEATURES = [
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
INVESTOR_CATEGORICAL = ["risk_tolerance"]
INVESTOR_NUMERIC = [f for f in INVESTOR_FEATURES if f not in INVESTOR_CATEGORICAL]

LOAN_FEATURES = [
    "amount",
    "ltv_ratio",
    "term_months",
    "sector",
    "arrears_flag",
    "dscr",
    "covenants_flag",
    "collateral_score",
]
LOAN_CATEGORICAL = ["sector", "arrears_flag", "covenants_flag"]
LOAN_NUMERIC = [f for f in LOAN_FEATURES if f not in LOAN_CATEGORICAL]


def _horizon_factor(horizon: str) -> float:
    return {"3m": 0.7, "6m": 1.0, "12m": 1.2}.get(horizon, 1.0)


def _heuristic_churn_probability(row: Dict, horizon: str) -> float:
    score = 0.1 * _horizon_factor(horizon)
    score += max(0, (50 - row.get("engagement_score", 50)) / 200)
    score += max(0, (0.4 - row.get("email_open_rate", 0.4)) * 0.8)
    score += 0.08 if row.get("risk_tolerance") == "low" else 0
    score += 0.05 if row.get("call_frequency", 0) < 2 else 0
    score += 0.08 if row.get("inactivity_days", 0) > 60 else 0
    score += 0.1 if row.get("redemption_intent") else 0
    score -= 0.03 if row.get("meetings_last_quarter", 0) >= 3 else 0
    score -= 0.02 if row.get("distribution_yield", 0.0) >= 0.08 else 0
    return float(max(0, min(score, 0.95)))


def _heuristic_default_probability(row: Dict, horizon: str) -> float:
    score = 0.05 * _horizon_factor(horizon)
    score += max(0, (row.get("ltv_ratio", 0.5) - 0.5)) * 0.6
    score += 0.15 if row.get("arrears_flag") else 0
    score += 0.1 if row.get("sector") in {"property", "hospitality"} else 0
    score += 0.05 if row.get("amount", 0) > 1_500_000 else 0
    score += 0.12 if row.get("dscr", 1.2) < 1.0 else 0
    score += 0.1 if row.get("covenants_flag") else 0
    score += max(0, 0.6 - row.get("collateral_score", 0.6)) * 0.4
    return float(max(0, min(score, 0.95)))


def _ensure_horizon_labels(df: pd.DataFrame, base_col: str, heuristic_fn) -> pd.DataFrame:
    """
    Ensure horizon-specific labels exist and are non-degenerate.
    Heuristic is called per row with the horizon string.
    """
    for horizon in HORIZONS:
        col = f"{base_col}_{horizon}"
        if col not in df.columns:
            df[col] = None

        missing_mask = df[col].isna()
        if missing_mask.all() or len(df[col].dropna().unique()) <= 1:
            df[col] = df.apply(lambda r, h=horizon: int(random.random() < heuristic_fn(r, h)), axis=1)
            continue

        df.loc[missing_mask, col] = df[missing_mask].apply(
            lambda r, h=horizon: int(random.random() < heuristic_fn(r, h)), axis=1
        )
        if df[col].nunique() == 1:
            df.loc[df.index[0], col] = 1 - int(df.loc[df.index[0], col])
    return df


def _build_investor_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL),
            ("numeric", StandardScaler(), INVESTOR_NUMERIC),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(random_state=42)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[0.6, 0.4],
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", ensemble)])


def _build_investor_logreg_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL),
            ("numeric", StandardScaler(), INVESTOR_NUMERIC),
        ]
    )
    log_reg = LogisticRegression(max_iter=200, class_weight="balanced")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", log_reg)])


def _build_investor_rf_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL),
            ("numeric", StandardScaler(), INVESTOR_NUMERIC),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=120, random_state=42, class_weight="balanced_subsample", n_jobs=-1, max_depth=None
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])


def _build_investor_adaboost_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL),
            ("numeric", StandardScaler(), INVESTOR_NUMERIC),
        ]
    )
    base = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf = AdaBoostClassifier(estimator=base, n_estimators=140, learning_rate=0.4, random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def _build_investor_knn_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL),
            ("numeric", StandardScaler(), INVESTOR_NUMERIC),
        ]
    )
    clf = KNeighborsClassifier(n_neighbors=20, weights="distance")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def _build_loan_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), LOAN_CATEGORICAL),
            ("numeric", StandardScaler(), LOAN_NUMERIC),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=180,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(random_state=42)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[0.6, 0.4],
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", ensemble)])


def _build_loan_logreg_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), LOAN_CATEGORICAL),
            ("numeric", StandardScaler(), LOAN_NUMERIC),
        ]
    )
    log_reg = LogisticRegression(max_iter=200, class_weight="balanced")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", log_reg)])


def _build_loan_rf_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), LOAN_CATEGORICAL),
            ("numeric", StandardScaler(), LOAN_NUMERIC),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=140, random_state=42, class_weight="balanced_subsample", n_jobs=-1, max_depth=None
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])


def _build_loan_adaboost_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), LOAN_CATEGORICAL),
            ("numeric", StandardScaler(), LOAN_NUMERIC),
        ]
    )
    base = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf = AdaBoostClassifier(estimator=base, n_estimators=150, learning_rate=0.35, random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def _build_loan_knn_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), LOAN_CATEGORICAL),
            ("numeric", StandardScaler(), LOAN_NUMERIC),
        ]
    )
    clf = KNeighborsClassifier(n_neighbors=20, weights="distance")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


CHURN_BUILDERS = {
    "ensemble": _build_investor_pipeline,
    "logreg": _build_investor_logreg_pipeline,
    "rf": _build_investor_rf_pipeline,
    "adaboost": _build_investor_adaboost_pipeline,
    "knn": _build_investor_knn_pipeline,
    "bagging": lambda: Pipeline(steps=[
        ("preprocess", ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL),
                ("numeric", StandardScaler(), INVESTOR_NUMERIC),
            ]
        )),
        ("model", BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=50,
            random_state=42,
        ))
    ]),
}

DEFAULT_BUILDERS = {
    "ensemble": _build_loan_pipeline,
    "logreg": _build_loan_logreg_pipeline,
    "rf": _build_loan_rf_pipeline,
    "adaboost": _build_loan_adaboost_pipeline,
    "knn": _build_loan_knn_pipeline,
    "bagging": lambda: Pipeline(steps=[
        ("preprocess", ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), LOAN_CATEGORICAL),
                ("numeric", StandardScaler(), LOAN_NUMERIC),
            ]
        )),
        ("model", BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=60,
            random_state=42,
        ))
    ]),
}


def _basic_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "avg_precision": float(average_precision_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
    }


def _calibration_summary(model: Pipeline, X, y) -> Dict[str, List[float]]:
    try:
        prob_pos = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, prob_pos, n_bins=10, strategy="uniform")
        return {"fraction_positives": frac_pos.tolist(), "mean_predictions": mean_pred.tolist()}
    except Exception:
        return {}


def _threshold_grid(model: Pipeline, X, y, grid=None) -> List[Dict[str, float]]:
    if grid is None:
        grid = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]
    out = []
    try:
        prob = model.predict_proba(X)[:, 1]
        total = len(y)
        for thr in grid:
            y_pred = (prob >= thr).astype(int)
            tp = int(((y == 1) & (y_pred == 1)).sum())
            fp = int(((y == 0) & (y_pred == 1)).sum())
            fn = int(((y == 1) & (y_pred == 0)).sum())
            tn = int(((y == 0) & (y_pred == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tpr
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            pos_rate = (tp + fp) / total if total else 0.0
            out.append(
                {
                    "threshold": float(thr),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "positive_rate": float(pos_rate),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "total": total,
                }
            )
    except Exception:
        return []
    return out


def _fit_and_score_pipeline(
    pipeline: Pipeline,
    X,
    y,
    categorical: List[str],
    numeric: List[str],
):
    """Train pipeline, then compute a common set of metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = _basic_metrics(y_test, y_pred)
    metrics["feature_importance"] = _permutation_importance(pipeline, X_test, y_test, categorical, numeric)
    metrics.update(cross_val_metrics(pipeline, X, y))
    metrics["calibration"] = _calibration_summary(pipeline, X_test, y_test)
    metrics["thresholds"] = _threshold_grid(pipeline, X_test, y_test)
    return pipeline, metrics


def _bucket_proportions_from_labels(labels: pd.Series) -> Dict[str, float]:
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for val in labels:
        prob = val if isinstance(val, (float, int)) else 0
        bucket = risk_bucket(prob)
        counts[bucket] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _investor_dataframe(session) -> pd.DataFrame:
    investors: List[Investor] = session.query(Investor).all()
    data = [
        {
            "id": i.id,
            "age": i.age,
            "aum": i.aum,
            "risk_tolerance": i.risk_tolerance,
            "engagement_score": i.engagement_score,
            "email_open_rate": i.email_open_rate,
            "call_frequency": i.call_frequency,
            "inactivity_days": i.inactivity_days,
            "redemption_intent": bool(i.redemption_intent),
            "distribution_yield": i.distribution_yield,
            "meetings_last_quarter": i.meetings_last_quarter,
            "churn_label": i.churn_label,
        }
        for i in investors
    ]
    return pd.DataFrame(data)


def _loan_dataframe(session) -> pd.DataFrame:
    loans: List[Loan] = session.query(Loan).all()
    data = [
        {
            "id": l.id,
            "investor_id": l.investor_id,
            "amount": l.amount,
            "ltv_ratio": l.ltv_ratio,
            "term_months": l.term_months,
            "sector": l.sector,
            "arrears_flag": bool(l.arrears_flag),
            "dscr": l.dscr,
            "covenants_flag": bool(l.covenants_flag),
            "collateral_score": l.collateral_score,
            "default_label": l.default_label,
        }
        for l in loans
    ]
    return pd.DataFrame(data)


def train_churn_model(session, horizon: str = "6m") -> Tuple[Pipeline, Dict[str, float]]:
    df = _investor_dataframe(session)
    if df.empty:
        raise ValueError("No investor data available for training.")

    df = _ensure_horizon_labels(df, "churn_label", _heuristic_churn_probability)
    label_col = f"churn_label_{horizon}"
    X = df[INVESTOR_FEATURES]
    y = df[label_col].astype(int)

    pipeline, metrics = _fit_and_score_pipeline(_build_investor_pipeline(), X, y, INVESTOR_CATEGORICAL, INVESTOR_NUMERIC)
    # store bucket proportions as baseline
    buckets = _bucket_proportions_from_labels(y)
    metrics["bucket_low"] = buckets.get("Low", 0)
    metrics["bucket_med"] = buckets.get("Medium", 0)
    metrics["bucket_high"] = buckets.get("High", 0)
    metrics["labels_sample"] = y.head(20).tolist()
    joblib.dump(pipeline, CHURN_MODEL_PATHS[horizon])
    return pipeline, metrics


def train_default_model(session, horizon: str = "6m") -> Tuple[Pipeline, Dict[str, float]]:
    df = _loan_dataframe(session)
    if df.empty:
        raise ValueError("No loan data available for training.")

    df = _ensure_horizon_labels(df, "default_label", _heuristic_default_probability)
    label_col = f"default_label_{horizon}"
    X = df[LOAN_FEATURES]
    y = df[label_col].astype(int)

    pipeline, metrics = _fit_and_score_pipeline(_build_loan_pipeline(), X, y, LOAN_CATEGORICAL, LOAN_NUMERIC)
    buckets = _bucket_proportions_from_labels(y)
    metrics["bucket_low"] = buckets.get("Low", 0)
    metrics["bucket_med"] = buckets.get("Medium", 0)
    metrics["bucket_high"] = buckets.get("High", 0)
    metrics["labels_sample"] = y.head(20).tolist()
    joblib.dump(pipeline, DEFAULT_MODEL_PATHS[horizon])
    return pipeline, metrics


def risk_bucket(probability: float) -> str:
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Medium"
    return "Low"


def _permutation_importance(model: Pipeline, X, y, categorical: List[str], numeric: List[str]) -> Dict[str, float]:
    """Compute normalized permutation importance for reporting."""
    try:
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1)
        preprocess: ColumnTransformer = model.named_steps["preprocess"]
        ohe: OneHotEncoder = preprocess.named_transformers_["categorical"]
        cat_features = ohe.get_feature_names_out(categorical) if hasattr(ohe, "get_feature_names_out") else categorical
        feature_names = list(numeric) + list(cat_features)
        importance = {name: float(max(0, imp)) for name, imp in zip(feature_names, result.importances_mean)}
        importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:8])
        return importance
    except Exception:
        return {}


def feature_importance_summary(session) -> Dict[str, List[Dict[str, float]]]:
    """
    Compute permutation importance for churn/default using the main pipeline (6m horizon).
    """
    out: Dict[str, List[Dict[str, float]]] = {}
    inv = _investor_dataframe(session)
    if not inv.empty:
        inv = _ensure_horizon_labels(inv, "churn_label", _heuristic_churn_probability)
        y = inv["churn_label_6m"]
        X = inv[INVESTOR_FEATURES]
        if y.nunique() > 1:
            pipe = _build_investor_pipeline()
            pipe.fit(X, y)
            imp = _permutation_importance(pipe, X, y, INVESTOR_CATEGORICAL, INVESTOR_NUMERIC)
            out["churn"] = [{"feature": k, "importance": v} for k, v in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)]
    loans = _loan_dataframe(session)
    if not loans.empty:
        loans = _ensure_horizon_labels(loans, "default_label", _heuristic_default_probability)
        y = loans["default_label_6m"]
        X = loans[LOAN_FEATURES]
        if y.nunique() > 1:
            pipe = _build_loan_pipeline()
            pipe.fit(X, y)
            imp = _permutation_importance(pipe, X, y, LOAN_CATEGORICAL, LOAN_NUMERIC)
            out["default"] = [{"feature": k, "importance": v} for k, v in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)]
    return out


def explain_investor_risk(features: Dict, probability: float) -> str:
    reasons = []
    if features.get("engagement_score", 0) < 45:
        reasons.append("low engagement score")
    if features.get("email_open_rate", 1) < 0.3:
        reasons.append("weak email response")
    if features.get("call_frequency", 0) < 2:
        reasons.append("infrequent calls")
    if features.get("risk_tolerance") == "low":
        reasons.append("conservative risk tolerance")
    if features.get("aum", 0) > 2_000_000:
        reasons.append("large AUM that may expect more touchpoints")

    if not reasons:
        return "Stable profile with healthy engagement and communication."
    descriptor = "Higher churn risk" if probability >= 0.5 else "Moderate churn risk"
    return f"{descriptor} driven by {', '.join(reasons[:3])}."


def explain_loan_risk(features: Dict, probability: float) -> str:
    reasons = []
    if features.get("ltv_ratio", 0) > 0.7:
        reasons.append("elevated LTV")
    if features.get("arrears_flag"):
        reasons.append("arrears history")
    if features.get("amount", 0) > 1_500_000:
        reasons.append("large exposure")
    if features.get("sector") in {"property", "hospitality"}:
        reasons.append(f"sector concentration in {features.get('sector')}")
    if features.get("term_months", 0) > 60:
        reasons.append("long tenor")

    if not reasons:
        return "Limited default signals based on current factors."
    descriptor = "Higher default risk" if probability >= 0.5 else "Moderate default risk"
    return f"{descriptor} linked to {', '.join(reasons[:3])}."


def load_or_train_model(path: Path, train_fn, session) -> Pipeline:
    if path.exists():
        return joblib.load(path)
    model, _ = train_fn(session)
    return model


def load_or_train_horizon(path_map: Dict[str, Path], train_fn, session, horizon: str) -> Pipeline:
    path = path_map[horizon]
    if path.exists():
        return joblib.load(path)
    model, _ = train_fn(session, horizon=horizon)
    return model


def _evaluate_family(
    build_fn,
    X,
    y,
    categorical: List[str],
    numeric: List[str],
) -> Dict[str, float]:
    """Reusable evaluator for alternative model families (logreg, RF, etc.)."""
    if len(y) == 0:
        return {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    pipeline = build_fn()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = _basic_metrics(y_test, y_pred)
    metrics["feature_importance"] = _permutation_importance(pipeline, X_test, y_test, categorical, numeric)
    metrics.update(cross_val_metrics(pipeline, X, y))
    metrics["calibration"] = _calibration_summary(pipeline, X_test, y_test)
    metrics["thresholds"] = _threshold_grid(pipeline, X_test, y_test)
    buckets = _bucket_proportions_from_labels(y)
    metrics["bucket_low"] = buckets.get("Low", 0)
    metrics["bucket_med"] = buckets.get("Medium", 0)
    metrics["bucket_high"] = buckets.get("High", 0)
    return metrics


def evaluate_churn_family(session, horizon: str, family: str) -> Dict[str, float]:
    df = _investor_dataframe(session)
    df = _ensure_horizon_labels(df, "churn_label", _heuristic_churn_probability)
    y = df[f"churn_label_{horizon}"].astype(int)
    X = df[INVESTOR_FEATURES]
    builder = CHURN_BUILDERS.get(family, _build_investor_pipeline)
    return _evaluate_family(builder, X, y, INVESTOR_CATEGORICAL, INVESTOR_NUMERIC)


def evaluate_default_family(session, horizon: str, family: str) -> Dict[str, float]:
    df = _loan_dataframe(session)
    df = _ensure_horizon_labels(df, "default_label", _heuristic_default_probability)
    y = df[f"default_label_{horizon}"].astype(int)
    X = df[LOAN_FEATURES]
    builder = DEFAULT_BUILDERS.get(family, _build_loan_pipeline)
    return _evaluate_family(builder, X, y, LOAN_CATEGORICAL, LOAN_NUMERIC)


def train_churn_family_model(session, horizon: str, family: str) -> Tuple[Pipeline, Dict[str, float]]:
    df = _investor_dataframe(session)
    if df.empty:
        raise ValueError("No investor data available for training.")
    df = _ensure_horizon_labels(df, "churn_label", _heuristic_churn_probability)
    label_col = f"churn_label_{horizon}"
    X = df[INVESTOR_FEATURES]
    y = df[label_col].astype(int)
    builder = CHURN_BUILDERS.get(family, _build_investor_pipeline)
    pipeline, metrics = _fit_and_score_pipeline(builder(), X, y, INVESTOR_CATEGORICAL, INVESTOR_NUMERIC)
    buckets = _bucket_proportions_from_labels(y)
    metrics["bucket_low"] = buckets.get("Low", 0)
    metrics["bucket_med"] = buckets.get("Medium", 0)
    metrics["bucket_high"] = buckets.get("High", 0)
    metrics["labels_sample"] = y.head(20).tolist()
    path_map = CHURN_FAMILY_PATHS.get(family, CHURN_MODEL_PATHS)
    joblib.dump(pipeline, path_map[horizon])
    return pipeline, metrics


def train_default_family_model(session, horizon: str, family: str) -> Tuple[Pipeline, Dict[str, float]]:
    df = _loan_dataframe(session)
    if df.empty:
        raise ValueError("No loan data available for training.")
    df = _ensure_horizon_labels(df, "default_label", _heuristic_default_probability)
    label_col = f"default_label_{horizon}"
    X = df[LOAN_FEATURES]
    y = df[label_col].astype(int)
    builder = DEFAULT_BUILDERS.get(family, _build_loan_pipeline)
    pipeline, metrics = _fit_and_score_pipeline(builder(), X, y, LOAN_CATEGORICAL, LOAN_NUMERIC)
    buckets = _bucket_proportions_from_labels(y)
    metrics["bucket_low"] = buckets.get("Low", 0)
    metrics["bucket_med"] = buckets.get("Medium", 0)
    metrics["bucket_high"] = buckets.get("High", 0)
    metrics["labels_sample"] = y.head(20).tolist()
    path_map = DEFAULT_FAMILY_PATHS.get(family, DEFAULT_MODEL_PATHS)
    joblib.dump(pipeline, path_map[horizon])
    return pipeline, metrics


def load_family_model(problem: str, family: str, session, horizon: str) -> Pipeline:
    """Load or train a model for the requested family and horizon."""
    if horizon not in HORIZONS:
        horizon = "6m"
    fam = family if family in MODEL_FAMILIES else "ensemble"
    if problem == "churn":
        path = CHURN_FAMILY_PATHS.get(fam, CHURN_MODEL_PATHS)[horizon]
        if path.exists():
            return joblib.load(path)
        if fam == "ensemble":
            model, _ = train_churn_model(session, horizon=horizon)
        else:
            model, _ = train_churn_family_model(session, horizon=horizon, family=fam)
        return model

    path = DEFAULT_FAMILY_PATHS.get(fam, DEFAULT_MODEL_PATHS)[horizon]
    if path.exists():
        return joblib.load(path)
    if fam == "ensemble":
        model, _ = train_default_model(session, horizon=horizon)
    else:
        model, _ = train_default_family_model(session, horizon=horizon, family=fam)
    return model


def _aum_band_val(aum: float) -> str:
    if aum < 1_000_000:
        return "<1M"
    if aum < 2_000_000:
        return "1-2M"
    if aum < 5_000_000:
        return "2-5M"
    return "5M+"


def _ltv_band_val(ltv: float) -> str:
    if ltv < 0.4:
        return "<40%"
    if ltv < 0.6:
        return "40-60%"
    if ltv < 0.8:
        return "60-80%"
    return "80%+"


def _assign_broker(name: str | None, idx: int) -> str:
    if name:
        key = name.strip().split(" ")[0].lower() if " " in name else name.lower()
        return f"Broker_{key[:3]}"
    return f"Broker_{idx % 5}"


def segment_risk_summary(session) -> Dict[str, Dict]:
    """Compute simple cohort-level averages and heatmaps for portfolio views."""
    investors = _investor_dataframe(session)
    loans = _loan_dataframe(session)
    out: Dict[str, Dict] = {"churn": {}, "default": {}}
    cohort_models: Dict[str, Dict] = {"churn": {}, "default": {}}
    heatmaps: Dict[str, Dict] = {}
    feature_stats: Dict[str, Dict] = {"churn": {}, "default": {}}
    # Map investor id -> pseudo broker
    broker_map = {}
    if not investors.empty:
        investors["broker"] = [ _assign_broker(row["name"], idx) if "name" in investors else _assign_broker(str(idx), idx) for idx, row in investors.iterrows() ]
        broker_map = {row["id"]: row["broker"] for _, row in investors.iterrows() if "id" in investors}
    if not investors.empty:
        investors = _ensure_horizon_labels(investors, "churn_label", _heuristic_churn_probability)
        investors["aum_band"] = investors["aum"].apply(_aum_band_val)
        label = investors.get("churn_label_6m") if "churn_label_6m" in investors else investors["churn_label"]
        out["churn"]["risk_tolerance"] = {
            k: {"avg": float(v[label.name].mean()), "bucket": risk_bucket(float(v[label.name].mean()))}
            for k, v in investors.groupby("risk_tolerance")
        }
        out["churn"]["aum_band"] = {
            k: {"avg": float(v[label.name].mean()), "bucket": risk_bucket(float(v[label.name].mean()))}
            for k, v in investors.groupby("aum_band")
        }
        # heatmap: risk tolerance x AUM band averages
        risk_order = sorted(investors["risk_tolerance"].unique())
        aum_order = sorted(
            investors["aum_band"].unique(),
            key=lambda x: ["<1M", "1-2M", "2-5M", "5M+"].index(x)
            if x in ["<1M", "1-2M", "2-5M", "5M+"]
            else 99,
        )
        pivot = (
            investors.pivot_table(values=label.name, index="risk_tolerance", columns="aum_band", aggfunc="mean", fill_value=0.0)
            .reindex(index=risk_order, columns=aum_order, fill_value=0.0)
        )
        heatmaps["churn_risk_tolerance_by_aum"] = {
            "rows": list(pivot.index),
            "cols": list(pivot.columns),
            "values": pivot.values.round(4).tolist(),
        }
        # broker x tolerance heatmap (avg churn)
        pivot_broker = investors.pivot_table(values=label.name, index="broker", columns="risk_tolerance", aggfunc="mean", fill_value=0.0)
        heatmaps["churn_broker_by_tolerance"] = {
            "rows": list(pivot_broker.index),
            "cols": list(pivot_broker.columns),
            "values": pivot_broker.values.round(4).tolist(),
        }
        # simple cohort model: mean engagement/open rate per tolerance -> risk
        for tol, grp in investors.groupby("risk_tolerance"):
            mean_eng = float(grp["engagement_score"].mean())
            mean_open = float(grp["email_open_rate"].mean())
            prob = float(_heuristic_churn_probability({"engagement_score": mean_eng, "email_open_rate": mean_open, "risk_tolerance": tol, "call_frequency": grp["call_frequency"].mean()}, "6m"))
            cohort_models["churn"][tol] = {"probability": prob, "bucket": risk_bucket(prob)}
        feature_stats["churn"]["risk_tolerance"] = {
            tol: {
                "mean_engagement": float(grp["engagement_score"].mean()),
                "mean_call_freq": float(grp["call_frequency"].mean()),
                "median_aum": float(grp["aum"].median()),
            }
            for tol, grp in investors.groupby("risk_tolerance")
        }
    if not loans.empty:
        loans = _ensure_horizon_labels(loans, "default_label", _heuristic_default_probability)
        loans["ltv_band"] = loans["ltv_ratio"].apply(_ltv_band_val)
        if broker_map:
            loans["broker"] = loans["investor_id"].map(broker_map).fillna("Broker_unk")
        label = loans.get("default_label_6m") if "default_label_6m" in loans else loans["default_label"]
        out["default"]["sector"] = {
            k: {"avg": float(v[label.name].mean()), "bucket": risk_bucket(float(v[label.name].mean()))}
            for k, v in loans.groupby("sector")
        }
        # heatmap: sector x LTV band averages
        sector_order = sorted(loans["sector"].unique())
        ltv_order = ["<40%", "40-60%", "60-80%", "80%+"]
        pivot_default = (
            loans.pivot_table(values=label.name, index="sector", columns="ltv_band", aggfunc="mean", fill_value=0.0)
            .reindex(index=sector_order, columns=ltv_order, fill_value=0.0)
        )
        heatmaps["default_sector_by_ltv"] = {
            "rows": list(pivot_default.index),
            "cols": list(pivot_default.columns),
            "values": pivot_default.values.round(4).tolist(),
        }
        if "broker" in loans.columns:
            pivot_broker_sector = loans.pivot_table(values=label.name, index="broker", columns="sector", aggfunc="mean", fill_value=0.0)
            heatmaps["default_broker_by_sector"] = {
                "rows": list(pivot_broker_sector.index),
                "cols": list(pivot_broker_sector.columns),
                "values": pivot_broker_sector.values.round(4).tolist(),
            }
        for sector, grp in loans.groupby("sector"):
            mean_ltv = float(grp["ltv_ratio"].mean())
            mean_dscr = float(grp["dscr"].mean())
            prob = float(_heuristic_default_probability({"ltv_ratio": mean_ltv, "dscr": mean_dscr, "sector": sector}, "6m"))
            cohort_models["default"][sector] = {"probability": prob, "bucket": risk_bucket(prob)}
        feature_stats["default"]["sector"] = {
            sector: {
                "mean_ltv": float(grp["ltv_ratio"].mean()),
                "dscr_std": float(grp["dscr"].std() or 0.0),
                "collateral_mean": float(grp["collateral_score"].mean()),
            }
            for sector, grp in loans.groupby("sector")
        }
    return {"segment_averages": out, "cohort_models": cohort_models, "heatmaps": heatmaps, "feature_stats": feature_stats}


def single_feature_benchmarks(session) -> Dict[str, Dict]:
    """Compute simple single-feature baselines (AUC) to show intrinsic signal strength."""
    benchmarks = {"churn": {}, "default": {}}
    churn_df = _investor_dataframe(session)
    if not churn_df.empty:
        churn_df = _ensure_horizon_labels(churn_df, "churn_label", _heuristic_churn_probability)
        y = churn_df["churn_label_6m"]
        for feat in ["engagement_score", "inactivity_days", "email_open_rate", "call_frequency", "aum", "risk_tolerance"]:
            X = churn_df[[feat]]
            benchmarks["churn"][feat] = _single_feature_auc(X, y, feat, problem="churn")
    default_df = _loan_dataframe(session)
    if not default_df.empty:
        default_df = _ensure_horizon_labels(default_df, "default_label", _heuristic_default_probability)
        y = default_df["default_label_6m"]
        for feat in ["ltv_ratio", "dscr", "collateral_score", "amount", "sector"]:
            X = default_df[[feat]]
            benchmarks["default"][feat] = _single_feature_auc(X, y, feat, problem="default")
    return benchmarks


def diagnostics_1d_curves(session) -> Dict[str, Dict]:
    """
    Produce simple 1D curves comparing logistic vs k-NN for a key feature.
    Churn: engagement_score. Default: ltv_ratio.
    """
    out: Dict[str, Dict] = {}
    churn_df = _investor_dataframe(session)
    if not churn_df.empty:
        churn_df = _ensure_horizon_labels(churn_df, "churn_label", _heuristic_churn_probability)
        y = churn_df["churn_label_6m"]
        xfeat = churn_df["engagement_score"]
        if y.nunique() > 1 and xfeat.nunique() > 1:
            grid = np.linspace(float(xfeat.min()), float(xfeat.max()), 25)
            X_train, X_test, y_train, _ = train_test_split(xfeat.values.reshape(-1, 1), y, test_size=0.2, random_state=42, stratify=y)
            log_clf = LogisticRegression(max_iter=200, class_weight="balanced")
            log_clf.fit(X_train, y_train)
            knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
            knn.fit(X_train, y_train)
            log_probs = log_clf.predict_proba(grid.reshape(-1, 1))[:, 1]
            knn_probs = knn.predict_proba(grid.reshape(-1, 1))[:, 1]
            out["churn"] = {"x": grid.tolist(), "logistic": log_probs.tolist(), "knn": knn_probs.tolist()}

    def_df = _loan_dataframe(session)
    if not def_df.empty:
        def_df = _ensure_horizon_labels(def_df, "default_label", _heuristic_default_probability)
        y = def_df["default_label_6m"]
        xfeat = def_df["ltv_ratio"]
        if y.nunique() > 1 and xfeat.nunique() > 1:
            grid = np.linspace(float(xfeat.min()), float(xfeat.max()), 25)
            X_train, X_test, y_train, _ = train_test_split(xfeat.values.reshape(-1, 1), y, test_size=0.2, random_state=42, stratify=y)
            log_clf = LogisticRegression(max_iter=200, class_weight="balanced")
            log_clf.fit(X_train, y_train)
            knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
            knn.fit(X_train, y_train)
            log_probs = log_clf.predict_proba(grid.reshape(-1, 1))[:, 1]
            knn_probs = knn.predict_proba(grid.reshape(-1, 1))[:, 1]
            out["default"] = {"x": grid.tolist(), "logistic": log_probs.tolist(), "knn": knn_probs.tolist()}
    return out


def model_selection_summary(session) -> Dict[str, Dict]:
    """
    Small hyperparameter sweep for RF/GB to show train/val/CV curves and seed robustness.
    """
    summary: Dict[str, Dict] = {}
    seeds = [1, 7, 21]
    for problem in ["churn", "default"]:
        df = _investor_dataframe(session) if problem == "churn" else _loan_dataframe(session)
        if df.empty:
            continue
        df = _ensure_horizon_labels(df, "churn_label" if problem == "churn" else "default_label",
                                    _heuristic_churn_probability if problem == "churn" else _heuristic_default_probability)
        label_col = "churn_label_6m" if problem == "churn" else "default_label_6m"
        y = df[label_col]
        X = df[INVESTOR_FEATURES] if problem == "churn" else df[LOAN_FEATURES]
        if y.nunique() <= 1:
            continue
        params = [50, 120, 200]
        train_scores = []
        val_scores = []
        cv_scores = []
        seed_pref = []
        for n_est in params:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL if problem == "churn" else LOAN_CATEGORICAL),
                    ("numeric", StandardScaler(), INVESTOR_NUMERIC if problem == "churn" else LOAN_NUMERIC),
                ]
            )
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight="balanced_subsample", n_jobs=-1, max_depth=None)
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
            )
            pipe.fit(X_train, y_train)
            prob_train = pipe.predict_proba(X_train)[:, 1]
            prob_val = pipe.predict_proba(X_val)[:, 1]
            train_scores.append(float(roc_auc_score(y_train, prob_train)) if y_train.nunique() > 1 else 0.0)
            val_scores.append(float(roc_auc_score(y_val, prob_val)) if y_val.nunique() > 1 else 0.0)
            cv_run = []
            for seed in seeds:
                try:
                    cv = cross_val_metrics(pipe, X, y, cv_splits=5).get("cv_roc_auc", 0.0)
                except Exception:
                    cv = 0.0
                cv_run.append(float(cv))
            cv_scores.append(float(np.mean(cv_run)))
            seed_pref.append(cv_run)

        # Gradient Boosting learning rate sweep (light)
        gb_params = [0.05, 0.1, 0.2]
        gb_train = []
        gb_val = []
        gb_cv = []
        for lr in gb_params:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), INVESTOR_CATEGORICAL if problem == "churn" else LOAN_CATEGORICAL),
                    ("numeric", StandardScaler(), INVESTOR_NUMERIC if problem == "churn" else LOAN_NUMERIC),
                ]
            )
            gb = GradientBoostingClassifier(random_state=42, learning_rate=lr)
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", gb)])
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
            )
            pipe.fit(X_train, y_train)
            prob_train = pipe.predict_proba(X_train)[:, 1]
            prob_val = pipe.predict_proba(X_val)[:, 1]
            gb_train.append(float(roc_auc_score(y_train, prob_train)) if y_train.nunique() > 1 else 0.0)
            gb_val.append(float(roc_auc_score(y_val, prob_val)) if y_val.nunique() > 1 else 0.0)
            cv_run = []
            for seed in seeds:
                try:
                    cv = cross_val_metrics(pipe, X, y, cv_splits=5).get("cv_roc_auc", 0.0)
                except Exception:
                    cv = 0.0
                cv_run.append(float(cv))
            gb_cv.append(float(np.mean(cv_run)))

        summary[problem] = {
            "rf_estimators": {"params": params, "train": train_scores, "val": val_scores, "cv": cv_scores, "seed_runs": seed_pref},
            "gb_learning_rate": {"params": gb_params, "train": gb_train, "val": gb_val, "cv": gb_cv},
            "seeds": seeds,
        }
    return summary


def non_linear_patterns(session) -> Dict[str, Dict]:
    """Binned empirical risk curves for key continuous features (show non-linearities)."""
    curves: Dict[str, Dict] = {}
    churn_df = _investor_dataframe(session)
    if not churn_df.empty:
        churn_df = _ensure_horizon_labels(churn_df, "churn_label", _heuristic_churn_probability)
        for feat in ["engagement_score", "inactivity_days"]:
            bins = np.linspace(churn_df[feat].min(), churn_df[feat].max(), 8)
            churn_df["bin"] = pd.cut(churn_df[feat], bins=bins, include_lowest=True)
            grouped = churn_df.groupby("bin")["churn_label_6m"].mean()
            centers = [float(interval.mid) for interval in grouped.index]
            curves.setdefault("churn", {})[feat] = {"x": centers, "y": grouped.tolist()}
    loan_df = _loan_dataframe(session)
    if not loan_df.empty:
        loan_df = _ensure_horizon_labels(loan_df, "default_label", _heuristic_default_probability)
        for feat in ["ltv_ratio", "dscr"]:
            bins = np.linspace(loan_df[feat].min(), loan_df[feat].max(), 8)
            loan_df["bin"] = pd.cut(loan_df[feat], bins=bins, include_lowest=True)
            grouped = loan_df.groupby("bin")["default_label_6m"].mean()
            centers = [float(interval.mid) for interval in grouped.index]
            curves.setdefault("default", {})[feat] = {"x": centers, "y": grouped.tolist()}
    return curves


def regression_forecaster(session) -> Dict[str, Dict]:
    """
    Simple linear forecaster for engagement_score_next and yield_next (synthetic) to power lightweight regression cards.
    Uses current investor features to simulate next-period engagement and yield targets.
    """
    inv_df = _investor_dataframe(session)
    if inv_df.empty:
        return {}

    df = inv_df.copy()
    rng = np.random.default_rng(seed=42)
    # Synthetic next-period engagement target driven by current signals + noise
    df["engagement_next"] = (
        df["engagement_score"] * 0.85
        + df["email_open_rate"] * 35
        + df["call_frequency"] * 3
        - df["inactivity_days"] * 0.25
        + rng.normal(0, 5, size=len(df))
    )
    features = ["engagement_score", "email_open_rate", "call_frequency", "inactivity_days"]
    X = df[features]
    y = df["engagement_next"]
    if y.nunique() <= 1:
        return {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    coeffs = {feat: float(coef) for feat, coef in zip(features, reg.coef_)}
    # Forecast next yield using AUM, engagement, and current distribution_yield
    df["yield_next"] = df["distribution_yield"] * 0.9 + df["engagement_score"] * 0.0005 + rng.normal(0, 0.005, size=len(df))
    y2 = df["yield_next"]
    X2 = df[["distribution_yield", "aum", "engagement_score"]]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    reg2 = LinearRegression()
    reg2.fit(X2_train, y2_train)
    preds2 = reg2.predict(X2_test)
    mse2 = float(mean_squared_error(y2_test, preds2))
    rmse2 = float(np.sqrt(mse2))
    mae2 = float(mean_absolute_error(y2_test, preds2))
    r22 = float(r2_score(y2_test, preds2))
    coeffs2 = {feat: float(coef) for feat, coef in zip(X2.columns, reg2.coef_)}

    return {
        "engagement": {
            "target": "engagement_score_next",
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "intercept": float(reg.intercept_),
            "coefficients": coeffs,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "yield": {
            "target": "distribution_yield_next",
            "r2": r22,
            "rmse": rmse2,
            "mae": mae2,
            "intercept": float(reg2.intercept_),
            "coefficients": coeffs2,
            "train_size": int(len(X2_train)),
            "test_size": int(len(X2_test)),
        },
    }


def regression_kpis(session) -> Dict[str, List[Dict]]:
    """
    Small KPI regression table: compare feature sets for engagement and yield forecasts.
    Returns per-target rows with mse/rmse/r2 and coefficients.
    """
    inv_df = _investor_dataframe(session)
    if inv_df.empty:
        return {}
    rng = np.random.default_rng(seed=21)
    df = inv_df.copy()
    df["engagement_next"] = (
        df["engagement_score"] * 0.85
        + df["email_open_rate"] * 35
        + df["call_frequency"] * 3
        - df["inactivity_days"] * 0.25
        + rng.normal(0, 5, size=len(df))
    )
    df["yield_next"] = df["distribution_yield"] * 0.9 + df["engagement_score"] * 0.0005 + rng.normal(0, 0.005, size=len(df))

    def _fit_rows(target: str, feature_sets: List[List[str]]):
        rows = []
        for feats in feature_sets:
            X = df[feats]
            y = df[target]
            if y.nunique() <= 1:
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            mse = float(mean_squared_error(y_test, preds))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, preds))
            mae = float(mean_absolute_error(y_test, preds))
            coeffs = {feat: float(c) for feat, c in zip(feats, reg.coef_)}
            rows.append(
                {
                    "features": feats,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "mae": mae,
                    "coefficients": coeffs,
                }
            )
        return rows

    return {
        "engagement": _fit_rows("engagement_next", [["engagement_score"], ["engagement_score", "email_open_rate", "call_frequency"], ["engagement_score", "inactivity_days"]]),
        "yield": _fit_rows("yield_next", [["distribution_yield"], ["distribution_yield", "engagement_score", "aum"]]),
    }


def segment_roc_bias(session) -> Dict[str, Dict]:
    """
    Lightweight segment overlays and disparity summaries:
    - Churn by risk_tolerance
    - Default by sector
    Computes AUC per segment plus TPR/FPR deltas at 0.5 threshold for bias bars.
    """

    def _metrics(df: pd.DataFrame, label_col: str, segment_col: str, builder):
        rows = []
        if df.empty or df[segment_col].nunique() == 0:
            return rows
        df = df.copy()
        df["segment"] = df[segment_col].astype(str)
        y = df[label_col]
        X = df[INVESTOR_FEATURES] if "churn" in label_col else df[LOAN_FEATURES]
        model = builder()
        if y.nunique() <= 1:
            return rows
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        overall_pred = (probs >= 0.5).astype(int)
        overall_tpr = float(((overall_pred == 1) & (y == 1)).mean()) if (y == 1).sum() else 0.0
        overall_fpr = float(((overall_pred == 1) & (y == 0)).mean()) if (y == 0).sum() else 0.0
        for seg, seg_df in df.groupby("segment"):
            if seg_df[label_col].nunique() <= 1:
                continue
            seg_probs = probs[seg_df.index]
            seg_pred = (seg_probs >= 0.5).astype(int)
            tpr = float(((seg_pred == 1) & (seg_df[label_col] == 1)).mean()) if (seg_df[label_col] == 1).sum() else 0.0
            fpr = float(((seg_pred == 1) & (seg_df[label_col] == 0)).mean()) if (seg_df[label_col] == 0).sum() else 0.0
            rows.append(
                {
                    "segment": seg,
                    "auc": float(roc_auc_score(seg_df[label_col], seg_probs)),
                    "positive_rate": float(seg_pred.mean()),
                    "tpr": tpr,
                    "fpr": fpr,
                    "tpr_gap": tpr - overall_tpr,
                    "fpr_gap": fpr - overall_fpr,
                }
            )
        return rows

    out: Dict[str, Dict] = {}
    inv = _investor_dataframe(session)
    if not inv.empty:
        inv = _ensure_horizon_labels(inv, "churn_label", _heuristic_churn_probability)
        rows = _metrics(inv, "churn_label_6m", "risk_tolerance", _build_investor_pipeline)
        if rows:
            out["churn"] = {
                "by_segment": rows,
                "max_tpr_gap": float(max((r["tpr_gap"] for r in rows), default=0.0)),
                "max_fpr_gap": float(max((r["fpr_gap"] for r in rows), default=0.0)),
            }
    loans = _loan_dataframe(session)
    if not loans.empty:
        loans = _ensure_horizon_labels(loans, "default_label", _heuristic_default_probability)
        rows = _metrics(loans, "default_label_6m", "sector", _build_loan_pipeline)
        if rows:
            out["default"] = {
                "by_segment": rows,
                "max_tpr_gap": float(max((r["tpr_gap"] for r in rows), default=0.0)),
                "max_fpr_gap": float(max((r["fpr_gap"] for r in rows), default=0.0)),
            }
    return out


def voting_importance(session) -> Dict[str, List[Dict]]:
    """
    Combine single-feature benchmarks with ensemble importances to create a simple voting score.
    """
    scores: Dict[str, List[Dict]] = {"churn": [], "default": []}
    benches = single_feature_benchmarks(session)
    importances = feature_importance_summary(session)

    def _blend(problem: str):
        bench = benches.get(problem, {})
        imp = importances.get(problem, {})
        rows = []
        for feat, auc_val in bench.items():
            if isinstance(auc_val, dict):
                auc_val = auc_val.get("roc_auc", 0.0)
            imp_val = 0.0
            if isinstance(imp, list):
                for item in imp:
                    if item.get("feature") == feat:
                        imp_val = item.get("importance", 0.0)
            score = 0.6 * auc_val + 0.4 * imp_val
            rows.append({"feature": feat, "auc": auc_val, "importance": imp_val, "score": score})
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)
        return rows

    scores["churn"] = _blend("churn")
    scores["default"] = _blend("default")
    return scores


def per_instance_contributions(session, problem: str) -> Dict[int, Dict[str, float]]:
    """
    SHAP-lite: use logistic coefficients on standardized features to estimate per-instance contributions.
    Only for interpretable baselines (logistic on top numeric/categorical).
    """
    if problem == "churn":
        df = _investor_dataframe(session)
        if df.empty:
            return {}
        df = _ensure_horizon_labels(df, "churn_label", _heuristic_churn_probability)
        y = df["churn_label_6m"].astype(int)
        X = df[["engagement_score", "email_open_rate", "call_frequency", "inactivity_days", "redemption_intent"]].copy()
        cat_cols = ["redemption_intent"]
    else:
        df = _loan_dataframe(session)
        if df.empty:
            return {}
        df = _ensure_horizon_labels(df, "default_label", _heuristic_default_probability)
        y = df["default_label_6m"].astype(int)
        X = df[["ltv_ratio", "dscr", "amount", "term_months", "covenants_flag", "arrears_flag"]].copy()
        cat_cols = ["covenants_flag", "arrears_flag"]

    # Encode categoricals, scale numerics
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_matrix = ohe.fit_transform(X[cat_cols]) if cat_cols else None
    cat_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
    num_cols = [c for c in X.columns if c not in cat_cols]
    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(X[num_cols]) if num_cols else None
    import numpy as np  # local import to avoid top clutter
    design = np.concatenate([m for m in [num_matrix, cat_matrix] if m is not None], axis=1)
    feature_names = num_cols + cat_names

    if y.nunique() <= 1:
        return {}
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(design, y)
    coefs = clf.coef_[0]
    contribs = {}
    for idx, row in enumerate(design):
        effects = {feature_names[i]: float(row[i] * coefs[i]) for i in range(len(feature_names))}
        contribs[int(df.iloc[idx]["id"])] = effects
    return contribs


def scenario_surfaces(session) -> Dict[str, Dict]:
    """
    Generate 2D surfaces for what-if contours:
    - Churn: engagement_score x inactivity_days
    - Default: ltv_ratio x dscr
    Returns grids with axes and probability values.
    """
    investors = _investor_dataframe(session)
    loans = _loan_dataframe(session)
    surfaces: Dict[str, Dict] = {}

    if not investors.empty:
        investors = _ensure_horizon_labels(investors, "churn_label", _heuristic_churn_probability)
        X = investors[INVESTOR_FEATURES]
        y = investors["churn_label_6m"].astype(int)
        model = _build_investor_pipeline()
        if y.nunique() > 1:
            model.fit(X, y)
            eng_axis = np.linspace(float(X["engagement_score"].min()), float(X["engagement_score"].max()), 20)
            inact_axis = np.linspace(float(X["inactivity_days"].min()), float(X["inactivity_days"].max()), 6)
            grid_vals = []
            for idle in inact_axis:
                rows = []
                for eng in eng_axis:
                    row = X.iloc[0:1].copy()
                    row["engagement_score"] = eng
                    row["inactivity_days"] = idle
                    prob = float(model.predict_proba(row)[:, 1][0])
                    rows.append(prob)
                grid_vals.append(rows)
            surfaces["churn"] = {
                "x": eng_axis.tolist(),
                "y": inact_axis.tolist(),
                "z": grid_vals,
                "x_label": "engagement_score",
                "y_label": "inactivity_days",
            }
    if not loans.empty:
        loans = _ensure_horizon_labels(loans, "default_label", _heuristic_default_probability)
        X = loans[LOAN_FEATURES]
        y = loans["default_label_6m"].astype(int)
        model = _build_loan_pipeline()
        if y.nunique() > 1:
            model.fit(X, y)
            ltv_axis = np.linspace(float(X["ltv_ratio"].min()), float(X["ltv_ratio"].max()), 20)
            dscr_axis = np.linspace(float(X["dscr"].min()), float(X["dscr"].max()), 6)
            grid_vals = []
            for dscr_val in dscr_axis:
                rows = []
                for ltv in ltv_axis:
                    row = X.iloc[0:1].copy()
                    row["ltv_ratio"] = ltv
                    row["dscr"] = dscr_val
                    prob = float(model.predict_proba(row)[:, 1][0])
                    rows.append(prob)
                grid_vals.append(rows)
            surfaces["default"] = {
                "x": ltv_axis.tolist(),
                "y": dscr_axis.tolist(),
                "z": grid_vals,
                "x_label": "ltv_ratio",
                "y_label": "dscr",
            }
    return surfaces


def correlation_summary(session) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    inv_df = _investor_dataframe(session)
    if not inv_df.empty:
        corr = inv_df[INVESTOR_NUMERIC].corr().fillna(0)
        high = []
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i+1:]:
                val = abs(corr.loc[c1, c2])
                if val >= 0.8:
                    high.append({"pair": f"{c1}-{c2}", "corr": float(val)})
        out["churn"] = {"high_corr": sorted(high, key=lambda x: x["corr"], reverse=True)[:5]}
    loan_df = _loan_dataframe(session)
    if not loan_df.empty:
        corr = loan_df[LOAN_NUMERIC].corr().fillna(0)
        high = []
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i+1:]:
                val = abs(corr.loc[c1, c2])
                if val >= 0.8:
                    high.append({"pair": f"{c1}-{c2}", "corr": float(val)})
        out["default"] = {"high_corr": sorted(high, key=lambda x: x["corr"], reverse=True)[:5]}
    return out


def imbalance_summary(session) -> Dict[str, Dict]:
    res = {}
    inv_df = _investor_dataframe(session)
    if not inv_df.empty:
        inv_df = _ensure_horizon_labels(inv_df, "churn_label", _heuristic_churn_probability)
        y = inv_df["churn_label_6m"]
        res["churn"] = {"positive_rate": float(y.mean()), "count": int(len(y))}
        # Simple segment disparity by risk_tolerance
        disparity = {}
        for tol, grp in inv_df.groupby("risk_tolerance"):
            disparity[tol] = float(grp["churn_label_6m"].mean())
        res["churn"]["segment_positive_rate"] = disparity
    loan_df = _loan_dataframe(session)
    if not loan_df.empty:
        loan_df = _ensure_horizon_labels(loan_df, "default_label", _heuristic_default_probability)
        y = loan_df["default_label_6m"]
        res["default"] = {"positive_rate": float(y.mean()), "count": int(len(y))}
        sector_disp = {}
        for sec, grp in loan_df.groupby("sector"):
            sector_disp[sec] = float(grp["default_label_6m"].mean())
        res["default"]["segment_positive_rate"] = sector_disp
    return res


def segment_roc_summary(session) -> Dict[str, Dict]:
    """
    Simple segment-wise ROC AUC for fairness overlays.
    """
    result: Dict[str, Dict] = {}
    inv_df = _investor_dataframe(session)
    if not inv_df.empty:
        inv_df = _ensure_horizon_labels(inv_df, "churn_label", _heuristic_churn_probability)
        X = inv_df[INVESTOR_FEATURES]
        y = inv_df["churn_label_6m"]
        model = _build_investor_pipeline()
        if y.nunique() > 1:
            model.fit(X, y)
            prob = model.predict_proba(X)[:, 1]
            seg_scores = {}
            for tol, grp in inv_df.groupby("risk_tolerance"):
                idx = grp.index
                if y.loc[idx].nunique() > 1:
                    seg_scores[tol] = float(roc_auc_score(y.loc[idx], prob[idx]))
            result["churn"] = seg_scores
            # average precision by segment
            ap_scores = {}
            for tol, grp in inv_df.groupby("risk_tolerance"):
                idx = grp.index
                if y.loc[idx].nunique() > 1:
                    ap_scores[tol] = float(average_precision_score(y.loc[idx], prob[idx]))
            result["churn_ap"] = ap_scores
    loan_df = _loan_dataframe(session)
    if not loan_df.empty:
        loan_df = _ensure_horizon_labels(loan_df, "default_label", _heuristic_default_probability)
        X = loan_df[LOAN_FEATURES]
        y = loan_df["default_label_6m"]
        model = _build_loan_pipeline()
        if y.nunique() > 1:
            model.fit(X, y)
            prob = model.predict_proba(X)[:, 1]
            seg_scores = {}
            for sector, grp in loan_df.groupby("sector"):
                idx = grp.index
                if y.loc[idx].nunique() > 1:
                    seg_scores[sector] = float(roc_auc_score(y.loc[idx], prob[idx]))
            result["default"] = seg_scores
            ap_scores = {}
            for sector, grp in loan_df.groupby("sector"):
                idx = grp.index
                if y.loc[idx].nunique() > 1:
                    ap_scores[sector] = float(average_precision_score(y.loc[idx], prob[idx]))
            result["default_ap"] = ap_scores
    return result


def eda_bundle(session) -> Dict[str, Dict]:
    """Lightweight EDA package: hist bins, correlation matrices, outlier counts."""
    out: Dict[str, Dict] = {}
    inv_df = _investor_dataframe(session)
    if not inv_df.empty:
        out["investor"] = {
            "hist": {
                "engagement_score": np.histogram(inv_df["engagement_score"], bins=8, range=(0, 100)),
                "email_open_rate": np.histogram(inv_df["email_open_rate"], bins=8, range=(0, 1)),
                "age": np.histogram(inv_df["age"], bins=8),
            },
            "corr": inv_df[INVESTOR_NUMERIC].corr().fillna(0).round(3).to_dict(),
            "outliers": {
                "engagement_score": int((abs((inv_df["engagement_score"] - inv_df["engagement_score"].mean()) / (inv_df["engagement_score"].std() or 1e-6)) > 3).sum()),
                "aum": int((abs((inv_df["aum"] - inv_df["aum"].mean()) / (inv_df["aum"].std() or 1e-6)) > 3).sum()),
            },
        }
        # convert hist tuples to lists for JSON
        for k, (counts, bins) in out["investor"]["hist"].items():
            out["investor"]["hist"][k] = {"counts": counts.tolist(), "bins": bins.tolist()}

    loan_df = _loan_dataframe(session)
    if not loan_df.empty:
        out["loan"] = {
            "hist": {
                "ltv_ratio": np.histogram(loan_df["ltv_ratio"], bins=8, range=(0, 1)),
                "dscr": np.histogram(loan_df["dscr"], bins=8),
            },
            "corr": loan_df[LOAN_NUMERIC].corr().fillna(0).round(3).to_dict(),
            "outliers": {
                "ltv_ratio": int((abs((loan_df["ltv_ratio"] - loan_df["ltv_ratio"].mean()) / (loan_df["ltv_ratio"].std() or 1e-6)) > 3).sum()),
                "dscr": int((abs((loan_df["dscr"] - loan_df["dscr"].mean()) / (loan_df["dscr"].std() or 1e-6)) > 3).sum()),
            },
        }
        for k, (counts, bins) in out["loan"]["hist"].items():
            out["loan"]["hist"][k] = {"counts": counts.tolist(), "bins": bins.tolist()}
    return out


def robustness_summary(session) -> Dict[str, Dict]:
    """
    Compose a lightweight robustness view using existing bootstrap_ci and sensitivity.
    """
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception:
        return {}
    return {
        "bootstrap_ci": metrics.get("bootstrap_ci", {}),
        "sensitivity": metrics.get("sensitivity", {}),
        "imbalance": metrics.get("imbalance", {}),
    }


def _single_feature_auc(X: pd.DataFrame, y: pd.Series, feature: str, problem: str) -> Dict[str, float]:
    """Fit a tiny logistic model on a single feature for quick benchmark."""
    if y.nunique() <= 1:
        return {"roc_auc": 0.0}
    categorical = [feature] if X[feature].dtype == object or X[feature].dtype == bool else []
    numeric = [] if categorical else [feature]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, prob)) if len(set(y_test)) > 1 else 0.0
        return {"roc_auc": auc}
    except Exception:
        return {"roc_auc": 0.0}


def imputation_benchmarks(session) -> Dict[str, Dict]:
    """Compare simple imputation strategies for churn/default using a fast logistic model."""
    results = {"churn": {}, "default": {}}

    def _bench(df: pd.DataFrame, features: list, label_col: str, categorical: list, numeric: list):
        if df.empty or df[label_col].nunique() <= 1:
            return {}
        X = df[features].copy()
        y = df[label_col].astype(int)
        base_len = len(X)
        missing_rows = X.isna().any(axis=1).sum()
        strategies = {
            "drop": "drop",
            "mean": SimpleImputer(strategy="mean"),
            "median": SimpleImputer(strategy="median"),
            "knn": KNNImputer(n_neighbors=5),
        }
        out = {}
        for name, imputer in strategies.items():
            try:
                X_work = X
                retained = base_len
                if name == "drop":
                    X_work = X.dropna()
                    y_work = y.loc[X_work.index]
                    retained = len(X_work)
                else:
                    y_work = y
                preprocess = ColumnTransformer(
                    transformers=[
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", Pipeline(steps=[("imputer", imputer if name != "drop" else SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                    ]
                )
                clf = LogisticRegression(max_iter=200, class_weight="balanced")
                pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
                X_train, X_test, y_train, y_test = train_test_split(
                    X_work, y_work, test_size=0.2, random_state=42, stratify=y_work if y_work.nunique() > 1 else None
                )
                pipe.fit(X_train, y_train)
                prob = pipe.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, prob)) if len(set(y_test)) > 1 else 0.0
                ap = float(average_precision_score(y_test, prob)) if len(set(y_test)) > 1 else 0.0
                out[name] = {"roc_auc": auc, "avg_precision": ap, "retained_pct": retained / base_len if base_len else 0.0, "missing_rows": int(missing_rows)}
            except Exception:
                out[name] = {"roc_auc": 0.0, "avg_precision": 0.0, "retained_pct": 0.0, "missing_rows": int(missing_rows)}
        return out

    inv_df = _investor_dataframe(session)
    if not inv_df.empty:
        inv_df = _ensure_horizon_labels(inv_df, "churn_label", _heuristic_churn_probability)
        res = _bench(inv_df, INVESTOR_FEATURES, "churn_label_6m", INVESTOR_CATEGORICAL, INVESTOR_NUMERIC)
        results["churn"] = res

    loan_df = _loan_dataframe(session)
    if not loan_df.empty:
        loan_df = _ensure_horizon_labels(loan_df, "default_label", _heuristic_default_probability)
        res = _bench(loan_df, LOAN_FEATURES, "default_label_6m", LOAN_CATEGORICAL, LOAN_NUMERIC)
        results["default"] = res
    return results


def bootstrap_metric_ci(session, problem: str, horizon: str = "6m", n_boot: int = 50) -> Dict[str, Dict]:
    """
    Bootstrap AUC / average precision for a simple interpretable logistic baseline.
    Provides mean/std and percentile intervals.
    """
    rng = np.random.default_rng(42)
    scores = []
    if problem == "churn":
        df = _investor_dataframe(session)
        if df.empty:
            return {}
        df = _ensure_horizon_labels(df, "churn_label", _heuristic_churn_probability)
        y = df[f"churn_label_{horizon}"].astype(int)
        X = df[["engagement_score", "inactivity_days", "email_open_rate"]]
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=200, class_weight="balanced")),
            ]
        )
    else:
        df = _loan_dataframe(session)
        if df.empty:
            return {}
        df = _ensure_horizon_labels(df, "default_label", _heuristic_default_probability)
        y = df[f"default_label_{horizon}"].astype(int)
        X = df[["ltv_ratio", "dscr", "collateral_score"]]
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=200, class_weight="balanced")),
            ]
        )

    if y.nunique() <= 1:
        return {}

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    n = len(y_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X_arr[idx]
        yb = y_arr[idx]
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                Xb, yb, test_size=0.25, random_state=42, stratify=yb if len(np.unique(yb)) > 1 else None
            )
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.0
            ap = average_precision_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.0
            scores.append({"auc": auc, "ap": ap})
        except Exception:
            continue

    if not scores:
        return {}
    auc_vals = np.array([s["auc"] for s in scores])
    ap_vals = np.array([s["ap"] for s in scores])
    ci = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return {
        "auc": {
            "mean": float(np.mean(auc_vals)),
            "std": float(np.std(auc_vals)),
            "ci": ci(auc_vals),
        },
        "avg_precision": {
            "mean": float(np.mean(ap_vals)),
            "std": float(np.std(ap_vals)),
            "ci": ci(ap_vals),
        },
    }


def bootstrap_coefficients(session, problem: str, horizon: str = "6m", n_boot: int = 50) -> Dict[str, Dict]:
    """
    Bootstrap coefficients for a simple logistic baseline to surface stability (mean/std/percentiles).
    """
    rng = np.random.default_rng(42)
    if problem == "churn":
        df = _investor_dataframe(session)
        if df.empty:
            return {}
        df = _ensure_horizon_labels(df, "churn_label", _heuristic_churn_probability)
        y = df[f"churn_label_{horizon}"].astype(int)
        X = df[["engagement_score", "inactivity_days", "email_open_rate"]]
    else:
        df = _loan_dataframe(session)
        if df.empty:
            return {}
        df = _ensure_horizon_labels(df, "default_label", _heuristic_default_probability)
        y = df[f"default_label_{horizon}"].astype(int)
        X = df[["ltv_ratio", "dscr", "collateral_score"]]
    if y.nunique() <= 1:
        return {}
    X_arr = StandardScaler().fit_transform(X)
    y_arr = y.to_numpy()
    n = len(y_arr)
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X_arr[idx]
        yb = y_arr[idx]
        if len(np.unique(yb)) < 2:
            continue
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(Xb, yb)
        coefs.append(clf.coef_[0])
    if not coefs:
        return {}
    coef_mat = np.vstack(coefs)
    summary = {}
    for i, name in enumerate(X.columns):
        vals = coef_mat[:, i]
        summary[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p2_5": float(np.percentile(vals, 2.5)),
            "p97_5": float(np.percentile(vals, 97.5)),
        }
    return summary


def knn_neighbors(session, problem: str, features: Dict, horizon: str = "6m", k: int = 10) -> List[Dict]:
    """Return nearest neighbours (feature space) with their labels for a sanity check."""
    if problem == "churn":
        df = _investor_dataframe(session)
        if df.empty:
            return []
        df = _ensure_horizon_labels(df, "churn_label", _heuristic_churn_probability)
        labels = df[f"churn_label_{horizon}"]
        X = df[INVESTOR_FEATURES]
        neigh_pipe = _build_investor_knn_pipeline()
        neigh_pipe.fit(X, labels)
        distances, indices = neigh_pipe.named_steps["model"].kneighbors(neigh_pipe.named_steps["preprocess"].transform(pd.DataFrame([features])), n_neighbors=min(k, len(df)))
        out = []
        for dist, idx in zip(distances[0], indices[0]):
            row = df.iloc[idx]
            out.append(
                {
                    "id": int(row.get("id", idx)),
                    "label": int(labels.iloc[idx]),
                    "distance": float(dist),
                    "engagement_score": float(row.get("engagement_score", 0)),
                    "inactivity_days": float(row.get("inactivity_days", 0)),
                    "email_open_rate": float(row.get("email_open_rate", 0)),
                }
            )
        return out

    df = _loan_dataframe(session)
    if df.empty:
        return []
    df = _ensure_horizon_labels(df, "default_label", _heuristic_default_probability)
    labels = df[f"default_label_{horizon}"]
    X = df[LOAN_FEATURES]
    neigh_pipe = _build_loan_knn_pipeline()
    neigh_pipe.fit(X, labels)
    distances, indices = neigh_pipe.named_steps["model"].kneighbors(neigh_pipe.named_steps["preprocess"].transform(pd.DataFrame([features])), n_neighbors=min(k, len(df)))
    out = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        out.append(
            {
                "id": int(row.get("id", idx)),
                "label": int(labels.iloc[idx]),
                "distance": float(dist),
                "sector": row.get("sector", ""),
                "ltv_ratio": float(row.get("ltv_ratio", 0)),
                "dscr": float(row.get("dscr", 0)),
            }
        )
    return out


HYPERPARAM_SUMMARY = {
    "churn": {
        "ensemble": {"rf_n_estimators": 150, "gb_learning_rate": 0.1, "gb_estimators": 100},
        "rf": {"n_estimators": 120, "max_depth": None},
        "logreg": {"max_iter": 200, "class_weight": "balanced"},
        "adaboost": {"n_estimators": 140, "learning_rate": 0.4, "base_depth": 2},
        "knn": {"n_neighbors": 20, "weights": "distance"},
    },
    "default": {
        "ensemble": {"rf_n_estimators": 180, "gb_learning_rate": 0.1, "gb_estimators": 100},
        "rf": {"n_estimators": 140, "max_depth": None},
        "logreg": {"max_iter": 200, "class_weight": "balanced"},
        "adaboost": {"n_estimators": 150, "learning_rate": 0.35, "base_depth": 2},
        "knn": {"n_neighbors": 20, "weights": "distance"},
    },
}


def compute_sensitivity(session) -> Dict[str, Dict]:
    """Perturb key features and record bucket flip rates."""
    results: Dict[str, Dict] = {"churn": {}, "default": {}}

    # Churn sensitivity
    invs = _investor_dataframe(session)
    if not invs.empty:
        model = load_family_model("churn", "ensemble", session, "6m")
        base_probs = model.predict_proba(invs[INVESTOR_FEATURES])[:, 1]
        base_buckets = [risk_bucket(p) for p in base_probs]
        key_feats = ["engagement_score", "inactivity_days", "email_open_rate", "call_frequency", "aum"]
        for feat in key_feats:
            if feat not in invs.columns:
                continue
            std = invs[feat].std() or 0.0
            if std == 0:
                continue
            shift = 0.5 * std
            for direction, delta in [("up", shift), ("down", -shift)]:
                df_shift = invs[INVESTOR_FEATURES].copy()
                df_shift[feat] = df_shift[feat] + delta
                probs = model.predict_proba(df_shift)[:, 1]
                buckets = [risk_bucket(p) for p in probs]
                flips = sum(1 for b, b2 in zip(base_buckets, buckets) if b != b2)
                pct = flips / len(buckets) if len(buckets) else 0.0
                results["churn"].setdefault(feat, {})[f"{direction}_flip_pct"] = pct

    # Default sensitivity
    loans = _loan_dataframe(session)
    if not loans.empty:
        model = load_family_model("default", "ensemble", session, "6m")
        base_probs = model.predict_proba(loans[LOAN_FEATURES])[:, 1]
        base_buckets = [risk_bucket(p) for p in base_probs]
        key_feats = ["ltv_ratio", "dscr", "collateral_score", "amount"]
        for feat in key_feats:
            if feat not in loans.columns:
                continue
            std = loans[feat].std() or 0.0
            if std == 0:
                continue
            shift = 0.5 * std
            for direction, delta in [("up", shift), ("down", -shift)]:
                df_shift = loans[LOAN_FEATURES].copy()
                df_shift[feat] = df_shift[feat] + delta
                probs = model.predict_proba(df_shift)[:, 1]
                buckets = [risk_bucket(p) for p in probs]
                flips = sum(1 for b, b2 in zip(base_buckets, buckets) if b != b2)
                pct = flips / len(buckets) if len(buckets) else 0.0
                results["default"].setdefault(feat, {})[f"{direction}_flip_pct"] = pct

    return results


def feature_stats(session) -> Dict[str, Dict]:
    """Capture baseline feature means/stds to support drift monitoring."""
    stats: Dict[str, Dict] = {"investor": {}, "loan": {}}
    inv_df = _investor_dataframe(session)
    if not inv_df.empty:
        for col in INVESTOR_NUMERIC:
            if col not in inv_df.columns:
                continue
            series = inv_df[col].astype(float)
            stats["investor"][col] = {
                "mean": float(series.mean()),
                "std": float(series.std() or 1e-6),
            }
    loan_df = _loan_dataframe(session)
    if not loan_df.empty:
        for col in LOAN_NUMERIC:
            if col not in loan_df.columns:
                continue
            series = loan_df[col].astype(float)
            stats["loan"][col] = {
                "mean": float(series.mean()),
                "std": float(series.std() or 1e-6),
            }
    return stats


def investor_to_features(investor: Investor) -> Dict:
    return {
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
    }


def loan_to_features(loan: Loan) -> Dict:
    return {
        "amount": loan.amount,
        "ltv_ratio": loan.ltv_ratio,
        "term_months": loan.term_months,
        "sector": loan.sector,
        "arrears_flag": bool(loan.arrears_flag),
        "dscr": loan.dscr,
        "covenants_flag": bool(loan.covenants_flag),
        "collateral_score": loan.collateral_score,
    }

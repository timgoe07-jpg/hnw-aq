from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score


def cross_val_metrics(model, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> Dict[str, float]:
    """Compute cross-validated metrics to reduce train/test split variance."""
    if len(np.unique(y)) < 2:
        # Degenerate; avoid crashes and return zeros
        return {
            "cv_accuracy": 0.0,
            "cv_precision": 0.0,
            "cv_recall": 0.0,
            "cv_roc_auc": 0.0,
            "cv_avg_precision": 0.0,
        }
    cv = StratifiedKFold(n_splits=min(cv_splits, len(y)), shuffle=True, random_state=42)
    prec_scorer = make_scorer(precision_score, zero_division=0)
    rec_scorer = make_scorer(recall_score, zero_division=0)
    metrics = {
        "cv_accuracy": float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="accuracy"))),
        "cv_precision": float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=prec_scorer))),
        "cv_recall": float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=rec_scorer))),
        "cv_roc_auc": float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc"))),
        "cv_avg_precision": float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="average_precision"))),
    }
    return metrics

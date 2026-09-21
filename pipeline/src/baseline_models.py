"""
baseline_models.py
-------------------
Trains the three non-sequential baselines from the project brief on the
flattened window features, with class-imbalance handling (failures are
rare) and the metrics you'll need for Experiment 1 / 2 / 3 comparisons.

This intentionally stays a thin wrapper -- the goal for Week 4 is a working,
reproducible baseline harness to compare the eventual LSTM against, not
hyperparameter-tuned final models.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                              recall_score, roc_auc_score)

import config

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


@dataclass
class EvalResult:
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    n_test: int
    n_positive: int


def _class_weight_ratio(y_train: np.ndarray) -> float:
    pos = max(y_train.sum(), 1)
    neg = len(y_train) - pos
    return neg / pos


def get_models(y_train: np.ndarray) -> Dict[str, object]:
    ratio = _class_weight_ratio(y_train)
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_SEED
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=ratio, eval_metric="aucpr",
            n_jobs=-1, random_state=config.RANDOM_SEED
        )
    else:
        print("  [warn] xgboost not installed -- skipping. `pip install xgboost` to include it.")
    return models


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test) -> Dict[str, EvalResult]:
    results = {}
    models = get_models(y_train)
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        results[name] = EvalResult(
            model_name=name,
            precision=precision_score(y_test, preds, zero_division=0),
            recall=recall_score(y_test, preds, zero_division=0),
            f1=f1_score(y_test, preds, zero_division=0),
            roc_auc=roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan"),
            pr_auc=average_precision_score(y_test, proba) if len(set(y_test)) > 1 else float("nan"),
            n_test=len(y_test),
            n_positive=int(y_test.sum()),
        )
    return results


def results_to_dataframe(results: Dict[str, EvalResult]):
    import pandas as pd
    rows = [r.__dict__ for r in results.values()]
    return pd.DataFrame(rows).set_index("model_name")

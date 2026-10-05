from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import column_types

MODELS = ("logistic", "gbm")

RATING_EDGES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 1.0]
RATING_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def build(kind: str, X: pd.DataFrame, seed: int = 0, calibrate: bool = True) -> Pipeline:
    numeric, categorical = column_types(X)
    if kind == "logistic":
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical),
        ])
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    elif kind == "gbm":
        pre = ColumnTransformer([
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), categorical),
        ])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            class_weight="balanced", random_state=seed,
        )
    else:
        raise ValueError(f"unknown model {kind!r}, choose from {MODELS}")

    if calibrate:
        clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    return Pipeline([("prep", pre), ("clf", clf)])


def cross_val_auc(model: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int = 5, seed: int = 0) -> np.ndarray:
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc")


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def choose_threshold(y: np.ndarray, p: np.ndarray, cost_fn: float = 5.0, cost_fp: float = 1.0) -> float:
    """Threshold that minimises expected cost. A missed default (false negative)
    is assumed to cost `cost_fn` times a wrongly declined good loan."""
    best_t, best_cost = 0.5, np.inf
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_t, best_cost = float(t), cost
    return best_t


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "threshold": threshold,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "approval_rate": float((pred == 0).mean()),
        "default_rate_approved": float(y[pred == 0].mean()) if (pred == 0).any() else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(y)),
        "base_rate": float(y.mean()),
    }


def importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int = 0, repeats: int = 10) -> pd.DataFrame:
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=repeats, random_state=seed, n_jobs=-1)
    return (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def rating(p: np.ndarray | pd.Series) -> pd.Series:
    return pd.cut(np.asarray(p), bins=RATING_EDGES, labels=RATING_LABELS, include_lowest=True).astype(str)


def rating_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"rating": rating(p), "pd": p, "default": y})
    g = df.groupby("rating", observed=True).agg(
        loans=("default", "size"), avg_pd=("pd", "mean"), observed_default_rate=("default", "mean"),
    )
    g["share"] = g["loans"] / g["loans"].sum()
    return g.reindex([r for r in RATING_LABELS if r in g.index])


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"pd": p, "default": y})
    df["bin"] = pd.qcut(df["pd"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(loans=("default", "size"), predicted=("pd", "mean"), observed=("default", "mean"))
    g.index = g.index.astype(str)
    return g.reset_index()


def roc_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y, p)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.clip(thr, 0, 1)})

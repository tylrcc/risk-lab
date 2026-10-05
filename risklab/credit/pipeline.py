from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import plotting
from . import data, model, report


@dataclass
class Config:
    source: str = "german"
    path: Path = Path("data/german_credit.csv")
    models: list[str] = field(default_factory=lambda: list(model.MODELS))
    test_size: float = 0.25
    seed: int = 0
    cost_fn: float = 5.0
    sample: int | None = None
    out: Path = Path("reports/credit")


def run(cfg: Config) -> dict[str, dict]:
    plotting.setup()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    df = data.load(cfg.source, Path(cfg.path), cfg.sample, cfg.seed)
    X, y, ids = data.split_features(df)
    X_tr, X_te, y_tr, y_te, _, id_te = train_test_split(
        X, y, ids, test_size=cfg.test_size, stratify=y, random_state=cfg.seed,
    )

    fitted, scores, metrics, cv = {}, {}, {}, {}
    for kind in cfg.models:
        m = model.build(kind, X_tr, cfg.seed)
        cv[kind] = model.cross_val_auc(m, X_tr, y_tr, seed=cfg.seed)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        threshold = model.choose_threshold(y_tr.to_numpy(), m.predict_proba(X_tr)[:, 1], cfg.cost_fn)
        fitted[kind], scores[kind] = m, p
        metrics[kind] = model.evaluate(y_te.to_numpy(), p, threshold)

    best = max(metrics, key=lambda k: metrics[k]["auc"])
    p = scores[best]
    yt = y_te.to_numpy()
    threshold = metrics[best]["threshold"]
    imp = model.importance(fitted[best], X_te, y_te, cfg.seed)
    ratings = model.rating_table(yt, p)
    calib = model.calibration_table(yt, p)

    report.roc_chart({k: (yt, scores[k], metrics[k]["auc"]) for k in cfg.models}, out / "roc.png")
    report.ks_chart(yt, p, out / "ks.png")
    report.calibration_chart(calib, out / "calibration.png")
    report.distribution_chart(yt, p, threshold, out / "distribution.png")
    report.importance_chart(imp, out / "importance.png")
    report.rating_chart(ratings, out / "ratings.png")

    scored = X_te.copy()
    scored.insert(0, data.ID, id_te.to_numpy())
    scored["actual_default"] = yt
    for k in cfg.models:
        scored[f"pd_{k}"] = scores[k]
    scored["pd"] = p
    scored["rating"] = model.rating(p).to_numpy()
    scored["decision"] = np.where(p >= threshold, "decline", "approve")
    scored["model"] = best
    scored.to_csv(out / "scored_loans.csv", index=False)

    pd.DataFrame(metrics).T.to_csv(out / "metrics.csv", index_label="model")
    imp.to_csv(out / "feature_importance.csv", index=False)
    ratings.to_csv(out / "rating_buckets.csv", index_label="rating")
    calib.to_csv(out / "calibration.csv", index=False)
    model.roc_table(yt, p).to_csv(out / "roc_curve.csv", index=False)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump({"model": fitted[best], "kind": best, "threshold": threshold, "features": list(X.columns)}, f)
    (out / "run.json").write_text(json.dumps({**asdict(cfg), "path": str(cfg.path), "out": str(out), "best": best}, indent=2))
    report.write_summary(metrics, cv, ratings, imp, out / "summary.md", asdict(cfg))
    return metrics


def score(model_path: Path, loans: pd.DataFrame) -> pd.DataFrame:
    """Apply a saved model to new loans. Missing columns are filled with NaN."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    X = loans.reindex(columns=bundle["features"])
    p = bundle["model"].predict_proba(X)[:, 1]
    out = loans.copy()
    out["pd"] = p
    out["rating"] = model.rating(p).to_numpy()
    out["decision"] = np.where(p >= bundle["threshold"], "decline", "approve")
    return out

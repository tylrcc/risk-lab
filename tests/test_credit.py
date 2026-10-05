from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklab.credit import data, model, pipeline

ROOT = Path(__file__).resolve().parents[1]
GERMAN = ROOT / "data" / "german_credit.csv"


@pytest.fixture(scope="module")
def german():
    df = data.load("german", GERMAN)
    return data.split_features(df)


def test_german_dataset_shape(german):
    X, y, ids = german
    assert len(X) == 1000
    assert y.mean() == pytest.approx(0.30)
    assert ids.is_unique


@pytest.mark.parametrize("kind", model.MODELS)
def test_model_beats_chance_on_holdout(german, kind):
    X, y, _ = german
    X_tr, X_te, y_tr, y_te = X.iloc[:750], X.iloc[750:], y.iloc[:750], y.iloc[750:]
    m = model.build(kind, X_tr, calibrate=False)
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    assert model.evaluate(y_te.to_numpy(), p, 0.5)["auc"] > 0.70


def test_ks_statistic_bounds():
    y = np.array([0, 0, 1, 1])
    assert model.ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert model.ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_threshold_moves_down_when_misses_cost_more():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 2000), 0, 1)
    assert model.choose_threshold(y, p, cost_fn=10) < model.choose_threshold(y, p, cost_fn=1)


def test_rating_buckets_are_ordered():
    p = np.array([0.01, 0.07, 0.15, 0.25, 0.40, 0.50, 0.90])
    assert list(model.rating(p)) == ["A", "B", "C", "D", "E", "F", "G"]


def test_lending_club_adapter_parses_strings(tmp_path):
    raw = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "loan_amnt": [1000, 2000, 3000, 4000],
        "term": [" 36 months", " 60 months", " 36 months", " 60 months"],
        "int_rate": ["10.5%", "15.0%", "8.0%", "20.0%"],
        "grade": ["A", "C", "A", "E"],
        "emp_length": ["< 1 year", "10+ years", "3 years", "n/a"],
        "annual_inc": [50000, 60000, 70000, 30000],
        "revol_util": ["20%", "80%", None, "95%"],
        "loan_status": ["Fully Paid", "Charged Off", "Current", "Default"],
        "ignored_column": ["x", "y", "z", "w"],
    })
    path = tmp_path / "lc.csv"
    raw.to_csv(path, index=False)
    df = data.load_lending_club(path)
    assert len(df) == 3
    assert df["default"].tolist() == [0, 1, 1]
    assert df["term"].tolist() == [36.0, 60.0, 60.0]
    assert df["int_rate"].tolist() == [10.5, 15.0, 20.0]
    assert df["emp_length"].tolist()[:2] == [0.0, 10.0]
    assert np.isnan(df["emp_length"].iloc[2])
    assert "ignored_column" not in df


def test_pipeline_writes_outputs_and_scores_new_loans(tmp_path):
    cfg = pipeline.Config(path=GERMAN, models=["logistic"], out=tmp_path, sample=400)
    metrics = pipeline.run(cfg)
    assert metrics["logistic"]["auc"] > 0.6
    for name in ("scored_loans.csv", "metrics.csv", "feature_importance.csv", "rating_buckets.csv",
                 "summary.md", "model.pkl", "roc.png", "calibration.png"):
        assert (tmp_path / name).exists(), name

    new = pd.read_csv(GERMAN).drop(columns=["default"]).head(5)
    scored = pipeline.score(tmp_path / "model.pkl", new)
    assert scored["pd"].between(0, 1).all()
    assert set(scored["decision"]) <= {"approve", "decline"}

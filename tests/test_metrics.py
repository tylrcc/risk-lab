import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risklab.portfolio import metrics


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=5000)
    return pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx, name="r")


def test_parametric_var_matches_closed_form(normal_returns):
    r = normal_returns
    expected = -(r.mean() + stats.norm.ppf(0.05) * r.std())
    assert metrics.parametric_var(r, 0.95) == pytest.approx(expected)


def test_historical_var_close_to_parametric_for_normal_data(normal_returns):
    h = metrics.historical_var(normal_returns, 0.95)
    p = metrics.parametric_var(normal_returns, 0.95)
    assert abs(h - p) < 0.001


def test_cornish_fisher_matches_normal_var_for_normal_data(normal_returns):
    cf = metrics.cornish_fisher_var(normal_returns, 0.95)
    p = metrics.parametric_var(normal_returns, 0.95)
    assert cf == pytest.approx(p, rel=0.03)


def test_cornish_fisher_is_larger_for_fat_tails():
    idx = pd.bdate_range("2015-01-01", periods=5000)
    r = pd.Series(stats.t.rvs(df=3, scale=0.01, size=len(idx), random_state=0), index=idx)
    assert metrics.cornish_fisher_var(r, 0.99) > metrics.parametric_var(r, 0.99)


def test_cvar_is_at_least_var(normal_returns):
    assert metrics.cvar(normal_returns, 0.95) >= metrics.historical_var(normal_returns, 0.95)


def test_var_scales_with_sqrt_horizon(normal_returns):
    one = metrics.parametric_var(normal_returns, 0.95, horizon=1)
    ten = metrics.parametric_var(normal_returns, 0.95, horizon=10)
    assert ten == pytest.approx(one * np.sqrt(10))


def test_monte_carlo_var_close_to_parametric(normal_returns):
    df = pd.DataFrame({"a": normal_returns, "b": normal_returns.shift(1).bfill()})
    w = np.array([0.5, 0.5])
    port = df @ w
    mc = metrics.monte_carlo_var(df, w, 0.95, n_sims=200_000, seed=0)
    assert mc == pytest.approx(metrics.parametric_var(port, 0.95), rel=0.05)


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20, 0.10])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_beta_of_series_against_itself_is_one(normal_returns):
    assert metrics.beta(normal_returns, normal_returns) == pytest.approx(1.0)


def test_kupiec_accepts_well_calibrated_var(normal_returns):
    var = metrics.rolling_var(normal_returns, 252, 0.95).shift(1)
    test = metrics.kupiec_test(normal_returns, var, 0.95)
    assert test["p_value"] > 0.05
    assert abs(test["rate"] - 0.05) < 0.01


def test_kupiec_rejects_bad_var(normal_returns):
    var = pd.Series(0.002, index=normal_returns.index)
    assert metrics.kupiec_test(normal_returns, var, 0.95)["p_value"] < 0.01


def test_ewma_vol_tracks_regime_change():
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(0)
    r = pd.Series(np.concatenate([rng.normal(0, 0.005, 300), rng.normal(0, 0.03, 300)]), index=idx)
    v = metrics.ewma_vol(r)
    assert v.iloc[-1] > 3 * v.iloc[299]

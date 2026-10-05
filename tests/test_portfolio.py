import numpy as np
import pandas as pd
import pytest

from risklab.portfolio import backtest, optimize


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2018-01-01", periods=900)
    vols = np.array([0.010, 0.015, 0.020, 0.006])
    corr = np.array([
        [1.0, 0.6, 0.4, -0.2],
        [0.6, 1.0, 0.5, -0.1],
        [0.4, 0.5, 1.0, 0.0],
        [-0.2, -0.1, 0.0, 1.0],
    ])
    cov = np.outer(vols, vols) * corr
    data = rng.multivariate_normal([0.0005, 0.0006, 0.0008, 0.0002], cov, len(idx))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


@pytest.mark.parametrize("name", list(optimize.STRATEGIES))
def test_weights_are_long_only_and_sum_to_one(returns, name):
    w = optimize.STRATEGIES[name](returns)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_min_variance_has_lowest_variance(returns):
    cov = returns.cov().to_numpy()
    mv = optimize.min_variance(returns)
    for name, fn in optimize.STRATEGIES.items():
        w = fn(returns)
        assert mv @ cov @ mv <= w @ cov @ w + 1e-10, name


def test_risk_parity_equalises_risk_contributions(returns):
    w = optimize.risk_parity(returns)
    rc = optimize.risk_contribution(w, returns.cov().to_numpy())
    assert np.allclose(rc, 0.25, atol=1e-3)


def test_max_weight_is_respected(returns):
    for fn in (optimize.min_variance, optimize.max_sharpe, optimize.risk_parity):
        w = fn(returns, max_weight=0.3)
        assert w.max() <= 0.3 + 1e-6


def test_efficient_frontier_is_monotone(returns):
    ef = optimize.efficient_frontier(returns, n_points=15)
    assert len(ef) > 5
    assert (np.diff(ef["return"]) > 0).all()
    assert (np.diff(ef["vol"]) >= -1e-9).all()


def test_backtest_skips_lookback_and_rebalances_monthly(returns):
    res = backtest.run(returns, "equal_weight", lookback=252, freq="M", cost_bps=0)
    assert res.returns.index[0] == returns.index[252]
    months = returns.index[252:].to_period("M").nunique()
    assert abs(len(res.weights) - months) <= 1


def test_backtest_costs_reduce_returns(returns):
    free = backtest.run(returns, "max_sharpe", lookback=252, cost_bps=0)
    paid = backtest.run(returns, "max_sharpe", lookback=252, cost_bps=50)
    assert (1 + paid.returns).prod() < (1 + free.returns).prod()


def test_equal_weight_backtest_matches_direct_computation(returns):
    res = backtest.run(returns, "equal_weight", lookback=20, freq="M", cost_bps=0)
    first_day = res.returns.index[0]
    assert res.returns.loc[first_day] == pytest.approx(returns.loc[first_day].mean())


def test_fixed_weight_benchmark_resets_to_target(returns):
    b = backtest.fixed_weight(returns, {"A": 0.6, "D": 0.4}, "60/40", freq="M")
    assert len(b) == len(returns)
    assert b.iloc[0] == pytest.approx(0.6 * returns["A"].iloc[0] + 0.4 * returns["D"].iloc[0])


def test_compare_table_has_all_series(returns):
    results = [backtest.run(returns, s, lookback=100) for s in ("equal_weight", "min_variance")]
    bench = {"A": returns["A"], "B": returns["B"]}
    table = backtest.compare(results, bench)
    assert set(table.index) == {"equal_weight", "min_variance", "A", "B"}
    assert table.loc["A", "beta"] == pytest.approx(1.0)

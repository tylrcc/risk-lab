"""Walk forward backtest: weights are fit on a trailing window and held until
the next rebalance, so no future data leaks into the allocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics
from .optimize import STRATEGIES


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each period. freq is 'W', 'M', or 'Q'."""
    period = {"W": "W", "M": "M", "Q": "Q"}[freq.upper()]
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(period)).last().values)


def run(
    returns: pd.DataFrame,
    strategy: str | Callable,
    lookback: int = 252,
    freq: str = "M",
    cost_bps: float = 5.0,
    **kwargs,
) -> BacktestResult:
    fn = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    name = strategy if isinstance(strategy, str) else fn.__name__
    rebal = set(rebalance_dates(returns.index, freq))

    n = returns.shape[1]
    w = None
    port = []
    weight_rows = {}
    turnover_rows = {}
    for i, (date, r) in enumerate(returns.iterrows()):
        if w is None and i < lookback:
            port.append(np.nan)
            continue
        if w is None or (date in rebal and i >= lookback):
            window = returns.iloc[max(0, i - lookback):i]
            new_w = fn(window, **kwargs) if len(window) >= 20 else np.full(n, 1 / n)
            turn = float(np.abs(new_w - (w if w is not None else 0)).sum())
            cost = turn * cost_bps / 10_000
            w = new_w
            weight_rows[date] = w.copy()
            turnover_rows[date] = turn
        else:
            cost = 0.0
        rv = r.to_numpy()
        gross = float(w @ rv)
        port.append(gross - cost)
        grown = w * (1 + rv)
        w = grown / grown.sum()

    series = pd.Series(port, index=returns.index, name=name).dropna()
    weights = pd.DataFrame.from_dict(weight_rows, orient="index", columns=returns.columns)
    weights.index.name = "date"
    return BacktestResult(name, series, weights, pd.Series(turnover_rows, name="turnover"))


def fixed_weight(returns: pd.DataFrame, weights: dict[str, float], name: str, freq: str = "M") -> pd.Series:
    """Benchmark held at fixed target weights, reset at each period end."""
    cols = list(weights)
    target = np.array([weights[c] for c in cols])
    r = returns[cols]
    rebal = set(rebalance_dates(r.index, freq))
    w = target.copy()
    out = []
    for date, row in r.iterrows():
        rv = row.to_numpy()
        out.append(float(w @ rv))
        grown = w * (1 + rv)
        w = target.copy() if date in rebal else grown / grown.sum()
    return pd.Series(out, index=r.index, name=name)


def compare(
    results: list[BacktestResult],
    benchmarks: dict[str, pd.Series],
    rf: float = 0.0,
) -> pd.DataFrame:
    start = max(res.returns.index[0] for res in results)
    primary = next(iter(benchmarks.values())).loc[start:]
    rows = {}
    for res in results:
        rows[res.name] = metrics.summary(res.returns.loc[start:], primary, rf)
        rows[res.name]["avg_turnover"] = float(res.turnover.mean())
    for name, b in benchmarks.items():
        rows[name] = metrics.summary(b.loc[start:], primary, rf)
        rows[name]["avg_turnover"] = 0.0
    return pd.DataFrame(rows).T

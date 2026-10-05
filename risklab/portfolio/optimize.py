"""Long only portfolio construction. Every function returns weights summing to one."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import TRADING_DAYS


def _solve(objective, n: int, max_weight: float, x0=None) -> np.ndarray:
    x0 = np.full(n, 1 / n) if x0 is None else x0
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def equal_weight(returns: pd.DataFrame, **_) -> np.ndarray:
    n = returns.shape[1]
    return np.full(n, 1 / n)


def min_variance(returns: pd.DataFrame, max_weight: float = 1.0, **_) -> np.ndarray:
    cov = returns.cov().to_numpy()
    return _solve(lambda w: w @ cov @ w, len(cov), max_weight)


def max_sharpe(returns: pd.DataFrame, rf: float = 0.0, max_weight: float = 1.0, **_) -> np.ndarray:
    mu = returns.mean().to_numpy() * TRADING_DAYS
    cov = returns.cov().to_numpy() * TRADING_DAYS

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol if vol > 0 else 0.0

    return _solve(neg_sharpe, len(mu), max_weight)


def risk_parity(returns: pd.DataFrame, max_weight: float = 1.0, **_) -> np.ndarray:
    """Equal risk contribution from each asset.

    Solves the convex form min 0.5 w'Cw - sum(log w)/n, whose minimiser is
    proportional to the risk parity portfolio, then rescales to sum to one.
    """
    cov = returns.cov().to_numpy() * TRADING_DAYS
    n = len(cov)

    def objective(w):
        return 0.5 * w @ cov @ w - np.log(w).sum() / n

    def grad(w):
        return cov @ w - 1 / (n * w)

    res = minimize(objective, np.full(n, 1 / n), jac=grad, method="L-BFGS-B",
                   bounds=[(1e-8, None)] * n, options={"maxiter": 1000})
    w = res.x / res.x.sum()
    if w.max() <= max_weight:
        return w

    target = np.full(n, 1 / n)

    def squared_error(v):
        contrib = v * (cov @ v) / (v @ cov @ v)
        return float(((contrib - target) ** 2).sum())

    return _solve(squared_error, n, max_weight, x0=np.minimum(w, max_weight))


def risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_var = weights @ cov @ weights
    return weights * (cov @ weights) / port_var


def efficient_frontier(returns: pd.DataFrame, n_points: int = 40, max_weight: float = 1.0) -> pd.DataFrame:
    mu = returns.mean().to_numpy() * TRADING_DAYS
    cov = returns.cov().to_numpy() * TRADING_DAYS
    n = len(mu)
    lo = float(mu @ min_variance(returns, max_weight))
    hi = float(mu.max())
    rows = []
    for target in np.linspace(lo, hi, n_points):
        res = minimize(
            lambda w: w @ cov @ w,
            np.full(n, 1 / n),
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
            ],
        )
        if res.success:
            rows.append({"return": target, "vol": float(np.sqrt(res.fun))})
    return pd.DataFrame(rows)


STRATEGIES = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
    "risk_parity": risk_parity,
}

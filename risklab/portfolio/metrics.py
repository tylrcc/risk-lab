"""Risk and performance metrics on daily return series.

VaR and CVaR are reported as positive loss fractions: a one day 95% VaR
of 0.02 means a 2% loss is exceeded on 5% of days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import TRADING_DAYS


# volatility

def annual_vol(r: pd.Series | pd.DataFrame, periods: int = TRADING_DAYS):
    return r.std(ddof=1) * np.sqrt(periods)


def rolling_vol(r: pd.Series, window: int = 21, periods: int = TRADING_DAYS) -> pd.Series:
    return r.rolling(window).std(ddof=1) * np.sqrt(periods)


def ewma_vol(r: pd.Series, lam: float = 0.94, periods: int = TRADING_DAYS) -> pd.Series:
    """RiskMetrics style exponentially weighted volatility."""
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * periods)


# value at risk

def historical_var(r: pd.Series, alpha: float = 0.95, horizon: int = 1) -> float:
    q = np.quantile(r.dropna(), 1 - alpha)
    return float(max(-q, 0.0) * np.sqrt(horizon))


def parametric_var(r: pd.Series, alpha: float = 0.95, horizon: int = 1) -> float:
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return float(max(-(mu + z * sd), 0.0) * np.sqrt(horizon))


def cornish_fisher_var(r: pd.Series, alpha: float = 0.95, horizon: int = 1) -> float:
    """Parametric VaR with a skew and kurtosis adjusted quantile."""
    mu, sd = r.mean(), r.std(ddof=1)
    s = stats.skew(r.dropna())
    k = stats.kurtosis(r.dropna())
    z = stats.norm.ppf(1 - alpha)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return float(max(-(mu + z_cf * sd), 0.0) * np.sqrt(horizon))


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon: int = 1,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Simulate correlated normal asset returns and read off the portfolio quantile."""
    rng = np.random.default_rng(seed)
    mu = returns.mean().to_numpy() * horizon
    cov = returns.cov().to_numpy() * horizon
    sims = rng.multivariate_normal(mu, cov, size=n_sims)
    port = sims @ weights
    return float(max(-np.quantile(port, 1 - alpha), 0.0))


def cvar(r: pd.Series, alpha: float = 0.95) -> float:
    """Expected shortfall: mean loss on days beyond the historical VaR."""
    r = r.dropna()
    cutoff = np.quantile(r, 1 - alpha)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else 0.0


def rolling_var(r: pd.Series, window: int = 252, alpha: float = 0.95) -> pd.Series:
    return -r.rolling(window).quantile(1 - alpha)


def kupiec_test(r: pd.Series, var: pd.Series, alpha: float = 0.95) -> dict:
    """Kupiec proportion of failures test for a VaR series.

    Returns the exception count, the expected count, the observed rate, and the
    p-value of the likelihood ratio test. A small p-value means the model's
    exception frequency is inconsistent with the stated confidence level.
    """
    aligned = pd.concat([r, var], axis=1).dropna()
    if aligned.empty:
        return {"n": 0, "exceptions": 0, "expected": 0.0, "rate": np.nan, "p_value": np.nan}
    actual, v = aligned.iloc[:, 0], aligned.iloc[:, 1]
    hits = (actual < -v).astype(int)
    n, x = len(hits), int(hits.sum())
    p = 1 - alpha
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        phat = x / n
        lr = -2 * (
            x * np.log(p) + (n - x) * np.log(1 - p)
            - x * np.log(phat) - (n - x) * np.log(1 - phat)
        )
    return {
        "n": n,
        "exceptions": x,
        "expected": n * p,
        "rate": x / n,
        "p_value": float(1 - stats.chi2.cdf(lr, df=1)),
    }


# performance

def cumulative(r: pd.Series | pd.DataFrame):
    return (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    wealth = cumulative(r)
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def cagr(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    total = float((1 + r).prod())
    years = len(r) / periods
    return total ** (1 / years) - 1 if years > 0 else np.nan


def sharpe(r: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = r - rf / periods
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = r - rf / periods
    down = ex[ex < 0].std(ddof=1)
    return float(ex.mean() / down * np.sqrt(periods)) if down > 0 else np.nan


def beta(r: pd.Series, bench: pd.Series) -> float:
    a = pd.concat([r, bench], axis=1).dropna()
    if len(a) < 2:
        return np.nan
    c = np.cov(a.iloc[:, 0], a.iloc[:, 1], ddof=1)
    return float(c[0, 1] / c[1, 1])


def tracking_error(r: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    a = pd.concat([r, bench], axis=1).dropna()
    return float((a.iloc[:, 0] - a.iloc[:, 1]).std(ddof=1) * np.sqrt(periods))


def information_ratio(r: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    a = pd.concat([r, bench], axis=1).dropna()
    active = a.iloc[:, 0] - a.iloc[:, 1]
    sd = active.std(ddof=1)
    return float(active.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def summary(r: pd.Series, bench: pd.Series | None = None, rf: float = 0.0, alpha: float = 0.95) -> dict:
    out = {
        "cagr": cagr(r),
        "annual_vol": float(annual_vol(r)),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "max_drawdown": max_drawdown(r),
        "var_hist_95": historical_var(r, alpha),
        "var_param_95": parametric_var(r, alpha),
        "var_cf_95": cornish_fisher_var(r, alpha),
        "cvar_95": cvar(r, alpha),
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r)),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }
    if bench is not None:
        out.update({
            "beta": beta(r, bench),
            "tracking_error": tracking_error(r, bench),
            "information_ratio": information_ratio(r, bench),
        })
    return out

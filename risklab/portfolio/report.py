from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import plotting
from . import metrics
from .backtest import BacktestResult
from .optimize import efficient_frontier


def growth_chart(results: list[BacktestResult], benchmarks: dict[str, pd.Series], out: Path) -> Path:
    start = max(r.returns.index[0] for r in results)
    fig, ax = plt.subplots()
    for r in results:
        ax.plot(metrics.cumulative(r.returns.loc[start:]), label=r.name, lw=1.4)
    for (name, b), ls in zip(benchmarks.items(), ["--", ":"]):
        ax.plot(metrics.cumulative(b.loc[start:]), label=name, lw=1.2, ls=ls, color="#6b7280")
    ax.set_title("Growth of $1, net of costs")
    ax.legend(ncol=3)
    return plotting.save(fig, out)


def drawdown_chart(results: list[BacktestResult], benchmarks: dict[str, pd.Series], out: Path) -> Path:
    start = max(r.returns.index[0] for r in results)
    fig, ax = plt.subplots()
    for r in results:
        ax.plot(metrics.drawdown(r.returns.loc[start:]), label=r.name, lw=1.1)
    first = next(iter(benchmarks.items()))
    ax.plot(metrics.drawdown(first[1].loc[start:]), label=first[0], lw=1.0, ls="--", color="#6b7280")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_title("Drawdown")
    ax.legend(ncol=3)
    return plotting.save(fig, out)


def vol_chart(r: pd.Series, out: Path) -> Path:
    fig, ax = plt.subplots()
    ax.plot(metrics.rolling_vol(r, 21), label="21 day rolling", lw=1.0)
    ax.plot(metrics.rolling_vol(r, 63), label="63 day rolling", lw=1.0)
    ax.plot(metrics.ewma_vol(r), label="EWMA (0.94)", lw=1.0)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_title(f"Annualised volatility, {r.name}")
    ax.legend()
    return plotting.save(fig, out)


def var_chart(r: pd.Series, alpha: float, out: Path, window: int = 252) -> tuple[Path, dict]:
    var = metrics.rolling_var(r, window, alpha).shift(1)
    test = metrics.kupiec_test(r, var, alpha)
    aligned = pd.concat([r, var], axis=1).dropna()
    hits = aligned[aligned.iloc[:, 0] < -aligned.iloc[:, 1]]
    fig, ax = plt.subplots()
    ax.plot(aligned.iloc[:, 0], lw=0.5, color="#9ca3af", label="daily return")
    ax.plot(-aligned.iloc[:, 1], lw=1.1, color="#dc2626", label=f"{alpha:.0%} historical VaR")
    ax.scatter(hits.index, hits.iloc[:, 0], s=10, color="#dc2626", zorder=3, label="exceptions")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_title(
        f"VaR backtest, {r.name}: {test['exceptions']} exceptions vs {test['expected']:.0f} expected "
        f"(Kupiec p = {test['p_value']:.2f})"
    )
    ax.legend(ncol=3)
    return plotting.save(fig, out), test


def weights_chart(res: BacktestResult, out: Path) -> Path:
    fig, ax = plt.subplots()
    w = res.weights
    ax.stackplot(w.index, w.T.to_numpy(), labels=w.columns, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_title(f"Allocation over time, {res.name}")
    ax.legend(ncol=min(len(w.columns), 8), loc="upper center", bbox_to_anchor=(0.5, -0.08))
    return plotting.save(fig, out)


def frontier_chart(returns: pd.DataFrame, points: dict[str, np.ndarray], out: Path) -> Path:
    ef = efficient_frontier(returns)
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ef["vol"], ef["return"], color="#1f2937", lw=1.2, label="efficient frontier")
    for t in returns.columns:
        ax.scatter(np.sqrt(cov.loc[t, t]), mu[t], s=14, color="#9ca3af")
        ax.annotate(t, (np.sqrt(cov.loc[t, t]), mu[t]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    for i, (name, w) in enumerate(points.items()):
        vol = float(np.sqrt(w @ cov.to_numpy() @ w))
        ret = float(w @ mu.to_numpy())
        ax.scatter(vol, ret, s=42, zorder=3, label=name, color=plotting.PALETTE[(i + 1) % len(plotting.PALETTE)])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_xlabel("annual volatility")
    ax.set_ylabel("annual return")
    ax.set_title("Efficient frontier, full sample")
    ax.legend()
    return plotting.save(fig, out)


def correlation_chart(returns: pd.DataFrame, out: Path) -> Path:
    c = returns.corr()
    n = len(c)
    fig, ax = plt.subplots(figsize=(0.55 * n + 2, 0.55 * n + 1.5))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), c.columns)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c.iloc[i, j]) > 0.6 else "#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Return correlation")
    return plotting.save(fig, out)


def md_table(df: pd.DataFrame, index_name: str = "") -> str:
    head = [index_name] + [str(c) for c in df.columns]
    rows = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * len(df.columns)]
    for idx, row in df.iterrows():
        rows.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(rows)


def write_summary(table: pd.DataFrame, var_tests: dict[str, dict], out: Path, config: dict) -> Path:
    pct = ["cagr", "annual_vol", "max_drawdown", "var_hist_95", "var_param_95", "var_cf_95", "cvar_95",
           "tracking_error", "best_day", "worst_day", "avg_turnover"]
    cols = ["cagr", "annual_vol", "sharpe", "sortino", "max_drawdown", "var_hist_95", "cvar_95", "beta", "avg_turnover"]
    show = table[cols].copy()
    for c in cols:
        show[c] = show[c].map((lambda v: f"{v:.1%}") if c in pct else (lambda v: f"{v:.2f}"))
    show.columns = ["CAGR", "Vol", "Sharpe", "Sortino", "Max DD", "VaR 95", "CVaR 95", "Beta", "Turnover"]

    lines = ["# Portfolio risk report", ""]
    lines.append(
        f"Universe: {', '.join(config['tickers'])}. Sample: {config['start']} to {config['end']}. "
        f"Lookback {config['lookback']} days, rebalance {config['freq']}, costs {config['cost_bps']} bps per unit turnover."
    )
    lines += ["", md_table(show), ""]
    lines.append("## VaR model check (Kupiec POF, rolling 252 day historical VaR)")
    lines.append("")
    lines.append("| Strategy | Days | Exceptions | Expected | p-value |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, t in var_tests.items():
        lines.append(f"| {name} | {t['n']} | {t['exceptions']} | {t['expected']:.1f} | {t['p_value']:.3f} |")
    lines.append("")
    lines.append("A p-value below 0.05 means the VaR model is rejected at that level: it is producing too many or too few exceptions.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out

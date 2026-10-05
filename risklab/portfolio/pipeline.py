from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import plotting
from . import backtest, data, metrics, report
from .optimize import STRATEGIES

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "GLD", "TLT"]


@dataclass
class Config:
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    start: str = "2015-01-01"
    end: str | None = None
    benchmark: str = "SPY"
    bond: str = "AGG"
    strategies: list[str] = field(default_factory=lambda: list(STRATEGIES))
    lookback: int = 252
    freq: str = "M"
    cost_bps: float = 5.0
    alpha: float = 0.95
    rf: float = 0.0
    max_weight: float = 1.0
    out: Path = Path("reports/portfolio")
    cache: Path = Path("data/prices.csv")
    refresh: bool = False


def run(cfg: Config) -> pd.DataFrame:
    plotting.setup()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = list(dict.fromkeys(cfg.tickers + [cfg.benchmark, cfg.bond]))
    prices = data.load_prices(universe, cfg.start, cfg.end, cache=Path(cfg.cache), refresh=cfg.refresh)
    returns = data.to_returns(prices)
    assets = returns[cfg.tickers]
    cfg.end = cfg.end or str(returns.index[-1].date())

    results = [
        backtest.run(assets, s, lookback=cfg.lookback, freq=cfg.freq, cost_bps=cfg.cost_bps,
                     rf=cfg.rf, max_weight=cfg.max_weight)
        for s in cfg.strategies
    ]
    benchmarks = {
        cfg.benchmark: returns[cfg.benchmark].rename(cfg.benchmark),
        "60/40": backtest.fixed_weight(returns, {cfg.benchmark: 0.6, cfg.bond: 0.4}, "60/40", cfg.freq),
    }
    table = backtest.compare(results, benchmarks, cfg.rf)

    report.growth_chart(results, benchmarks, out / "growth.png")
    report.drawdown_chart(results, benchmarks, out / "drawdown.png")
    report.correlation_chart(assets, out / "correlation.png")
    full_sample = {s: STRATEGIES[s](assets, rf=cfg.rf, max_weight=cfg.max_weight) for s in cfg.strategies}
    report.frontier_chart(assets, full_sample, out / "frontier.png")

    var_tests = {}
    for res in results:
        _, var_tests[res.name] = report.var_chart(res.returns, cfg.alpha, out / f"var_{res.name}.png")
        report.weights_chart(res, out / f"weights_{res.name}.png")
    report.vol_chart(results[0].returns, out / "volatility.png")

    table.to_csv(out / "metrics.csv", index_label="series")
    _export_long(results, benchmarks, out)
    pd.DataFrame(var_tests).T.to_csv(out / "var_backtest.csv", index_label="strategy")
    pd.DataFrame(full_sample, index=assets.columns).to_csv(out / "weights_full_sample.csv", index_label="ticker")
    report.write_summary(table, var_tests, out / "summary.md", asdict(cfg))
    return table


def _export_long(results: list[backtest.BacktestResult], benchmarks: dict[str, pd.Series], out: Path) -> None:
    start = max(r.returns.index[0] for r in results)
    series = {r.name: r.returns for r in results} | {k: v.loc[start:] for k, v in benchmarks.items()}
    frames = []
    for name, s in series.items():
        s = s.loc[start:]
        frames.append(pd.DataFrame({
            "date": s.index, "series": name, "return": s.values,
            "cumulative": metrics.cumulative(s).values, "drawdown": metrics.drawdown(s).values,
            "rolling_vol_21d": metrics.rolling_vol(s, 21).values,
            "rolling_var_95": metrics.rolling_var(s, 252, 0.95).values,
        }))
    pd.concat(frames).to_csv(out / "returns.csv", index=False)

    w = []
    for r in results:
        melted = r.weights.reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
        melted.insert(1, "strategy", r.name)
        w.append(melted)
    pd.concat(w).to_csv(out / "weights.csv", index=False)

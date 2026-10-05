from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="risklab", description="Portfolio risk analytics and credit default prediction.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("portfolio", help="VaR, volatility, optimisation and benchmark backtest")
    p.add_argument("--tickers", nargs="+", help="asset universe (default: 8 large caps plus GLD and TLT)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--bond", default="AGG", help="bond ETF used for the 60/40 benchmark")
    p.add_argument("--strategies", nargs="+", choices=["equal_weight", "min_variance", "max_sharpe", "risk_parity"])
    p.add_argument("--lookback", type=int, default=252, help="estimation window in trading days")
    p.add_argument("--freq", default="M", choices=["W", "M", "Q"], help="rebalance frequency")
    p.add_argument("--cost-bps", type=float, default=5.0, help="one way cost per unit of turnover")
    p.add_argument("--alpha", type=float, default=0.95, help="VaR confidence level")
    p.add_argument("--rf", type=float, default=0.0, help="annual risk free rate")
    p.add_argument("--max-weight", type=float, default=1.0)
    p.add_argument("--out", default="reports/portfolio")
    p.add_argument("--cache", default="data/prices.csv")
    p.add_argument("--refresh", action="store_true", help="ignore the price cache")

    c = sub.add_parser("credit", help="train and evaluate default probability models")
    c.add_argument("--source", default="german", choices=["german", "lendingclub"])
    c.add_argument("--path", help="dataset path (default: data/german_credit.csv)")
    c.add_argument("--models", nargs="+", choices=["logistic", "gbm"])
    c.add_argument("--test-size", type=float, default=0.25)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--cost-fn", type=float, default=5.0, help="cost of a missed default relative to a declined good loan")
    c.add_argument("--sample", type=int, help="subsample rows (useful for large Lending Club files)")
    c.add_argument("--out", default="reports/credit")

    s = sub.add_parser("score", help="score new loans with a saved model")
    s.add_argument("loans", help="CSV with the same feature columns used in training")
    s.add_argument("--model", default="reports/credit/model.pkl")
    s.add_argument("--out", help="write scored CSV here instead of stdout")

    args = parser.parse_args(argv)

    if args.cmd == "portfolio":
        from .portfolio import pipeline

        cfg = pipeline.Config(
            tickers=[t.upper() for t in args.tickers] if args.tickers else pipeline.DEFAULT_TICKERS,
            start=args.start, end=args.end, benchmark=args.benchmark.upper(), bond=args.bond.upper(),
            strategies=args.strategies or pipeline.Config().strategies,
            lookback=args.lookback, freq=args.freq, cost_bps=args.cost_bps, alpha=args.alpha, rf=args.rf,
            max_weight=args.max_weight, out=Path(args.out), cache=Path(args.cache), refresh=args.refresh,
        )
        table = pipeline.run(cfg)
        cols = ["cagr", "annual_vol", "sharpe", "max_drawdown", "var_hist_95", "cvar_95"]
        print(table[cols].to_string(float_format=lambda v: f"{v:8.4f}"))
        print(f"\nreport written to {cfg.out}/summary.md")

    elif args.cmd == "credit":
        from .credit import pipeline

        path = Path(args.path) if args.path else Path("data/german_credit.csv")
        cfg = pipeline.Config(
            source=args.source, path=path, models=args.models or pipeline.Config().models,
            test_size=args.test_size, seed=args.seed, cost_fn=args.cost_fn, sample=args.sample, out=Path(args.out),
        )
        metrics = pipeline.run(cfg)
        table = pd.DataFrame(metrics).T[["auc", "gini", "ks", "brier", "precision", "recall", "approval_rate"]]
        print(table.to_string(float_format=lambda v: f"{v:8.4f}"))
        print(f"\nreport written to {cfg.out}/summary.md")

    elif args.cmd == "score":
        from .credit import pipeline

        scored = pipeline.score(Path(args.model), pd.read_csv(args.loans))
        if args.out:
            scored.to_csv(args.out, index=False)
            print(f"wrote {args.out}")
        else:
            scored.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

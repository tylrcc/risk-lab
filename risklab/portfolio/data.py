from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker.

    Reads from `cache` when it exists and covers every requested ticker,
    otherwise downloads from Yahoo Finance and writes the cache.
    """
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    if cache is not None and cache.exists() and not refresh:
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers) <= set(cached.columns):
            out = cached[tickers].loc[start:end]
            if len(out) > 0:
                return out.dropna(how="all")

    prices = download_prices(tickers, start, end)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    return prices


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no price data returned for {tickers}")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if close.shape[1] == 1 and len(tickers) == 1:
        close.columns = tickers
    close = close[tickers].sort_index()
    close.index.name = "date"
    return close.dropna(how="all").ffill().dropna()


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()

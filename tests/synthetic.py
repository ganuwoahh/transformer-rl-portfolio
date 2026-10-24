"""Synthetic market data with the same layout as `src.data.load_market_data()`.
Used for offline tests of the feature pipeline (no internet / yfinance needed)."""
import numpy as np
import pandas as pd

from src import config as cfg


def make_synthetic(seed: int = 0, n_days: int = 2800):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-10-01", periods=n_days, name="date")
    tickers = list(dict.fromkeys(cfg.ASSETS + [cfg.BENCHMARK]))

    rets = rng.normal(0.0003, rng.uniform(0.01, 0.03, len(tickers)), (n_days, len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    open_ = close.shift(1).bfill() * (1 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, close.shape)))
    volume = pd.DataFrame(rng.lognormal(15, 0.3, close.shape), index=dates, columns=tickers)

    prices = pd.concat({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, axis=1)
    prices.columns.names = ["field", "ticker"]

    t = np.arange(n_days)
    macro = pd.DataFrame(
        {
            "vix": np.clip(16 + 6 * np.abs(np.sin(t / 150)) + rng.normal(0, 1, n_days), 8, None),
            "tnx": np.clip(2.5 + np.cumsum(rng.normal(0, 0.02, n_days)), 0.3, None),
            "dxy": 95 * np.exp(np.cumsum(rng.normal(0, 0.003, n_days))),
            "rf": np.clip(1.0 + 3.0 * t / n_days + rng.normal(0, 0.05, n_days), 0.0, None),
        },
        index=dates,
    )
    return prices, macro

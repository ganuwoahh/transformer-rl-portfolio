"""Download, cache and align market data from yfinance.

Output of `load_market_data()`:
  prices: DataFrame, columns = MultiIndex (field, ticker), fields = open/high/low/close/volume
          (split/dividend adjusted), rows = SPY trading days.
  macro : DataFrame, columns = vix, tnx, dxy, rf (all closes), same rows.

Everything is aligned to the SPY trading calendar. BTC-USD trades 7 days a week, so
weekend moves get folded into the following Monday's return.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import config as cfg

FIELDS = ["open", "high", "low", "close", "volume"]


def download_ohlcv(tickers, start, end=None, retries: int = 3) -> pd.DataFrame:
    """Download daily OHLCV for `tickers`. Returns columns MultiIndex (field, ticker)."""
    import yfinance as yf  # imported lazily so the rest of the package works without it

    tickers = list(dict.fromkeys(tickers))
    raw = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers, start=start, end=end, auto_adjust=True,
                group_by="column", progress=False, threads=True,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # rate limits happen on shared Colab IPs
            print(f"  attempt {attempt}/{retries} failed: {exc}")
        time.sleep(5 * attempt)
    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance returned no data (rate limit or network issue). "
            "Wait a few minutes and retry, or run `pip install -U yfinance`."
        )

    if not isinstance(raw.columns, pd.MultiIndex):  # single-ticker fallback
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])
    raw.columns = pd.MultiIndex.from_arrays(
        [raw.columns.get_level_values(0).str.lower(), raw.columns.get_level_values(1)],
        names=["field", "ticker"],
    )
    raw = raw.loc[:, raw.columns.get_level_values("field").isin(FIELDS)]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw = raw.sort_index()

    close = raw["close"]
    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise RuntimeError(f"No data returned for: {missing}")
    return raw


def align_to_calendar(df: pd.DataFrame, calendar: pd.DatetimeIndex, max_ffill: int = 5) -> pd.DataFrame:
    """Reindex to `calendar`, forward-filling short gaps (holidays on other exchanges)."""
    return df.reindex(calendar).ffill(limit=max_ffill)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{a}|{b}" for a, b in df.columns]
    return out


def _unflatten(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = pd.MultiIndex.from_tuples(
        [tuple(c.split("|", 1)) for c in df.columns], names=["field", "ticker"]
    )
    return out


def load_market_data(refresh: bool = False, start=cfg.DATA_START, end=cfg.DATA_END):
    """Return (prices, macro). Uses the CSV cache in DATA_DIR/raw unless refresh=True."""
    raw_dir = cfg.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    p_prices, p_macro = raw_dir / "prices.csv", raw_dir / "macro.csv"

    if not refresh and p_prices.exists() and p_macro.exists():
        prices = _unflatten(pd.read_csv(p_prices, index_col=0, parse_dates=True))
        macro = pd.read_csv(p_macro, index_col=0, parse_dates=True)
        return prices, macro

    traded = list(dict.fromkeys(cfg.ASSETS + [cfg.BENCHMARK, cfg.CALENDAR_TICKER]))
    print(f"Downloading {len(traded)} tickers...")
    prices = download_ohlcv(traded, start, end)
    calendar = prices["close"][cfg.CALENDAR_TICKER].dropna().index
    prices = align_to_calendar(prices, calendar)

    print("Downloading macro series...")
    macro_tickers = list(cfg.MACRO_TICKERS.values()) + [cfg.RF_TICKER]
    macro = download_ohlcv(macro_tickers, start, end)["close"]
    macro = align_to_calendar(macro, calendar)
    rename = {v: k for k, v in cfg.MACRO_TICKERS.items()}
    rename[cfg.RF_TICKER] = "rf"
    macro = macro.rename(columns=rename)[["vix", "tnx", "dxy", "rf"]]

    # Trim to the first date on which every traded asset has a price, then require no gaps.
    close = prices["close"]
    all_ok = close.notna().all(axis=1)
    if not all_ok.any():
        raise ValueError("No date on which all assets have prices.")
    first_ok = all_ok.idxmax()
    prices, macro = prices.loc[first_ok:], macro.loc[first_ok:]
    if prices["close"].isna().any().any():
        bad = prices["close"].columns[prices["close"].isna().any()].tolist()
        raise ValueError(f"Gaps in close prices after alignment: {bad}")
    if macro.isna().any().any():
        bad = macro.columns[macro.isna().any()].tolist()
        raise ValueError(f"Gaps in macro series after alignment: {bad}")

    _flatten(prices).to_csv(p_prices)
    macro.to_csv(p_macro)
    print(f"Cached to {raw_dir}")
    return prices, macro


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and cache market data.")
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = ap.parse_args()
    prices, macro = load_market_data(refresh=args.refresh)
    print(f"Prices: {prices.shape}, {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Macro:  {macro.shape}")
    print(prices["close"].tail(3).round(2).to_string())

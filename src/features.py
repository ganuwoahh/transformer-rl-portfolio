"""Feature engineering + causal normalisation.

Timing convention (important for avoiding look-ahead):
  * Row t of every feature array uses data up to and including the close of day t only.
  * `returns[t]` is the simple return realised from close t-1 to close t.
  * The environment observes row t, chooses weights at close t, and earns `returns[t+1]`.
  * `rf[t]` is the daily cash return earned over the same interval as `returns[t]`; it uses
    the T-bill yield observed at close t-1, i.e. known when the position was opened.

Normalisation: each feature is z-scored with the mean/std of the *previous* 252 trading days
(`shift(1)`), so a value never influences its own statistics and nothing from the future is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------- normalisation
def causal_zscore(df: pd.DataFrame, window: int = cfg.ZSCORE_WINDOW, clip: float = cfg.ZSCORE_CLIP) -> pd.DataFrame:
    roll = df.rolling(window, min_periods=window)
    mu = roll.mean().shift(1)
    sd = roll.std().shift(1)
    return ((df - mu) / sd.where(sd > 0)).clip(-clip, clip)


# --------------------------------------------------------------------------- raw features
def _rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100.0 * avg_gain / (avg_gain + avg_loss)  # algebraically equal to 100 - 100/(1+RS)


def compute_asset_features(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-asset raw (un-normalised) features. Each value is a (T, N) DataFrame."""
    log_ret = np.log(close).diff()
    feats: dict[str, pd.DataFrame] = {"log_ret": log_ret}

    for w in cfg.VOL_WINDOWS:  # annualised rolling volatility
        feats[f"vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(cfg.TRADING_DAYS)

    # MACD, scaled by price so it is comparable across assets
    fast = close.ewm(span=cfg.MACD_FAST, adjust=False, min_periods=cfg.MACD_FAST).mean()
    slow = close.ewm(span=cfg.MACD_SLOW, adjust=False, min_periods=cfg.MACD_SLOW).mean()
    macd = (fast - slow) / close
    signal = macd.ewm(span=cfg.MACD_SIGNAL, adjust=False, min_periods=cfg.MACD_SIGNAL).mean()
    feats["macd"] = macd
    feats["macd_hist"] = macd - signal

    feats["rsi"] = (_rsi(close, cfg.RSI_WINDOW) - 50.0) / 50.0

    # Sharpe momentum: annualised rolling mean/std of daily log returns
    w = cfg.SHARPE_MOM_WINDOW
    mean, std = log_ret.rolling(w).mean(), log_ret.rolling(w).std()
    feats["sharpe_mom"] = mean / std.where(std > 0) * np.sqrt(cfg.TRADING_DAYS)

    # Volume relative to its recent average (log ratio)
    vol = volume.where(volume > 0).ffill()
    feats["vol_ratio"] = np.log(vol / vol.rolling(cfg.VOLUME_WINDOW).mean())
    return feats


def compute_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    vix, dxy, tnx = np.log(macro["vix"]), np.log(macro["dxy"]), macro["tnx"]
    return pd.DataFrame(
        {
            "vix_log": vix, "tnx": tnx, "dxy_log": dxy,
            "vix_chg5": vix.diff(5), "tnx_chg5": tnx.diff(5), "dxy_chg5": dxy.diff(5),
        },
        index=macro.index,
    )


# --------------------------------------------------------------------------- container
@dataclass(frozen=True)
class MarketData:
    dates: pd.DatetimeIndex
    assets: list
    asset_feature_names: list
    macro_feature_names: list
    asset_features: np.ndarray   # (T, N, F) float32, z-scored
    macro_features: np.ndarray   # (T, M)    float32, z-scored
    returns: np.ndarray          # (T, N)    simple returns, close[t-1] -> close[t]
    rf: np.ndarray               # (T,)      daily cash return over the same interval
    benchmark_returns: np.ndarray  # (T,)    SPY simple returns

    @property
    def n_steps(self) -> int:
        return len(self.dates)

    def split_bounds(self, train_end=cfg.TRAIN_END, val_end=cfg.VAL_END) -> dict:
        """Half-open index ranges [start, end) for train / val / test.

        Environments take these bounds rather than sliced copies, so a validation or test
        episode can still look back into earlier rows for its lookback window (that data is
        in the past, so it is legitimate).
        """
        i1 = int(self.dates.searchsorted(pd.Timestamp(train_end), side="right"))
        i2 = int(self.dates.searchsorted(pd.Timestamp(val_end), side="right"))
        return {"train": (0, i1), "val": (i1, i2), "test": (i2, self.n_steps)}

    def summary(self) -> str:
        lines = [
            f"Dates   : {self.dates[0].date()} -> {self.dates[-1].date()}  (T={self.n_steps})",
            f"Assets  : {len(self.assets)}  {self.assets}",
            f"Features: {len(self.asset_feature_names)} per asset {self.asset_feature_names}",
            f"Macro   : {len(self.macro_feature_names)} {self.macro_feature_names}",
            f"Array shapes: asset_features={self.asset_features.shape}, macro_features={self.macro_features.shape}",
        ]
        for name, (a, b) in self.split_bounds().items():
            if b > a:
                lines.append(f"  {name:5s}: {b - a:5d} days  {self.dates[a].date()} -> {self.dates[b - 1].date()}")
            else:
                lines.append(f"  {name:5s}: empty")
        return "\n".join(lines)


# --------------------------------------------------------------------------- pipeline
def build_market_data(prices: pd.DataFrame, macro: pd.DataFrame, assets=None, benchmark: str = cfg.BENCHMARK) -> MarketData:
    assets = list(assets or cfg.ASSETS)
    close, volume = prices["close"][assets], prices["volume"][assets]

    raw = compute_asset_features(close, volume)
    names = list(raw)
    z = {k: causal_zscore(v) for k, v in raw.items()}
    asset_feats = np.stack([z[k].to_numpy(dtype=float) for k in names], axis=-1)  # (T, N, F)

    macro = macro.loc[close.index]
    macro_raw = compute_macro_features(macro)
    macro_z = causal_zscore(macro_raw)
    macro_feats = macro_z.to_numpy(dtype=float)

    returns = close.pct_change().to_numpy(dtype=float)
    bench = prices["close"][benchmark].pct_change().to_numpy(dtype=float)
    rf = (macro["rf"] / 100.0 / cfg.TRADING_DAYS).shift(1).to_numpy(dtype=float)

    ok = (
        np.isfinite(asset_feats).all(axis=(1, 2))
        & np.isfinite(macro_feats).all(axis=1)
        & np.isfinite(returns).all(axis=1)
        & np.isfinite(bench)
        & np.isfinite(rf)
    )
    if not ok.any():
        raise ValueError("No fully valid rows: not enough history for the feature warm-up.")
    start = int(np.argmax(ok))
    if not ok[start:].all():
        bad_rows = close.index[start:][~ok[start:]]
        raise ValueError(f"Non-finite values after warm-up on {len(bad_rows)} dates, e.g. {bad_rows[:3].tolist()}")

    s = slice(start, None)
    return MarketData(
        dates=close.index[s],
        assets=assets,
        asset_feature_names=names,
        macro_feature_names=list(macro_raw.columns),
        asset_features=asset_feats[s].astype(np.float32),
        macro_features=macro_feats[s].astype(np.float32),
        returns=returns[s],
        rf=rf[s],
        benchmark_returns=bench[s],
    )


if __name__ == "__main__":
    from .data import load_market_data

    prices, macro = load_market_data()
    md = build_market_data(prices, macro)
    print(md.summary())

"""Performance metrics. (Plots get added here later.)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .portfolio import BacktestResult

COLUMNS = {
    "cum_return": "Cum. return", "ann_return": "Ann. return", "ann_vol": "Ann. vol",
    "sharpe": "Sharpe", "max_drawdown": "Max DD", "calmar": "Calmar",
    "cvar_95": "CVaR 95% (1d)", "turnover": "Turnover (x/yr)",
}


def compute_metrics(res: BacktestResult, periods: int = cfg.TRADING_DAYS) -> dict:
    r = np.asarray(res.net_returns, dtype=float)
    excess = r - res.rf
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    mdd = float((equity / peak - 1.0).min())
    ann_ret = float(equity[-1] ** (periods / len(r)) - 1.0)
    sd = excess.std(ddof=1)
    var = np.quantile(r, 0.05)
    return {
        "cum_return": float(equity[-1] - 1.0),
        "ann_return": ann_ret,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(periods)),
        "sharpe": float(excess.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "cvar_95": float(r[r <= var].mean()),  # mean of the worst 5% of days (negative = loss)
        "turnover": float(np.mean(res.traded) * periods),  # traded notional per year, in multiples of NAV
    }


def metrics_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame({name: compute_metrics(r) for name, r in results.items()}).T.rename(columns=COLUMNS)


def pretty(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy().astype(object)
    for c in ["Cum. return", "Ann. return", "Ann. vol", "Max DD"]:
        out[c] = table[c].map(lambda x: f"{x * 100:.1f}%")
    out["CVaR 95% (1d)"] = table["CVaR 95% (1d)"].map(lambda x: f"{x * 100:.2f}%")
    out["Sharpe"] = table["Sharpe"].map(lambda x: f"{x:.2f}")
    out["Calmar"] = table["Calmar"].map(lambda x: f"{x:.2f}")
    out["Turnover (x/yr)"] = table["Turnover (x/yr)"].map(lambda x: f"{x:.1f}x")
    return out

# Rule baselines with the same backtest as the agent (no params)

from __future__ import annotations

import numpy as np

from . import config as cfg
from .portfolio import BacktestResult, backtest


def equal_weight(data, every: int = cfg.REBALANCE_EVERY):
    # 1/N in every risky assest every 'every' days excluding cash
    n = len(data.assets)
    target = np.append(np.full(n, 1.0 / n), 0.0)

    def weights_fn(t, w_drift, k):
        return target if k % every == 0 else w_drift

    return weights_fn


def inverse_vol(data, window: int = 30, every: int = cfg.REBALANCE_EVERY):
    # weights are 1/volatility every 'every' days
    n = len(data.assets)
    fallback = np.append(np.full(n, 1.0 / n), 0.0)

    def weights_fn(t, w_drift, k):
        if k % every != 0:
            return w_drift
        past = data.returns[max(0, t - window + 1) : t + 1]
        vol = past.std(axis=0, ddof=1) if len(past) > 1 else np.zeros(n)
        if not np.all(vol > 0):
            return fallback
        inv = 1.0 / vol
        return np.append(inv / inv.sum(), 0.0)

    return weights_fn


def spy_buy_and_hold(data, start: int, end: int) -> BacktestResult:
    #raw index benchmark
    r = data.benchmark_returns[start:end].copy()
    return BacktestResult(
        dates=data.dates[start:end],
        net_returns=r,
        traded=np.zeros(len(r)),
        weights=np.full((len(r), len(data.assets) + 1), np.nan),
        rf=data.rf[start:end].copy(),
    )


def run_baselines(data, start: int, end: int, cost_bps: float = cfg.COST_BPS) -> dict:
    return {
        "Equal-weight (monthly)": backtest(data, start, end, equal_weight(data), cost_bps),
        "Inverse-vol (monthly)": backtest(data, start, end, inverse_vol(data), cost_bps),
        "SPY buy & hold": spy_buy_and_hold(data, start, end),
    }

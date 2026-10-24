"""Non-learning baselines, run through the same cost-aware backtester as the RL agent.

All three trade the same universe as the agent (not a broad market), so they are fair yardsticks.
None has fitted parameters, so evaluating them on any split cannot leak information.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg
from .portfolio import BacktestResult, backtest


def equal_weight(data, every: int = cfg.REBALANCE_EVERY):
    """1/N in every risky asset, rebalanced every `every` days (no cash)."""
    n = len(data.assets)
    target = np.append(np.full(n, 1.0 / n), 0.0)

    def weights_fn(t, w_drift, k):
        return target if k % every == 0 else w_drift  # holding w_drift = no trading

    return weights_fn


def inverse_vol(data, window: int = 30, every: int = cfg.REBALANCE_EVERY):
    """Weights proportional to 1 / trailing volatility, rebalanced every `every` days."""
    n = len(data.assets)
    fallback = np.append(np.full(n, 1.0 / n), 0.0)

    def weights_fn(t, w_drift, k):
        if k % every != 0:
            return w_drift
        past = data.returns[max(0, t - window + 1) : t + 1]  # days <= t only
        vol = past.std(axis=0, ddof=1) if len(past) > 1 else np.zeros(n)
        if not np.all(vol > 0):
            return fallback
        inv = 1.0 / vol
        return np.append(inv / inv.sum(), 0.0)

    return weights_fn


def spy_buy_and_hold(data, start: int, end: int) -> BacktestResult:
    """SPY benchmark: raw index returns, no costs (it is a benchmark, not a strategy)."""
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

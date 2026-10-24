"""Portfolio accounting shared by the RL environment and the baselines.

One implementation for both guarantees every strategy is charged costs identically.

Conventions
-----------
* Weight vectors have N+1 entries: N risky assets, then cash (last entry).
* "Return day" t is the interval close[t-1] -> close[t]. A decision made at the close of
  day t earns the return of day t+1.
* Costs: `cost_rate` per unit of risky-asset notional traded (buys and sells both count),
  deducted from portfolio value at the moment of rebalancing. Cash itself is free to trade.
* Every backtest / episode starts fully in cash, so buying the initial positions costs money.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg


def action_to_weights(action, logit_scale: float = cfg.LOGIT_SCALE) -> np.ndarray:
    """Map an action in [-1, 1]^(N+1) to long-only weights summing to 1 (softmax)."""
    z = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * logit_scale
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def portfolio_step(w_drift, w_target, asset_ret, rf, cost_rate):
    """Rebalance from `w_drift` to `w_target` at close t, then hold through close t+1.

    Returns (net_growth, w_next, traded, cost):
      net_growth : portfolio value multiple over the period, after costs
      w_next     : weights after price drift (the starting weights of the next period)
      traded     : risky-asset notional traded, as a fraction of portfolio value
      cost       : fraction of portfolio value paid in costs
    """
    w_drift = np.asarray(w_drift, dtype=np.float64)
    w_target = np.asarray(w_target, dtype=np.float64)
    asset_ret = np.asarray(asset_ret, dtype=np.float64)

    traded = float(np.abs(w_target[:-1] - w_drift[:-1]).sum())
    cost = cost_rate * traded
    risky = w_target[:-1] * (1.0 + asset_ret)
    cash = w_target[-1] * (1.0 + rf)
    gross = float(risky.sum() + cash)
    w_next = np.append(risky, cash) / gross
    return (1.0 - cost) * gross, w_next, traded, cost


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    net_returns: np.ndarray   # (K,) daily returns after costs
    traded: np.ndarray        # (K,) risky notional traded each day / portfolio value
    weights: np.ndarray       # (K, N+1) target weights chosen each day (last column = cash)
    rf: np.ndarray            # (K,) daily risk-free return

    @property
    def equity(self) -> np.ndarray:
        return np.cumprod(1.0 + self.net_returns)


def backtest(data, start: int, end: int, weights_fn, cost_bps: float = cfg.COST_BPS) -> BacktestResult:
    """Run a weights rule over return days [start, end).

    `weights_fn(t, w_drift, k)` is called at the close of day t = start-1+k (k = 0, 1, ...)
    and returns target weights. It may only use data up to and including day t.
    """
    if start < 1 or end > data.n_steps or end <= start:
        raise ValueError(f"bad range [{start}, {end}) for {data.n_steps} rows")
    n = len(data.assets)
    cost_rate = cost_bps / 1e4
    w = np.zeros(n + 1)
    w[-1] = 1.0
    K = end - start
    net = np.empty(K)
    traded = np.empty(K)
    weights = np.empty((K, n + 1))
    for k in range(K):
        t = start - 1 + k
        target = np.asarray(weights_fn(t, w, k), dtype=np.float64)
        if abs(target.sum() - 1.0) > 1e-6 or (target < -1e-12).any():
            raise ValueError(f"invalid target weights at step {k}: {target}")
        growth, w, tr, _ = portfolio_step(w, target, data.returns[t + 1], data.rf[t + 1], cost_rate)
        net[k], traded[k], weights[k] = growth - 1.0, tr, target
    return BacktestResult(data.dates[start:end], net, traded, weights, data.rf[start:end].copy())

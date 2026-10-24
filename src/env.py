"""Gymnasium environment for daily portfolio allocation with transaction costs.

Timing (no look-ahead)
  * At step k the agent sees feature rows up to and including the close of day t.
  * It chooses target weights at that close, pays costs to get there, and earns the
    return of day t+1.
  * Episodes start fully in cash.

Observation (Dict)
  "seq"     : (lookback, D) z-scored features, oldest row first. Column index of an asset
              feature is asset_idx * F + feature_idx; the macro features are the last M columns.
  "weights" : (N+1,) current weights after price drift (risky assets..., cash)
Action     : Box[-1, 1]^(N+1), turned into long-only weights by softmax (see portfolio.py)
Reward     : reward_scale * log(net portfolio growth over the day), net of costs
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as cfg
from .features import MarketData
from .portfolio import action_to_weights, portfolio_step


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: MarketData,
        start: int,
        end: int,
        lookback: int = cfg.LOOKBACK,
        episode_len=cfg.EPISODE_LEN,
        cost_bps: float = cfg.COST_BPS,
        logit_scale: float = cfg.LOGIT_SCALE,
        reward_scale: float = cfg.REWARD_SCALE,
    ):
        """`start`/`end` delimit the return days [start, end) this env may earn.

        episode_len=None runs one deterministic pass over the whole range (use for validation
        and test). An integer gives random-start episodes of that length (use for training).
        """
        super().__init__()
        span = end - start
        if start < lookback:
            raise ValueError(f"start={start} < lookback={lookback}: not enough history before the first decision")
        if end > data.n_steps or span <= 0:
            raise ValueError(f"bad range [{start}, {end}) for {data.n_steps} rows")
        if episode_len is not None and episode_len > span:
            raise ValueError(f"episode_len={episode_len} is longer than the range ({span} days)")

        self.data, self.start, self.end, self.lookback = data, start, end, lookback
        self.span = span
        self.ep_len = span if episode_len is None else int(episode_len)
        self.cost_rate = cost_bps / 1e4
        self.logit_scale, self.reward_scale = logit_scale, reward_scale

        T, n, F = data.asset_features.shape
        self.n_assets = n
        self._X = np.concatenate([data.asset_features.reshape(T, n * F), data.macro_features], axis=1).astype(np.float32)
        self.obs_dim = self._X.shape[1]

        clip = cfg.ZSCORE_CLIP
        self.observation_space = spaces.Dict(
            {
                "seq": spaces.Box(-clip, clip, shape=(lookback, self.obs_dim), dtype=np.float32),
                "weights": spaces.Box(0.0, 1.0, shape=(n + 1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n + 1,), dtype=np.float32)
        self._first, self._k = start, 0
        self._w = self._cash_weights()

    @classmethod
    def for_split(cls, data: MarketData, split: str, **kwargs):
        """Env over the 'train', 'val' or 'test' split. Train uses random-start episodes;
        val/test do one deterministic pass over the whole split."""
        a, b = data.split_bounds()[split]
        lookback = kwargs.get("lookback", cfg.LOOKBACK)
        kwargs.setdefault("episode_len", cfg.EPISODE_LEN if split == "train" else None)
        return cls(data, max(a, lookback), b, **kwargs)

    def _cash_weights(self) -> np.ndarray:
        w = np.zeros(self.n_assets + 1)
        w[-1] = 1.0
        return w

    def _obs(self, t: int) -> dict:
        return {
            "seq": self._X[t - self.lookback + 1 : t + 1].copy(),
            "weights": np.clip(self._w, 0.0, 1.0).astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        offset = int(self.np_random.integers(0, self.span - self.ep_len + 1)) if self.span > self.ep_len else 0
        self._first = self.start + offset  # first return day earned in this episode
        self._k = 0
        self._w = self._cash_weights()
        return self._obs(self._first - 1), {}

    def step(self, action):
        t = self._first - 1 + self._k  # decision made at the close of day t
        target = action_to_weights(action, self.logit_scale)
        growth, w_next, traded, cost = portfolio_step(
            self._w, target, self.data.returns[t + 1], self.data.rf[t + 1], self.cost_rate
        )
        self._w = w_next
        self._k += 1
        reward = self.reward_scale * float(np.log(growth))
        truncated = self._k >= self.ep_len
        info = {"net_return": growth - 1.0, "traded": traded, "cost": cost, "target_weights": target, "day": t + 1}
        return self._obs(t + 1), reward, False, truncated, info

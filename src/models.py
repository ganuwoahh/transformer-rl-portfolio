"""Feature extractors for Stable-Baselines3 PPO.

Both take the Dict observation from PortfolioEnv ("seq": (L, D), "weights": (N+1,)) and output
[encoded market history | today's macro features | current weights], so the actor and critic
heads that sit on top are identical and the only difference between models is the encoder.

  * TransformerExtractor : 2-layer, 4-head Transformer encoder over the L-day window.
  * MLPExtractor         : MLP over the flattened window (same information as the Transformer).
                           window=1 gives a "last day only" MLP with no history.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, macro_dim: int = 6):
        L, D = observation_space["seq"].shape
        n_w = observation_space["weights"].shape[0]
        super().__init__(observation_space, features_dim=d_model + macro_dim + n_w)
        self.macro_dim = macro_dim
        self.embed = nn.Linear(D, d_model)
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))
        nn.init.normal_(self.pos, std=0.02)
        # dropout=0: PPO compares action probabilities before/after each update, and dropout noise would corrupt that
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)

    def forward(self, obs):
        seq = obs["seq"]                                   # (B, L, D)
        h = self.encoder(self.embed(seq) + self.pos)[:, -1]  # token for the most recent day
        macro = seq[:, -1, -self.macro_dim:]               # today's macro features (last M columns)
        return torch.cat([h, macro, obs["weights"]], dim=1)


class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, window=None, hidden: int = 64, macro_dim: int = 6):
        L, D = observation_space["seq"].shape
        n_w = observation_space["weights"].shape[0]
        super().__init__(observation_space, features_dim=hidden + macro_dim + n_w)
        self.window = L if window is None else window
        self.macro_dim = macro_dim
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(self.window * D, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, obs):
        seq = obs["seq"]
        h = self.net(seq[:, -self.window:])
        macro = seq[:, -1, -self.macro_dim:]
        return torch.cat([h, macro, obs["weights"]], dim=1)


def policy_kwargs_for(name: str, macro_dim: int) -> dict:
    """SB3 policy_kwargs for 'transformer', 'mlp' (full window) or 'mlp_last' (last day only)."""
    if name == "transformer":
        cls, kw = TransformerExtractor, dict(macro_dim=macro_dim)
    elif name == "mlp":
        cls, kw = MLPExtractor, dict(macro_dim=macro_dim)
    elif name == "mlp_last":
        cls, kw = MLPExtractor, dict(window=1, macro_dim=macro_dim)
    else:
        raise ValueError(f"unknown model {name!r}")
    return dict(features_extractor_class=cls, features_extractor_kwargs=kw,
                net_arch=dict(pi=[64], vf=[64]), log_std_init=-1.0)

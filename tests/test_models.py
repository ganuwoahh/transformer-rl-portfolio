import numpy as np
import pytest

from src import config as cfg
from src.evaluate import run_policy
from src.features import build_market_data
from src.portfolio import action_to_weights, backtest
from tests.synthetic import make_synthetic

gym = pytest.importorskip("gymnasium")
from src.env import PortfolioEnv  # noqa: E402


class _FixedModel:
    """Stand-in for a trained SB3 model that always outputs the same action."""

    def __init__(self, action):
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, None


def _md():
    prices, macro = make_synthetic()
    return build_market_data(prices, macro)


def test_run_policy_matches_backtest():
    md = _md()
    env = PortfolioEnv.for_split(md, "val")
    action = np.random.default_rng(5).uniform(-1, 1, env.action_space.shape).astype(np.float32)
    res = run_policy(_FixedModel(action), env)
    target = action_to_weights(action)
    ref = backtest(md, env.start, env.end, lambda t, w, k: target, cost_bps=cfg.COST_BPS)
    np.testing.assert_allclose(res.net_returns, ref.net_returns, atol=1e-12)
    np.testing.assert_allclose(res.weights, ref.weights)
    assert res.dates.equals(ref.dates)


def test_extractors_shapes_and_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from src.models import MLPExtractor, TransformerExtractor

    md = _md()
    macro_dim = md.macro_features.shape[1]
    env = PortfolioEnv.for_split(md, "train")
    obs, _ = env.reset(seed=0)
    batch = {k: torch.as_tensor(np.stack([v, v]), dtype=torch.float32) for k, v in obs.items()}
    for ext in (
        TransformerExtractor(env.observation_space, macro_dim=macro_dim),
        MLPExtractor(env.observation_space, macro_dim=macro_dim),
        MLPExtractor(env.observation_space, window=1, macro_dim=macro_dim),
    ):
        out = ext(batch)
        assert out.shape == (2, ext.features_dim)
        out.sum().backward()
        assert all(p.grad is not None for p in ext.parameters())


def test_ppo_trains_a_few_steps_with_each_model():
    pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.models import policy_kwargs_for

    md = _md()
    for name in ("transformer", "mlp", "mlp_last"):
        venv = DummyVecEnv([lambda: PortfolioEnv.for_split(md, "train") for _ in range(2)])
        model = sb3.PPO("MultiInputPolicy", venv, n_steps=64, batch_size=64, n_epochs=1, device="cpu", seed=0,
                        policy_kwargs=policy_kwargs_for(name, md.macro_features.shape[1]))
        model.learn(total_timesteps=256)
        res = run_policy(model, PortfolioEnv.for_split(md, "val"))   # real model through the evaluation path
        assert np.isfinite(res.net_returns).all()

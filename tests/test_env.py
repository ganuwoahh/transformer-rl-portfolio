"""Accounting, environment and baseline tests. Run: python -m pytest -q tests"""
import numpy as np
import pytest

from src import config as cfg
from src.baselines import equal_weight, inverse_vol, run_baselines
from src.evaluate import compute_metrics
from src.features import build_market_data
from src.portfolio import BacktestResult, action_to_weights, backtest, portfolio_step
from tests.synthetic import make_synthetic


def _md():
    prices, macro = make_synthetic()
    return build_market_data(prices, macro)


# ----------------------------------------------------------------- accounting
def test_portfolio_step_costs_and_drift():
    c = 0.001
    # cash -> 50/50 in two assets: trade = 1.0 of notional
    g, w, traded, cost = portfolio_step([0, 0, 1.0], [0.5, 0.5, 0.0], [0.02, -0.01], 0.0, c)
    assert traded == pytest.approx(1.0) and cost == pytest.approx(c)
    assert g == pytest.approx((1 - c) * 1.005)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.51 / 1.005)
    # holding the drifted weights = no trade = no cost
    g2, _, traded2, cost2 = portfolio_step(w, w, [0.0, 0.0], 0.0, c)
    assert traded2 == pytest.approx(0.0) and cost2 == 0.0 and g2 == pytest.approx(1.0)
    # all risky -> all cash trades the whole risky book
    _, _, traded3, _ = portfolio_step([0.5, 0.5, 0.0], [0, 0, 1.0], [0.0, 0.0], 0.0, c)
    assert traded3 == pytest.approx(1.0)
    # cash earns the risk-free rate
    g4, _, _, _ = portfolio_step([0, 0, 1.0], [0, 0, 1.0], [0.1, 0.1], 0.0002, c)
    assert g4 == pytest.approx(1.0002)


def test_action_to_weights():
    w = action_to_weights(np.zeros(13))
    assert w.sum() == pytest.approx(1.0) and np.allclose(w, 1 / 13)
    w = action_to_weights(np.random.default_rng(0).uniform(-5, 5, 13))  # out-of-range actions are clipped
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()


def test_backtest_alignment_and_cash_return():
    md = _md()
    a, b = 700, 900
    n = len(md.assets)
    cash = np.zeros(n + 1)
    cash[-1] = 1.0
    res = backtest(md, a, b, lambda t, w, k: cash, cost_bps=5.0)
    np.testing.assert_allclose(res.net_returns, md.rf[a:b])      # all cash earns exactly rf over days a..b-1
    asset0 = np.zeros(n + 1)
    asset0[0] = 1.0
    res = backtest(md, a, b, lambda t, w, k: asset0, cost_bps=0.0)
    np.testing.assert_allclose(res.net_returns, md.returns[a:b, 0])  # all-in asset 0, free trading
    assert res.dates[0] == md.dates[a] and res.dates[-1] == md.dates[b - 1]


def test_backtest_rejects_invalid_weights():
    md = _md()
    with pytest.raises(ValueError):
        backtest(md, 700, 710, lambda t, w, k: np.ones(len(md.assets) + 1), cost_bps=5.0)


# ----------------------------------------------------------------- baselines
def test_equal_weight_only_trades_on_rebalance_days():
    md = _md()
    res = backtest(md, 700, 900, equal_weight(md), cost_bps=5.0)
    assert res.traded[0] == pytest.approx(1.0)                     # initial purchase out of cash
    off = np.array([k % cfg.REBALANCE_EVERY != 0 for k in range(len(res.traded))])
    assert np.allclose(res.traded[off], 0.0)                       # no trading between rebalances
    assert (res.traded[~off][1:] > 0).all()                        # rebalance days do trade (drift)
    assert np.allclose(res.weights[0, :-1], 1 / len(md.assets)) and res.weights[0, -1] == 0.0


def test_inverse_vol_uses_only_past_data():
    md = _md()
    k0 = 0
    w_a = inverse_vol(md)(699, np.zeros(13), k0)
    # corrupt all returns AFTER day 699: the weights chosen at close 699 must not change
    import dataclasses
    future = md.returns.copy()
    future[700:] *= 5.0
    md2 = dataclasses.replace(md, returns=future)
    w_b = inverse_vol(md2)(699, np.zeros(13), k0)
    np.testing.assert_allclose(w_a, w_b)
    assert w_a.sum() == pytest.approx(1.0) and (w_a >= 0).all() and w_a[-1] == 0.0


def test_run_baselines_keys_and_lengths():
    md = _md()
    res = run_baselines(md, 700, 900)
    assert set(res) == {"Equal-weight (monthly)", "Inverse-vol (monthly)", "SPY buy & hold"}
    assert all(len(r.net_returns) == 200 for r in res.values())


# ----------------------------------------------------------------- metrics
def test_metrics_known_values():
    r = np.array([0.10, -0.50, 0.20])
    res = BacktestResult(None, r, np.zeros(3), np.zeros((3, 2)), np.zeros(3))
    m = compute_metrics(res, periods=252)
    assert m["cum_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert m["max_drawdown"] == pytest.approx(-0.5)
    assert m["ann_return"] == pytest.approx((1.1 * 0.5 * 1.2) ** (252 / 3) - 1)
    # an initial loss counts as drawdown from the starting value of 1.0
    res = BacktestResult(None, np.array([-0.2, 0.1]), np.zeros(2), np.zeros((2, 2)), np.zeros(2))
    assert compute_metrics(res)["max_drawdown"] == pytest.approx(-0.2)
    r = np.linspace(-0.05, 0.05, 200)
    m = compute_metrics(BacktestResult(None, r, np.full(200, 0.01), np.zeros((200, 2)), np.zeros(200)))
    assert m["cvar_95"] <= np.quantile(r, 0.05) + 1e-12
    assert m["turnover"] == pytest.approx(0.01 * 252)


# ----------------------------------------------------------------- environment
gym = pytest.importorskip("gymnasium")
from src.env import PortfolioEnv  # noqa: E402


def test_env_observation_alignment_and_reward():
    md = _md()
    env = PortfolioEnv.for_split(md, "val")
    start = env.start
    obs, _ = env.reset(seed=0)
    assert env.observation_space["seq"].shape == (cfg.LOOKBACK, env.obs_dim)
    # first decision is at the close of day start-1: the last row of "seq" must be that day's features
    np.testing.assert_array_equal(obs["seq"][-1], env._X[start - 1])
    assert obs["weights"][-1] == 1.0                                 # starts in cash
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, term, trunc, info = env.step(a)
    assert info["day"] == start                                      # earns the return of day `start`
    np.testing.assert_array_equal(obs["seq"][-1], env._X[start])     # next obs includes day `start`
    w = action_to_weights(a)
    growth = (1 - cfg.COST_BPS / 1e4 * np.abs(w[:-1]).sum()) * (w[:-1] @ (1 + md.returns[start]) + w[-1] * (1 + md.rf[start]))
    assert reward == pytest.approx(cfg.REWARD_SCALE * np.log(growth))
    assert env.observation_space.contains(obs) and not term and not trunc


def test_env_matches_backtest():
    md = _md()
    env = PortfolioEnv.for_split(md, "val")
    rng = np.random.default_rng(3)
    action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
    env.reset(seed=0)
    rets, trunc = [], False
    while not trunc:
        _, _, _, trunc, info = env.step(action)
        rets.append(info["net_return"])
    target = action_to_weights(action)
    res = backtest(md, env.start, env.end, lambda t, w, k: target, cost_bps=cfg.COST_BPS)
    np.testing.assert_allclose(rets, res.net_returns, atol=1e-12)    # same numbers via two code paths
    assert len(rets) == env.end - env.start


def test_env_random_start_episodes_stay_in_bounds():
    md = _md()
    env = PortfolioEnv.for_split(md, "train")
    firsts, last_days = set(), []
    for seed in range(15):
        env.reset(seed=seed)
        firsts.add(env._first)
        assert env._first >= env.start
        steps, trunc, info = 0, False, {}
        while not trunc:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            steps += 1
        assert steps == cfg.EPISODE_LEN
        last_days.append(info["day"])
    assert len(firsts) > 5                                           # starts really are random
    assert max(last_days) < env.end                                  # never earns a return outside the split


def test_env_rejects_insufficient_history():
    md = _md()
    with pytest.raises(ValueError):
        PortfolioEnv(md, start=10, end=300)

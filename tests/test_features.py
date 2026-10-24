"""Leakage and sanity tests for the feature pipeline. Run: python -m pytest -q tests"""
import numpy as np
import pandas as pd

from src import config as cfg
from src.features import build_market_data, causal_zscore
from tests.synthetic import make_synthetic


def _common(a, b):
    """Row indices of the dates shared by two MarketData objects."""
    shared = a.dates.intersection(b.dates)
    return a.dates.get_indexer(shared), b.dates.get_indexer(shared)


def test_shapes_and_finite():
    prices, macro = make_synthetic()
    md = build_market_data(prices, macro)
    T, N, F = md.asset_features.shape
    assert N == len(cfg.ASSETS) and F == len(md.asset_feature_names)
    assert md.macro_features.shape == (T, len(md.macro_feature_names))
    assert md.returns.shape == (T, N) and md.rf.shape == (T,) and md.benchmark_returns.shape == (T,)
    assert np.isfinite(md.asset_features).all() and np.isfinite(md.macro_features).all()
    assert np.abs(md.asset_features).max() <= cfg.ZSCORE_CLIP + 1e-6


def test_zscore_is_roughly_standardised():
    prices, macro = make_synthetic()
    md = build_market_data(prices, macro)
    x = md.asset_features[:, :, 0]  # log_ret z-scores
    assert abs(x.mean()) < 0.1 and 0.8 < x.std() < 1.2


def test_truncation_does_not_change_past_features():
    """Features on date t must be identical whether or not data after t exists."""
    prices, macro = make_synthetic()
    full = build_market_data(prices, macro)
    cut = prices.index[2000]
    part = build_market_data(prices.loc[:cut], macro.loc[:cut])
    i_full, i_part = _common(full, part)
    assert len(i_full) > 500
    np.testing.assert_allclose(full.asset_features[i_full], part.asset_features[i_part], atol=1e-6)
    np.testing.assert_allclose(full.macro_features[i_full], part.macro_features[i_part], atol=1e-6)
    np.testing.assert_allclose(full.returns[i_full], part.returns[i_part])
    np.testing.assert_allclose(full.rf[i_full], part.rf[i_part])


def test_perturbing_the_future_does_not_change_past_features():
    prices, macro = make_synthetic()
    base = build_market_data(prices, macro)
    cut = prices.index[1800]
    p2, m2 = prices.copy(), macro.copy()
    rng = np.random.default_rng(1)
    future = p2.index > cut
    for field in ["open", "high", "low", "close"]:
        p2.loc[future, field] = p2.loc[future, field].to_numpy() * rng.uniform(0.5, 2.0, (future.sum(), p2[field].shape[1]))
    m2.loc[m2.index > cut, "vix"] *= 3.0
    pert = build_market_data(p2, m2)
    keep = base.dates <= cut
    np.testing.assert_allclose(base.asset_features[keep], pert.asset_features[: keep.sum()], atol=1e-6)
    np.testing.assert_allclose(base.macro_features[keep], pert.macro_features[: keep.sum()], atol=1e-6)


def test_zscore_excludes_current_observation():
    s = pd.DataFrame({"x": np.r_[np.zeros(300), 1000.0]})
    z = causal_zscore(s, window=252, clip=1e9)
    # the spike is judged against a history of zeros (std==0 -> NaN), not against itself
    assert not np.isfinite(z["x"].iloc[-1])


def test_split_bounds_partition_the_sample():
    prices, macro = make_synthetic()
    md = build_market_data(prices, macro)
    b = md.split_bounds(train_end="2018-12-31", val_end="2020-12-31")
    assert b["train"][0] == 0 and b["train"][1] == b["val"][0]
    assert b["val"][1] == b["test"][0] and b["test"][1] == md.n_steps
    tr, va, te = (md.dates[a:z] for a, z in (b["train"], b["val"], b["test"]))
    assert tr.max() <= pd.Timestamp("2018-12-31") < va.min()
    assert va.max() <= pd.Timestamp("2020-12-31") < te.min()

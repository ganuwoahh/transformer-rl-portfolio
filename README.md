# Transformer-Based Deep RL for Portfolio Allocation

PPO agents with a Transformer encoder policy vs. an MLP policy vs. classic baselines, trained on
daily multi-asset data with transaction costs and a strict, leak-free train / validation / test split.

> **Status: work in progress.** Data and feature pipeline done; environment, training and
> evaluation in progress. Results, methodology and limitations will be filled in here.

## Repo layout

```
src/
  config.py     all settings (universe, dates, feature windows)
  data.py       yfinance download, caching, calendar alignment
  features.py   features + causal z-scoring, MarketData container
  env.py        Gymnasium environment                       (TODO)
  models.py     MLP / Transformer feature extractors        (TODO)
  baselines.py  equal-weight, buy-and-hold                  (TODO)
  train.py      PPO training                                (TODO)
  evaluate.py   metrics + plots                             (TODO)
tests/          leakage + sanity tests for the feature pipeline
notebooks/      Colab runner
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.data        # download + cache data
python -m src.features    # build features, print summary
python -m pytest -q tests # leakage tests
```

## Leakage controls

* Every feature at date *t* uses data up to the close of *t* only.
* Z-scores use the mean/std of the *previous* 252 trading days (never the current or future values).
* Tests check that truncating or perturbing future data leaves all past features unchanged.
* Train 2016-2020, validation 2021-2022, test 2023-present; test set untouched until the end.

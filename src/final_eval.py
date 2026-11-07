"""
model chosen by val sharpe
one shot testing
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from . import config as cfg
from .baselines import run_baselines
from .data import load_market_data
from .env import PortfolioEnv
from .evaluate import compute_metrics, metrics_table, pretty, run_policy
from .features import build_market_data


def save_result(res, path):
    np.savez(path, dates=np.array(res.dates.strftime("%Y-%m-%d").tolist(), dtype="U10"),
             net_returns=res.net_returns, traded=res.traded, weights=res.weights, rf=res.rf)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std across seeds, one row per model."""
    out = {}
    for name, g in df.groupby("model"):
        out[name] = {
            "Sharpe": f"{g.sharpe.mean():.2f} ± {g.sharpe.std():.2f}",
            "Ann. return": f"{g.ann_return.mean() * 100:.1f}% ± {g.ann_return.std() * 100:.1f}",
            "Max DD": f"{g.max_drawdown.mean() * 100:.1f}% ± {g.max_drawdown.std() * 100:.1f}",
            "Calmar": f"{g.calmar.mean():.2f} ± {g.calmar.std():.2f}",
            "CVaR 95% (1d)": f"{g.cvar_95.mean() * 100:.2f}% ± {g.cvar_95.std() * 100:.2f}",
            "Turnover (x/yr)": f"{g.turnover.mean():.1f} ± {g.turnover.std():.1f}",
            "seeds": len(g),
        }
    return pd.DataFrame(out).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=cfg.COST_BPS)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    prices, macro = load_market_data()
    md = build_market_data(prices, macro)
    a, b = md.split_bounds()["test"]
    out = cfg.RESULTS_DIR / "test" / f"{args.tag}_{args.cost_bps:g}bps"
    out.mkdir(parents=True, exist_ok=True)
    print(f"TEST: {md.dates[a].date()} -> {md.dates[b - 1].date()} ({b - a} days), costs {args.cost_bps:g} bps")

    rows, per_run = [], {}
    for run in sorted((cfg.RESULTS_DIR / "runs" / args.tag).glob("*_seed*")):
        if not (run / "best.zip").exists():
            continue
        name, seed = run.name.rsplit("_seed", 1)
        env = PortfolioEnv.for_split(md, "test", cost_bps=args.cost_bps)
        assert (env.start, env.end) == (a, b)
        res = run_policy(PPO.load(run / "best", device="cpu"), env)
        save_result(res, out / f"{run.name}.npz")
        rows.append({"model": name, "seed": int(seed), **compute_metrics(res)})
        per_run[f"{name} seed{seed}"] = res
    if not rows:
        raise SystemExit(f"no finished runs found under {cfg.RESULTS_DIR / 'runs' / args.tag}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_run_metrics.csv", index=False)

    base = run_baselines(md, a, b, args.cost_bps)
    for k, r in base.items():
        save_result(r, out / f"baseline_{re.sub(r'[^a-z0-9]+', '_', k.lower()).strip('_')}.npz")

    print("\n=== Agents: mean ± std across seeds ===")
    print(summarize(df).to_string())
    print("\n=== Baselines ===")
    print(pretty(metrics_table(base)).to_string())
    print("\n=== Every individual run ===")
    print(pretty(metrics_table(per_run)).to_string())
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()

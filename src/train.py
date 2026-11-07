"""
PPO training with validation-Sharpe checkpointing and early stopping.

Example:  python -m src.train --model transformer --seed 0 --timesteps 500000
Output :  RESULTS_DIR/runs/<tag>/<model>_seed<seed>/{best.zip, last.zip, val_curve.csv, summary.json}
A run whose summary.json already exists is skipped, so interrupted batches can simply be re-launched.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from . import config as cfg
from .data import load_market_data
from .env import PortfolioEnv
from .evaluate import compute_metrics, run_policy
from .features import build_market_data
from .models import policy_kwargs_for

# n_steps is per environment: 4 envs x 512 = 2048 samples per update
PPO_KWARGS = dict(n_steps=512, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, learning_rate=3e-4)


class ValSharpeCallback(BaseCallback):
    """Every `eval_freq` timesteps: run the deterministic policy over the validation split,
    keep the best checkpoint by Sharpe, stop after `patience` evaluations without improvement."""

    def __init__(self, val_env, out_dir, eval_freq: int, patience: int):
        super().__init__()
        self.val_env, self.out_dir = val_env, out_dir
        self.eval_freq, self.patience = eval_freq, patience
        self.next_eval = eval_freq
        self.best_sharpe, self.best_step, self.bad = -np.inf, 0, 0
        self.rows = []
        self.t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.eval_freq
        m = compute_metrics(run_policy(self.model, self.val_env))
        if np.isfinite(m["sharpe"]) and m["sharpe"] > self.best_sharpe:
            self.best_sharpe, self.best_step, self.bad = m["sharpe"], self.num_timesteps, 0
            self.model.save(self.out_dir / "best")
        else:
            self.bad += 1
        self.rows.append({"timesteps": self.num_timesteps, **m})
        pd.DataFrame(self.rows).to_csv(self.out_dir / "val_curve.csv", index=False)
        print(f"[{self.num_timesteps:>8,}] val Sharpe {m['sharpe']:+.2f} | ann ret {m['ann_return'] * 100:+6.1f}% | "
              f"turnover {m['turnover']:6.1f}x/yr | best {self.best_sharpe:+.2f} @ {self.best_step:,} | "
              f"{time.time() - self.t0:.0f}s", flush=True)
        return not (self.patience and self.bad >= self.patience)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["transformer", "mlp", "mlp_last"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--eval-freq", type=int, default=20_480)
    ap.add_argument("--patience", type=int, default=10, help="stop after this many evals without a new best (0 = never)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--force", action="store_true", help="re-run even if a finished run exists")
    args = ap.parse_args()

    out_dir = cfg.RESULTS_DIR / "runs" / args.tag / f"{args.model}_seed{args.seed}"
    if (out_dir / "summary.json").exists() and not args.force:
        print(f"already finished, skipping: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, macro = load_market_data()
    md = build_market_data(prices, macro)
    train_env = DummyVecEnv([lambda: PortfolioEnv.for_split(md, "train") for _ in range(args.n_envs)])
    val_env = PortfolioEnv.for_split(md, "val")

    model = PPO("MultiInputPolicy", train_env, seed=args.seed, device=args.device, verbose=0,
                policy_kwargs=policy_kwargs_for(args.model, macro_dim=md.macro_features.shape[1]), **PPO_KWARGS)
    n_params = int(sum(p.numel() for p in model.policy.parameters()))
    print(f"{args.model} seed {args.seed}: {n_params:,} parameters, device={model.device}", flush=True)

    cb = ValSharpeCallback(val_env, out_dir, args.eval_freq, args.patience)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    secs = time.time() - t0

    model.save(out_dir / "last")
    if not (out_dir / "best.zip").exists():
        model.save(out_dir / "best")
    summary = dict(model=args.model, seed=args.seed, params=n_params, device=str(model.device),
                   timesteps_run=int(model.num_timesteps), best_val_sharpe=float(cb.best_sharpe),
                   best_step=int(cb.best_step), seconds=round(secs, 1),
                   steps_per_sec=round(model.num_timesteps / secs, 1))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done: {summary}", flush=True)


if __name__ == "__main__":
    main()

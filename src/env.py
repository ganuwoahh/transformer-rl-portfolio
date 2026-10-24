"""Gymnasium portfolio environment. (TODO: Day 1, next step)

Agreed design
-------------
* Observation (Dict space):
    "seq"     : (L, N*F + M) last L=60 rows of z-scored asset + macro features
    "weights" : (N+1,) current portfolio weights incl. a cash slot (w_{t-1} in the state vector)
* Action: N+1 logits -> softmax -> target weights (long-only, fully invested; cash earns `rf`).
* Transition: rebalance at close t, earn `returns[t+1]` and `rf[t+1]`.
* Reward: log(1 + w . r_{t+1} - cost * turnover), cost = 5-10 bps per unit of one-way turnover.
* Training episodes: random start, 252 days. Evaluation: one full pass over a split.
* Uses MarketData.split_bounds() so val/test episodes can look back into earlier rows.
"""

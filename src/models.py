"""SB3 feature extractors. (TODO: Day 2)

* TransformerExtractor: linear embed -> learned positional encoding -> 2-layer / 4-head
  TransformerEncoder over the L-day window -> last-token (or mean) pooling -> concat with the
  current weights -> actor / critic heads.
* MLPExtractor: the ablation baseline over the same observation (flattened window).
"""

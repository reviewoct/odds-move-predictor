"""Causal feature construction and labelling for tick-level odds data."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

DOWN, FLAT, UP = 0, 1, 2
N_FEATURES = 3


def build_features(
    prices: np.ndarray, spread_mean: Optional[np.ndarray] = None, fit_end: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a (T, K) price matrix into (T, K, 3) features.

    Features per book, all computable from information available at time t:
        0  last log-price change (x100)
        1  deviation from the cross-book consensus, de-meaned per book so that
           a constant margin difference is not mistaken for a mispricing (x100)
        2  log(1 + ticks since this book last moved)

    `spread_mean` is the per-book normalisation constant. If it is not given
    it is fitted on rows [0, fit_end) only, so no test data leaks into it.
    """
    logp = np.log(prices.astype(np.float64))
    T, K = logp.shape

    ret = np.zeros_like(logp)
    ret[1:] = logp[1:] - logp[:-1]

    spread = logp - logp.mean(axis=1, keepdims=True)
    if spread_mean is None:
        spread_mean = spread[: fit_end or T].mean(axis=0)
    dev = spread - spread_mean

    moved = ret != 0
    idx = np.where(moved, np.arange(T)[:, None], 0)
    last_move = np.maximum.accumulate(idx, axis=0)
    age = np.log1p(np.arange(T)[:, None] - last_move)

    feats = np.stack([ret * 100.0, dev * 100.0, age], axis=-1).astype(np.float32)
    return feats, spread_mean


def make_labels(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Direction of each book's *next* price change within `horizon` ticks.

    Returns (T, K) int64 with DOWN / FLAT / UP. The last `horizon` rows do not
    have a full look-ahead window and must not be used for training or scoring.
    """
    logp = np.log(prices.astype(np.float64))
    T, K = logp.shape
    ret = np.zeros_like(logp)
    ret[1:] = logp[1:] - logp[:-1]

    label = np.full((T, K), FLAT, dtype=np.int64)
    # Walk from the far end of the horizon inwards so the nearest move wins.
    for h in range(horizon, 0, -1):
        future = np.zeros_like(ret)
        future[:-h] = ret[h:]
        label = np.where(future > 0, UP, np.where(future < 0, DOWN, label))
    return label


class WindowSampler:
    """Serves (window, label) batches by index without materialising windows."""

    def __init__(self, feats: np.ndarray, labels: np.ndarray, window: int, horizon: int):
        self.feats = torch.from_numpy(feats)
        self.labels = torch.from_numpy(labels)
        self.window = window
        self.horizon = horizon
        self._offsets = torch.arange(-window + 1, 1)

    def valid_range(self, start: int, end: int) -> Tuple[int, int]:
        """Clip [start, end) to indices with a full window and full look-ahead."""
        lo = max(start, self.window - 1)
        hi = min(end, self.feats.shape[0] - self.horizon)
        return lo, hi

    def batch(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feats[idx[:, None] + self._offsets]  # (B, L, K, F)
        y = self.labels[idx]  # (B, K)
        return x, y

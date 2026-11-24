"""Synthetic multi-bookmaker odds feed with a hidden regime process.

Each bookmaker is a black box that tracks a shared latent "fair" price with
its own reaction speed, dead-band, margin and tick grid. A hidden Markov chain
switches the market between calm / active / shock regimes, which changes both
the volatility of the fair price and how quickly books re-price.

The point of the simulator is to have a feed where the ground truth is known,
so the inverse-modelling pipeline can be validated end to end before it is
pointed at real tick data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

CALM, ACTIVE, SHOCK = 0, 1, 2


@dataclass
class SimConfig:
    n_steps: int = 200_000
    seed: int = 0
    # Per-book behaviour. Book 0 is the sharp "leader", the rest follow.
    update_prob: List[float] = field(default_factory=lambda: [0.60, 0.30, 0.18, 0.10, 0.06])
    deadband: List[float] = field(default_factory=lambda: [0.004, 0.008, 0.010, 0.014, 0.018])
    margin: List[float] = field(default_factory=lambda: [0.025, 0.045, 0.050, 0.060, 0.075])
    quote_noise: float = 0.002
    tick: float = 0.01
    # Regime process (rows: from, cols: to).
    transition: List[List[float]] = field(
        default_factory=lambda: [
            [0.995, 0.004, 0.001],
            [0.020, 0.970, 0.010],
            [0.050, 0.150, 0.800],
        ]
    )
    regime_vol: List[float] = field(default_factory=lambda: [0.0015, 0.005, 0.015])
    regime_speedup: List[float] = field(default_factory=lambda: [1.0, 1.6, 2.5])
    jump_prob_in_shock: float = 0.05
    jump_scale: float = 0.08
    mean_reversion: float = 0.0005

    @property
    def n_books(self) -> int:
        return len(self.update_prob)


def simulate(cfg: SimConfig = SimConfig()) -> Dict[str, np.ndarray]:
    """Run the simulator.

    Returns a dict with
        prices   (T, K) float32 decimal odds as quoted by each book
        regime   (T,)   int8    hidden regime id
        fair     (T,)   float32 latent fair logit
    """
    rng = np.random.default_rng(cfg.seed)
    T, K = cfg.n_steps, cfg.n_books
    P = np.asarray(cfg.transition)
    assert np.allclose(P.sum(axis=1), 1.0), "transition rows must sum to 1"
    cum_P = P.cumsum(axis=1)

    upd = np.asarray(cfg.update_prob)
    band = np.asarray(cfg.deadband)
    margin = np.asarray(cfg.margin)

    prices = np.empty((T, K), dtype=np.float32)
    regime = np.empty(T, dtype=np.int8)
    fair = np.empty(T, dtype=np.float32)

    x = 0.0  # fair logit
    q = np.zeros(K)  # each book's internal view of the fair logit
    r = CALM

    u_regime = rng.random(T)
    z = rng.standard_normal(T)
    u_jump = rng.random(T)
    z_jump = rng.standard_normal(T)
    u_upd = rng.random((T, K))
    z_quote = rng.standard_normal((T, K)) * cfg.quote_noise

    for t in range(T):
        r = int(np.searchsorted(cum_P[r], u_regime[t]))
        r = min(r, 2)
        x = x * (1.0 - cfg.mean_reversion) + cfg.regime_vol[r] * z[t]
        if r == SHOCK and u_jump[t] < cfg.jump_prob_in_shock:
            x += cfg.jump_scale * z_jump[t]

        # A book re-prices only if it is stale by more than its dead-band,
        # and then only with its own (regime-dependent) reaction probability.
        stale = np.abs(x - q) > band
        fire = stale & (u_upd[t] < np.minimum(1.0, upd * cfg.regime_speedup[r]))
        q = np.where(fire, x + z_quote[t], q)

        prob = 1.0 / (1.0 + np.exp(-q))
        quoted = 1.0 / (prob * (1.0 + margin))  # margin shortens the odds
        prices[t] = np.round(quoted / cfg.tick) * cfg.tick
        regime[t] = r
        fair[t] = x

    return {"prices": prices, "regime": regime, "fair": fair}

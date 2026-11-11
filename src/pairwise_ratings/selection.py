"""Pair-selection strategies. Each returns an unordered pair (a, b) with a < b."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pairwise_ratings.glicko2 import Glicko2

STRATEGIES = ("random", "uncertainty", "closest", "info_gain")


def _argmax_pair(score: NDArray[np.float64], rng: np.random.Generator) -> tuple[int, int]:
    """Best pair in the upper triangle, ties broken uniformly at random."""
    rows, cols = np.triu_indices(score.shape[0], k=1)
    values = score[rows, cols]
    best = np.flatnonzero(values >= values.max() - 1e-12 * max(1.0, abs(values.max())))
    pick = int(rng.choice(best))
    return int(rows[pick]), int(cols[pick])


def random_pair(n: int, rng: np.random.Generator) -> tuple[int, int]:
    a, b = rng.choice(n, size=2, replace=False)
    return (int(a), int(b)) if a < b else (int(b), int(a))


def uncertainty_pair(model: Glicko2, rng: np.random.Generator) -> tuple[int, int]:
    """The player with the largest rating deviation against a uniformly random opponent.

    Pairing the two most uncertain players instead locks into a few repeated pairs: after one round
    every deviation is nearly equal and the same players keep coming out on top.
    """
    phi = model.phi
    top = np.flatnonzero(phi >= phi.max() - 1e-12)
    a = int(rng.choice(top))
    b = int(rng.choice(np.delete(np.arange(phi.size), a)))
    return (a, b) if a < b else (b, a)


def closest_pair(p: NDArray[np.float64], rng: np.random.Generator) -> tuple[int, int]:
    """The most even match: maximize p(1 - p) over predicted win probabilities."""
    return _argmax_pair(p * (1.0 - p), rng)


def info_gain_scores(model: Glicko2) -> NDArray[np.float64]:
    """Predicted drop in total posterior variance (Glicko-2 scale) if i and j are compared.

    For player i the Glicko-2 update gives 1/phi'^2 = 1/phi*^2 + g(phi_j)^2 E (1 - E), whatever the
    outcome, so the reduction phi*^2 - phi'^2 is known before the comparison is made.
    phi* uses the current volatility in place of the not-yet-computed new one.
    """
    phi_star2 = model.phi**2 + model.sigma**2
    g_j = 1.0 / np.sqrt(1.0 + 3.0 * model.phi**2 / math.pi**2)
    diff = model.mu[:, None] - model.mu[None, :]
    e = 1.0 / (1.0 + np.exp(-g_j[None, :] * diff))
    info = g_j[None, :] ** 2 * e * (1.0 - e)
    reduction = phi_star2[:, None] - 1.0 / (1.0 / phi_star2[:, None] + info)
    out: NDArray[np.float64] = reduction + reduction.T
    return out


def info_gain_pair(model: Glicko2, rng: np.random.Generator) -> tuple[int, int]:
    return _argmax_pair(info_gain_scores(model), rng)


def select(strategy: str, model: Glicko2, rng: np.random.Generator) -> tuple[int, int]:
    if strategy == "random":
        return random_pair(model.mu.shape[0], rng)
    if strategy == "uncertainty":
        return uncertainty_pair(model, rng)
    if strategy == "closest":
        return closest_pair(model.win_prob_matrix(), rng)
    if strategy == "info_gain":
        return info_gain_pair(model, rng)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

"""Bradley-Terry maximum likelihood with Hunter's MM algorithm (Hunter 2004, Ann. Stat. 32(1)).

Each player gets one virtual win and one virtual loss against a fixed player of strength 1.
That weak prior keeps undefeated and winless players finite and pins the scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MAX_ITER = 10_000
# Stop when no log-strength moves by more than this in one MM step.
TOL = 1e-9


def fit(wins: NDArray[np.float64], init: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    """Return log-strengths. wins[i, j] counts comparisons where i beat j."""
    n = wins.shape[0]
    games = wins + wins.T
    w = wins.sum(axis=1) + 1.0
    gamma = np.ones(n) if init is None else np.exp(init)
    for _ in range(MAX_ITER):
        denom = (games / (gamma[:, None] + gamma[None, :])).sum(axis=1) + 2.0 / (gamma + 1.0)
        new = w / denom
        if np.max(np.abs(np.log(new) - np.log(gamma))) < TOL:
            gamma = new
            break
        gamma = new
    out: NDArray[np.float64] = np.log(gamma)
    return out


class BradleyTerry:
    """Refits from all comparisons so far whenever `refit` is called."""

    def __init__(self, n_players: int) -> None:
        self.wins = np.zeros((n_players, n_players))
        self.theta = np.zeros(n_players)

    def predict(self, a: int, b: int) -> float:
        return float(1.0 / (1.0 + np.exp(self.theta[b] - self.theta[a])))

    def win_prob_matrix(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = 1.0 / (1.0 + np.exp(self.theta[None, :] - self.theta[:, None]))
        return out

    def update(self, winner: int, loser: int) -> None:
        self.wins[winner, loser] += 1.0

    def refit(self) -> None:
        self.theta = fit(self.wins, self.theta)

    def scores(self) -> NDArray[np.float64]:
        return self.theta.copy()

"""Elo ratings with a fixed K factor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_RATING = 1500.0
# K = 32 is the common default for new players in FIDE-style systems.
DEFAULT_K = 32.0


def expected(r_a: float, r_b: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0)))


class Elo:
    def __init__(self, n_players: int, k: float = DEFAULT_K) -> None:
        self.k = k
        self.r = np.full(n_players, DEFAULT_RATING)

    def predict(self, a: int, b: int) -> float:
        return expected(float(self.r[a]), float(self.r[b]))

    def win_prob_matrix(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = 1.0 / (
            1.0 + 10.0 ** ((self.r[None, :] - self.r[:, None]) / 400.0)
        )
        return out

    def update(self, winner: int, loser: int) -> None:
        change = self.k * (1.0 - self.predict(winner, loser))
        self.r[winner] += change
        self.r[loser] -= change

    def scores(self) -> NDArray[np.float64]:
        return self.r.copy()

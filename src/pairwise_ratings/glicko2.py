"""Glicko-2, following Glickman, "Example of the Glicko-2 system" (2013).

Step numbers in comments refer to that paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Conversion factor between the Glicko and Glicko-2 scales, 400 / ln(10) (step 2).
SCALE = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOLATILITY = 0.06
# System constant tau; the paper suggests values between 0.3 and 1.2.
DEFAULT_TAU = 0.5
# Convergence tolerance for the volatility iteration (step 5).
EPSILON = 1e-6


@dataclass(frozen=True)
class Rating:
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    volatility: float = DEFAULT_VOLATILITY


def g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / math.pi**2)


def expected(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-g(phi_j) * (mu - mu_j)))


def new_volatility(phi: float, sigma: float, delta: float, v: float, tau: float) -> float:
    """Step 5: solve f(x) = 0 with the Illinois variant of regula falsi."""
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    big_a = a
    if delta * delta > phi * phi + v:
        big_b = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        big_b = a - k * tau
    f_a, f_b = f(big_a), f(big_b)
    while abs(big_b - big_a) > EPSILON:
        big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(big_c)
        if f_c * f_b <= 0:
            big_a, f_a = big_b, f_b
        else:
            f_a /= 2.0
        big_b, f_b = big_c, f_c
    return math.exp(big_a / 2.0)


def update_scaled(
    mu: float,
    phi: float,
    sigma: float,
    opponents: list[tuple[float, float, float]],
    tau: float = DEFAULT_TAU,
) -> tuple[float, float, float]:
    """One rating period on the Glicko-2 scale.

    `opponents` holds (mu_j, phi_j, score) with score 1 for a win, 0.5 for a draw, 0 for a loss.
    Returns (mu', phi', sigma').
    """
    if not opponents:
        # Step 6 for a player who did not compete: only the deviation grows.
        return mu, math.sqrt(phi * phi + sigma * sigma), sigma
    v_inv = 0.0
    delta_sum = 0.0
    for mu_j, phi_j, score in opponents:
        g_j = g(phi_j)
        e_j = expected(mu, mu_j, phi_j)
        v_inv += g_j * g_j * e_j * (1.0 - e_j)
        delta_sum += g_j * (score - e_j)
    v = 1.0 / v_inv
    delta = v * delta_sum
    sigma_new = new_volatility(phi, sigma, delta, v, tau)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + phi_new * phi_new * delta_sum
    return mu_new, phi_new, sigma_new


def rate(player: Rating, games: list[tuple[Rating, float]], tau: float = DEFAULT_TAU) -> Rating:
    """One rating period on the Glicko scale: `games` holds (opponent, score) pairs."""
    opponents = [((o.rating - DEFAULT_RATING) / SCALE, o.rd / SCALE, score) for o, score in games]
    mu, phi, sigma = update_scaled(
        (player.rating - DEFAULT_RATING) / SCALE,
        player.rd / SCALE,
        player.volatility,
        opponents,
        tau,
    )
    return Rating(mu * SCALE + DEFAULT_RATING, phi * SCALE, sigma)


class Glicko2:
    """A pool of players where each comparison is its own rating period for its two players."""

    def __init__(self, n_players: int, tau: float = DEFAULT_TAU) -> None:
        self.tau = tau
        self.mu = np.zeros(n_players)
        self.phi = np.full(n_players, DEFAULT_RD / SCALE)
        self.sigma = np.full(n_players, DEFAULT_VOLATILITY)

    def predict(self, a: int, b: int) -> float:
        """P(a beats b), using the combined deviation as in Glickman (1999)."""
        combined = math.sqrt(self.phi[a] ** 2 + self.phi[b] ** 2)
        return 1.0 / (1.0 + math.exp(-g(combined) * (self.mu[a] - self.mu[b])))

    def win_prob_matrix(self) -> NDArray[np.float64]:
        combined = np.sqrt(self.phi[:, None] ** 2 + self.phi[None, :] ** 2)
        g_c = 1.0 / np.sqrt(1.0 + 3.0 * combined**2 / math.pi**2)
        diff = self.mu[:, None] - self.mu[None, :]
        out: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-g_c * diff))
        return out

    def update(self, winner: int, loser: int) -> None:
        w = (float(self.mu[winner]), float(self.phi[winner]), float(self.sigma[winner]))
        lo = (float(self.mu[loser]), float(self.phi[loser]), float(self.sigma[loser]))
        new_w = update_scaled(*w, [(lo[0], lo[1], 1.0)], self.tau)
        new_l = update_scaled(*lo, [(w[0], w[1], 0.0)], self.tau)
        self.mu[winner], self.phi[winner], self.sigma[winner] = new_w
        self.mu[loser], self.phi[loser], self.sigma[loser] = new_l

    def scores(self) -> NDArray[np.float64]:
        return self.mu.copy()

    def ratings(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = self.mu * SCALE + DEFAULT_RATING
        return out

    def rds(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = self.phi * SCALE
        return out

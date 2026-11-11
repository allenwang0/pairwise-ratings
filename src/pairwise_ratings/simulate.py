"""Simulated comparisons among players with known latent skill.

Skill is drawn from N(0, skill_sd) on the logit scale and outcomes follow the Bradley-Terry model
P(i beats j) = 1 / (1 + exp(s_j - s_i)). Each seed spawns independent streams for skills, pair
choice and outcomes, so every condition run with the same seed faces the same players.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pairwise_ratings import metrics
from pairwise_ratings.bradley_terry import BradleyTerry
from pairwise_ratings.elo import Elo
from pairwise_ratings.glicko2 import Glicko2
from pairwise_ratings.selection import random_pair, select

SYSTEMS = ("elo", "glicko2", "bradley_terry")


@dataclass(frozen=True)
class Config:
    n_players: int = 32
    budget: int = 1280
    checkpoint_every: int = 10
    skill_sd: float = 1.0


@dataclass(frozen=True)
class Run:
    checkpoints: NDArray[np.int64]
    tau: NDArray[np.float64]
    rho: NDArray[np.float64]
    predicted: NDArray[np.float64]
    true_p: NDArray[np.float64]
    outcome: NDArray[np.float64]


def make_system(name: str, n: int) -> Elo | Glicko2 | BradleyTerry:
    if name == "elo":
        return Elo(n)
    if name == "glicko2":
        return Glicko2(n)
    if name == "bradley_terry":
        return BradleyTerry(n)
    raise ValueError(f"unknown system {name!r}; expected one of {SYSTEMS}")


def simulate(system: str, strategy: str, config: Config, seed: int) -> Run:
    if strategy != "random" and system != "glicko2":
        raise ValueError("only random selection works without Glicko-2 deviations")
    skill_rng, pick_rng, outcome_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    n = config.n_players
    skill = skill_rng.normal(0.0, config.skill_sd, n)
    model = make_system(system, n)
    checkpoints = np.arange(config.checkpoint_every, config.budget + 1, config.checkpoint_every)
    tau = np.empty(checkpoints.size)
    rho = np.empty(checkpoints.size)
    predicted = np.empty(config.budget)
    true_p = np.empty(config.budget)
    outcome = np.empty(config.budget)
    k = 0
    for t in range(config.budget):
        if isinstance(model, Glicko2):
            a, b = select(strategy, model, pick_rng)
        else:
            a, b = random_pair(n, pick_rng)
        predicted[t] = model.predict(a, b)
        true_p[t] = 1.0 / (1.0 + np.exp(skill[b] - skill[a]))
        a_wins = outcome_rng.random() < true_p[t]
        outcome[t] = float(a_wins)
        model.update(a, b) if a_wins else model.update(b, a)
        if t + 1 == checkpoints[k]:
            if isinstance(model, BradleyTerry):
                model.refit()
            est = model.scores()
            tau[k] = metrics.kendall(est, skill)
            rho[k] = metrics.spearman(est, skill)
            k += 1
            if k == checkpoints.size:
                break
    return Run(checkpoints, tau, rho, predicted, true_p, outcome)

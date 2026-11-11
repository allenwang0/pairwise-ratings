"""Ranking accuracy and calibration metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau, spearmanr

# Clip probabilities before taking logs so a confident wrong prediction costs a finite amount.
LOG_EPS = 1e-12
N_BINS = 10


def kendall(estimate: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    """Kendall tau-b; 0 when the estimate is constant (no ranking yet)."""
    if np.ptp(estimate) == 0:
        return 0.0
    return float(kendalltau(estimate, truth).statistic)


def spearman(estimate: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    if np.ptp(estimate) == 0:
        return 0.0
    return float(spearmanr(estimate, truth).statistic)


def log_loss(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    q = np.clip(p, LOG_EPS, 1.0 - LOG_EPS)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def brier(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(
    p: NDArray[np.float64], y: NDArray[np.float64], n_bins: int = N_BINS
) -> list[dict[str, float]]:
    """Equal-width bins on [0, 1]: mean prediction, observed frequency, count per bin."""
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for k in range(n_bins):
        mask = idx == k
        if mask.any():
            bins.append(
                {
                    "p_mean": float(p[mask].mean()),
                    "freq": float(y[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return bins


def ece(p: NDArray[np.float64], y: NDArray[np.float64], n_bins: int = N_BINS) -> float:
    """Expected calibration error: count-weighted mean |frequency - prediction| over bins."""
    bins = reliability(p, y, n_bins)
    total = sum(b["count"] for b in bins)
    return float(sum(b["count"] * abs(b["freq"] - b["p_mean"]) for b in bins) / total)

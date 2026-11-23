"""Run conditions over many seeds and summarize them."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pairwise_ratings import metrics
from pairwise_ratings.simulate import Config, Run, simulate

# Two-sided 95% normal quantile for the confidence band of the mean over seeds.
Z95 = 1.96
TAU_TARGET = 0.8


def band(values: NDArray[np.float64]) -> dict[str, list[float]]:
    """values has shape (seeds, checkpoints)."""
    mean = values.mean(axis=0)
    half = Z95 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return {
        "mean": mean.round(5).tolist(),
        "lo": (mean - half).round(5).tolist(),
        "hi": (mean + half).round(5).tolist(),
    }


def comparisons_to_target(
    tau: NDArray[np.float64], checkpoints: NDArray[np.int64], target: float = TAU_TARGET
) -> dict[str, Any]:
    """Per seed, the first checkpoint with tau >= target. Seeds that never reach it are counted.

    per_seed keeps every seed's value in seed order (None = never reached) so reports can show
    the distribution and pair seeds across conditions, which share players seed for seed.
    """
    hit = tau >= target
    reached = hit.any(axis=1)
    all_first = checkpoints[hit.argmax(axis=1)]
    first = all_first[reached]
    out: dict[str, Any] = {
        "target": target,
        "seeds": int(tau.shape[0]),
        "never": int((~reached).sum()),
        "per_seed": [int(v) if ok else None for v, ok in zip(all_first, reached, strict=True)],
    }
    if first.size:
        q1, med, q3 = np.percentile(first, [25, 50, 75])
        out.update(median=float(med), q1=float(q1), q3=float(q3))
    else:
        out.update(median=None, q1=None, q3=None)
    mean_curve = tau.mean(axis=0)
    crossing = np.flatnonzero(mean_curve >= target)
    out["mean_curve_crossing"] = int(checkpoints[crossing[0]]) if crossing.size else None
    return out


def calibration(runs: list[Run]) -> dict[str, Any]:
    p = np.concatenate([r.predicted for r in runs])
    y = np.concatenate([r.outcome for r in runs])
    true_p = np.concatenate([r.true_p for r in runs])
    return {
        "comparisons": int(p.size),
        "log_loss": round(metrics.log_loss(p, y), 5),
        "oracle_log_loss": round(metrics.log_loss(true_p, y), 5),
        "brier": round(metrics.brier(p, y), 5),
        "ece": round(metrics.ece(p, y), 5),
        "mean_abs_error_vs_true_p": round(float(np.mean(np.abs(p - true_p))), 5),
        "bins": metrics.reliability(p, y),
    }


def run_condition(system: str, strategy: str, config: Config, seeds: range) -> dict[str, Any]:
    runs = [simulate(system, strategy, config, s) for s in seeds]
    tau = np.stack([r.tau for r in runs])
    rho = np.stack([r.rho for r in runs])
    checkpoints = runs[0].checkpoints
    return {
        "system": system,
        "strategy": strategy,
        "tau": band(tau),
        "rho": band(rho),
        "final_tau": round(float(tau[:, -1].mean()), 4),
        "final_tau_per_seed": tau[:, -1].round(5).tolist(),
        "reach": comparisons_to_target(tau, checkpoints),
        "calibration": calibration(runs),
    }

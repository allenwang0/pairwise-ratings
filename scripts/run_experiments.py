"""Run both experiments and write results/results.json and site/card.json."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pairwise_ratings.experiment import TAU_TARGET, run_condition
from pairwise_ratings.glicko2 import Rating, rate
from pairwise_ratings.simulate import Config

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_CONDITIONS = [("elo", "random"), ("glicko2", "random"), ("bradley_terry", "random")]
STRATEGY_CONDITIONS = [
    ("glicko2", "random"),
    ("glicko2", "uncertainty"),
    ("glicko2", "closest"),
    ("glicko2", "info_gain"),
]
CARD_SERIES = {"info_gain": "solid", "random": "dashed", "closest": "dots"}
# Glickman (2013), "Example of the Glicko-2 system", final step.
PAPER_EXAMPLE = {"rating": 1464.06, "rd": 151.52, "volatility": 0.05999}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def glickman_example() -> dict[str, Any]:
    games = [(Rating(1400, 30), 1.0), (Rating(1550, 100), 0.0), (Rating(1700, 300), 0.0)]
    out = rate(Rating(1500, 200, 0.06), games, tau=0.5)
    return {
        "paper": PAPER_EXAMPLE,
        "computed": {"rating": out.rating, "rd": out.rd, "volatility": out.volatility},
    }


def run(args: tuple[str, str, Config, int]) -> dict[str, Any]:
    system, strategy, config, seeds = args
    return run_condition(system, strategy, config, range(seeds))


def card(results: dict[str, Any], command: str) -> dict[str, Any]:
    meta = results["meta"]
    by_strategy = {c["strategy"]: c for c in results["strategies"]}
    active = by_strategy["info_gain"]["reach"]["mean_curve_crossing"]
    baseline = by_strategy["random"]["reach"]["mean_curve_crossing"]
    if active is None or baseline is None:
        sys.exit("mean tau never reached the target; raise --budget before writing card.json")
    saved = baseline - active
    x = meta["checkpoints"]
    return {
        "schema_version": 1,
        "repo": "pairwise-ratings",
        "sha": meta["sha"],
        "generated_at": meta["generated_at"],
        "command": command,
        "data_note": (
            f"Seeded simulation with synthetic players: seeds 0-{meta['seeds'] - 1}, "
            f"{meta['n_players']} players, {meta['budget']} comparisons per run"
        ),
        "metric": {
            "label": f"Comparisons saved to mean Kendall tau {TAU_TARGET}, info-gain vs random",
            "value": saved,
            "unit": "comparisons",
            "higher_is_better": True,
        },
        "takeaway": (
            f"Info-gain pairing reaches mean Kendall tau {TAU_TARGET} after {active} comparisons; "
            f"random pairing needs {baseline}."
        ),
        "chart": {
            "type": "line",
            "x_label": "Comparisons",
            "x_unit": "count",
            "x_scale": "linear",
            "y_label": "Kendall tau vs true skill",
            "y_unit": "unitless",
            "y_scale": "linear",
            "series": [
                {
                    "name": name.replace("_", "-"),
                    "style": style,
                    "points": [
                        [xi, yi] for xi, yi in zip(x, by_strategy[name]["tau"]["mean"], strict=True)
                    ],
                }
                for name, style in CARD_SERIES.items()
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--budget", type=int, default=Config.budget)
    parser.add_argument("--players", type=int, default=Config.n_players)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--card", type=Path, default=ROOT / "site" / "card.json")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run with uncommitted code changes"
    )
    args = parser.parse_args()

    dirty = git("status", "--porcelain", "--", "src", "scripts", "pyproject.toml", "uv.lock")
    if dirty and not args.allow_dirty:
        sys.exit(f"uncommitted code changes; commit first so the recorded sha is honest:\n{dirty}")
    config = Config(n_players=args.players, budget=args.budget)
    conditions = list(dict.fromkeys(SYSTEM_CONDITIONS + STRATEGY_CONDITIONS))
    with ProcessPoolExecutor() as pool:
        summaries = list(pool.map(run, [(s, st, config, args.seeds) for s, st in conditions]))
    by_key = {(c["system"], c["strategy"]): c for c in summaries}
    command = shlex.join(["uv", "run", "python", "scripts/run_experiments.py", *sys.argv[1:]])
    results = {
        "meta": {
            "sha": git("rev-parse", "HEAD"),
            "dirty": bool(dirty),
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": command,
            "data": "simulated",
            "seeds": args.seeds,
            "n_players": config.n_players,
            "budget": config.budget,
            "skill_sd": config.skill_sd,
            "checkpoints": list(
                range(config.checkpoint_every, config.budget + 1, config.checkpoint_every)
            ),
            "tau_target": TAU_TARGET,
        },
        "glickman_example": glickman_example(),
        "systems": [by_key[k] for k in SYSTEM_CONDITIONS],
        "strategies": [by_key[k] for k in STRATEGY_CONDITIONS],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    args.card.parent.mkdir(parents=True, exist_ok=True)
    args.card.write_text(json.dumps(card(results, command), indent=1) + "\n")
    print(f"wrote {args.out / 'results.json'} and {args.card}")


if __name__ == "__main__":
    main()

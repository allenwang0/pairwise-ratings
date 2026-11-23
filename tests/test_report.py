"""The published report, README block and portfolio card must say what results.json says.

Expected values are recomputed here from results.json with numpy, independently of
build_report.py's helpers, so a generator bug (reached-only medians instead of counting runs that
never get there, the mean-curve crossing instead of per-run times, a stale page) fails a test.
"""

import html
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "tests" / "card.schema.json").read_text())
RESULTS = ROOT / "results" / "results.json"


def check_card(card: dict[str, object]) -> None:
    jsonschema.validate(card, SCHEMA)
    chart = card["chart"]
    assert isinstance(chart, dict)
    for s in chart["series"]:
        assert all(math.isfinite(v) for point in s["points"] for v in point)
    assert (ROOT / "site" / "card.json").stat().st_size <= 200_000


def text_of(page: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", page)).replace(" ", " ")


def build(tmp: Path, results: Path, readme: Path | None = None) -> str:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_report.py"),
        "--results",
        str(results),
        "--out",
        str(tmp / "index.html"),
    ]
    if readme:
        cmd += ["--readme", str(readme)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    return (tmp / "index.html").read_text()


def censored_median(per_seed: list[int | None]) -> float:
    return float(np.median([np.inf if v is None else v for v in per_seed]))


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("run")
    run = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiments.py"),
        "--seeds",
        "4",
        "--players",
        "8",
        "--budget",
        "300",
        "--out",
        str(out),
        "--card",
        str(out / "card.json"),
        "--allow-dirty",
    ]
    subprocess.run(run, check=True, cwd=ROOT)
    build(out, out / "results.json")
    return out


@pytest.fixture(scope="module")
def committed(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    tmp = tmp_path_factory.mktemp("page")
    readme = tmp / "README.md"
    readme.write_text((ROOT / "README.md").read_text())
    page = build(tmp, RESULTS, readme)
    return page, readme.read_text()


def test_quick_run_writes_a_valid_card_and_page(quick_run: Path) -> None:
    check_card(json.loads((quick_run / "card.json").read_text()))
    page = (quick_run / "index.html").read_text()
    assert not re.search(r'="[^"]*\b(NaN|nan|inf)\b[^"]*"', page)
    # One self-contained file: no external scripts, styles or images are loaded.
    assert not re.search(
        r'<(script|img|link)[^>]+(src|href)="https?://', page.replace('<link rel="canonical"', "")
    )


def test_hero_is_bradley_terry_final_tau(committed: tuple[str, str]) -> None:
    # Owner rule: the hero is a positive measured result, and it must be the committed one.
    page, _ = committed
    results = json.loads(RESULTS.read_text())
    tau = {c["system"]: float(np.mean(c["final_tau_per_seed"])) for c in results["systems"]}
    assert max(tau, key=tau.__getitem__) == "bradley_terry"
    hero = re.search(r'<p class="hero-metric"><span class="value">([^<]+)</span>', page)
    assert hero and hero.group(1) == f"{tau['bradley_terry']:.3f}"
    text = text_of(page)
    assert f"vs {tau['glicko2']:.3f} for Glicko-2 and {tau['elo']:.3f} for Elo" in text
    assert "Simulation, not real players" in text, "simulated data must be labelled in the hero"


def test_time_to_target_counts_runs_that_never_get_there(committed: tuple[str, str]) -> None:
    # Medians over reached runs only would flatter rules that leave many runs stuck.
    page, _ = committed
    results = json.loads(RESULTS.read_text())
    reach = {c["strategy"]: c["reach"] for c in results["strategies"]}
    ig, rnd = (
        censored_median(reach["info_gain"]["per_seed"]),
        censored_median(reach["random"]["per_seed"]),
    )
    text = text_of(page)
    assert f"Information gain: median {ig:,.0f}" in text
    assert f"Random: median {rnd:,.0f}" in text
    for name, r in reach.items():
        assert r["never"] == sum(v is None for v in r["per_seed"]), name
    saved = rnd - ig
    assert f"saves a median {saved:,.0f} comparisons".replace("-", "−") in text
    ci = re.search(r"95% bootstrap interval (\S+) to (\S+?)[;)]", text)
    assert ci, "bootstrap interval missing"
    lo, hi = (float(v.replace("−", "-").replace(",", "")) for v in ci.groups())
    assert lo <= saved <= hi
    assert ("includes zero" in text) == (lo <= 0 <= hi), (
        "the page must say when the interval includes zero"
    )


def test_old_unstable_headline_is_gone(committed: tuple[str, str]) -> None:
    page, readme = committed
    for text in (text_of(page), readme, (ROOT / "README.md").read_text()):
        assert "1,010" not in text and "27%" not in text and "270 fewer" not in text


def test_committed_page_readme_and_card_are_current(committed: tuple[str, str]) -> None:
    page, readme = committed
    assert page == (ROOT / "site" / "index.html").read_text(), (
        "site/index.html is stale: run make report"
    )
    assert readme == (ROOT / "README.md").read_text(), (
        "README highlights are stale: run make report"
    )
    card = json.loads((ROOT / "site" / "card.json").read_text())
    results = json.loads(RESULTS.read_text())
    check_card(card)
    assert card["sha"] == results["meta"]["sha"]
    assert not results["meta"]["dirty"]
    bt = next(c for c in results["systems"] if c["system"] == "bradley_terry")
    assert card["metric"]["value"] == bt["final_tau"]


def test_evidence_ids_resolve(committed: tuple[str, str]) -> None:
    page, _ = committed
    cited = set(re.findall(r'href="#evidence-(e\d+)"', page))
    rows = set(re.findall(r'id="evidence-(e\d+)"', page))
    assert cited and cited <= rows


def test_glickman_example_matches_paper() -> None:
    ex = json.loads(RESULTS.read_text())["glickman_example"]
    assert abs(ex["computed"]["rating"] - ex["paper"]["rating"]) < 0.02
    assert abs(ex["computed"]["rd"] - ex["paper"]["rd"]) < 0.02


def test_missing_results_fails_loudly(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_report.py"),
            "--results",
            str(tmp_path / "none.json"),
            "--out",
            str(tmp_path / "x.html"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "make results" in proc.stderr

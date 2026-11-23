import json
import math
import re
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "tests" / "card.schema.json").read_text())


def check_card(card: dict[str, object]) -> None:
    jsonschema.validate(card, SCHEMA)
    chart = card["chart"]
    assert isinstance(chart, dict)
    for s in chart["series"]:
        assert all(math.isfinite(v) for point in s["points"] for v in point)
    assert (ROOT / "site" / "card.json").stat().st_size <= 200_000


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("run")
    script = ROOT / "scripts"
    run = [
        sys.executable,
        str(script / "run_experiments.py"),
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
    subprocess.run(
        [
            sys.executable,
            str(script / "build_report.py"),
            "--results",
            str(out / "results.json"),
            "--out",
            str(out / "index.html"),
        ],
        check=True,
        cwd=ROOT,
    )
    return out


def test_quick_run_writes_a_valid_card(quick_run: Path) -> None:
    check_card(json.loads((quick_run / "card.json").read_text()))


def test_report_has_every_view_and_no_broken_numbers(quick_run: Path) -> None:
    page = (quick_run / "index.html").read_text()
    for section in ("systems", "strategies", "reach", "calibration", "glickman"):
        assert re.search(rf'<section [^>]*id="{section}"', page)
    assert page.count('takeaway">') == 5
    assert "simulated" in page
    assert not re.search(r'="[^"]*\b(NaN|nan|inf|Infinity)\b[^"]*"', page)
    # Only inline assets: the page must work offline and leak no requests. Plain links (back to
    # the portfolio, to the code) are allowed; anything the browser would fetch is not.
    assert not re.search(r'\bsrc="(https?:)?//', page)
    assert not re.search(r'<link[^>]*href="(https?:)?//', page)
    assert "url(http" not in page and "fetch(" not in page


def test_committed_card_matches_committed_results() -> None:
    # The portfolio site shows card.json as-is, so its numbers must be the ones in results.json.
    card = json.loads((ROOT / "site" / "card.json").read_text())
    results = json.loads((ROOT / "results" / "results.json").read_text())
    check_card(card)
    reach = {c["strategy"]: c["reach"]["mean_curve_crossing"] for c in results["strategies"]}
    assert card["metric"]["value"] == reach["random"] - reach["info_gain"]
    assert card["sha"] == results["meta"]["sha"]
    assert not results["meta"]["dirty"]


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


def test_readme_numbers_match_results() -> None:
    # The README headline numbers are hand-written; this keeps them tied to the committed run.
    readme = (ROOT / "README.md").read_text()
    results = json.loads((ROOT / "results" / "results.json").read_text())
    reach = {c["strategy"]: c["reach"]["mean_curve_crossing"] for c in results["strategies"]}
    info_gain, random = reach["info_gain"], reach["random"]
    strat = {c["strategy"]: c["reach"] for c in results["strategies"]}
    bt = next(c["reach"] for c in results["systems"] if c["system"] == "bradley_terry")
    assert (
        f"**{info_gain:,} comparisons with information-gain pairing instead of {random:,}**"
        in readme
    )
    saved = random - info_gain
    assert f"{saved:,} ({saved / random:.0%}".replace("%", " %") in readme
    # The headline is an average-curve crossing; the README must also give the typical run.
    med_ig, med_rnd = strat["info_gain"]["median"], strat["random"]["median"]
    typical = (med_rnd - med_ig) / med_rnd
    assert f"{med_ig:.0f} vs {med_rnd:.0f} comparisons, about {typical * 100:.0f} %" in readme
    assert f"after {bt['mean_curve_crossing']} comparisons" in readme
    never = f"{strat['uncertainty']['never']} and {strat['closest']['never']} of 100 runs"
    assert never in readme and f"against {strat['random']['never']} for random" in readme
    tau = {c["system"]: c["final_tau"] for c in results["systems"]}
    assert f"Bradley-Terry ranks best (rank agreement {tau['bradley_terry']:.3f}" in readme
    assert f"Elo worst ({tau['elo']:.3f})" in readme
    assert max(tau, key=tau.__getitem__) == "bradley_terry"
    assert min(tau, key=tau.__getitem__) == "elo"

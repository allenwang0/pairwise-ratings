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
        assert f'<section id="{section}">' in page
    assert page.count('class="takeaway"') == 5
    assert "simulated" in page
    assert not re.search(r'="[^"]*(NaN|nan|inf)[^"]*"', page)
    # Only inline assets: the page must work offline and leak no requests.
    assert "http://" not in page and "https://" not in page


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

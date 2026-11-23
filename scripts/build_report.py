"""Build site/index.html (and optionally the README results block) from results/results.json.

Usage:
  python3 scripts/build_report.py [--results results/results.json] [--out site/index.html]
      [--readme README.md]

Every number on the page is computed here from results.json, including the bootstrap interval on
comparisons saved, which resamples the per-seed values the experiment records. Nothing is typed by
hand. Page components and charts come from the vendored report kit in report-kit/. Standard
library only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "report-kit"))
import report_kit as rk  # noqa: E402
from report_kit import Raw, esc  # noqa: E402

REPO = "allenwang0/pairwise-ratings"
REPO_URL = f"https://github.com/{REPO}"
PAGE_URL = "https://allenwang0.github.io/pairwise-ratings/"
HOME_URL = "https://allenwang0.github.io/"
NAMES = {
    "elo": "Elo",
    "glicko2": "Glicko-2",
    "bradley_terry": "Bradley–Terry",
    "random": "Random",
    "uncertainty": "Least-known player",
    "closest": "Most even match",
    "info_gain": "Information gain",
}
SUBJECT_SYSTEM = "bradley_terry"
SUBJECT_STRATEGY = "info_gain"
BOOT_SAMPLES = 10_000
BOOT_SEED = 0
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
THIN = " "


# ---------------------------------------------------------------- statistics


def num(v: float) -> str:
    """Whole number with a true minus sign."""
    return f"{v:,.0f}".replace("-", "\u2212")


def tau3(v: float) -> str:
    return f"{v:.3f}"


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default) on values that may include +inf."""
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if s[hi] == math.inf:
        return math.inf if pos > lo or s[lo] == math.inf else s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def censored(per_seed: list[int | None]) -> list[float]:
    """Comparisons to target per seed; a seed that never reached it counts as +inf (later than
    the budget), so quartiles describe every run, not only the ones that got there."""
    return [math.inf if v is None else float(v) for v in per_seed]


def bootstrap_saved(base: list[float], active: list[float]) -> dict[str, Any]:
    """Median comparisons saved (base median minus active median) with a 95% percentile bootstrap
    interval. Seeds are resampled in pairs: seed s faces the same players under both rules."""
    rng = random.Random(BOOT_SEED)
    n = len(base)
    point = statistics.median(base) - statistics.median(active)
    draws = []
    skipped = 0
    for _ in range(BOOT_SAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        d = statistics.median([base[i] for i in idx]) - statistics.median([active[i] for i in idx])
        if math.isfinite(d):
            draws.append(d)
        else:
            skipped += 1
    draws.sort()
    return {
        "point": point,
        "lo": quantile(draws, 0.025) if draws else math.nan,
        "hi": quantile(draws, 0.975) if draws else math.nan,
        "share_positive": sum(d > 0 for d in draws) / len(draws) if draws else math.nan,
        "samples": BOOT_SAMPLES,
        "skipped": skipped,
    }


def derive(results: dict[str, Any]) -> dict[str, Any]:
    meta = results["meta"]
    systems = {c["system"]: c for c in results["systems"]}
    strategies = {c["strategy"]: c for c in results["strategies"]}
    x = meta["checkpoints"]
    target = meta["tau_target"]
    ranked = sorted(systems, key=lambda k: -systems[k]["final_tau"])
    # Does the subject system have the highest mean tau at every checkpoint?
    lead_all = all(
        systems[SUBJECT_SYSTEM]["tau"]["mean"][i]
        >= max(c["tau"]["mean"][i] for c in systems.values())
        for i in range(len(x))
    )
    first_lead = next(
        (
            x[i]
            for i in range(len(x))
            if all(
                systems[SUBJECT_SYSTEM]["tau"]["mean"][j]
                >= max(c["tau"]["mean"][j] for c in systems.values())
                for j in range(i, len(x))
            )
        ),
        None,
    )
    reach = {
        k: censored(c["reach"]["per_seed"])
        for k, c in strategies.items()
        if "per_seed" in c["reach"]
    }
    boxes = {
        k: {
            "q1": quantile(v, 0.25),
            "median": quantile(v, 0.5),
            "q3": quantile(v, 0.75),
            "never": sum(math.isinf(t) for t in v),
            "n": len(v),
        }
        for k, v in reach.items()
    }
    boot = bootstrap_saved(reach["random"], reach[SUBJECT_STRATEGY]) if reach else None
    # How flat is the random-pairing mean curve near the target? (why a mean-curve crossing moves)
    rnd = strategies["random"]["tau"]["mean"]
    near = [i for i, v in enumerate(rnd) if abs(v - target) <= 0.005]
    flat = (
        {
            "from": x[near[0]],
            "to": x[near[-1]],
            "lo": min(rnd[near[0] : near[-1] + 1]),
            "hi": max(rnd[near[0] : near[-1] + 1]),
        }
        if near
        else None
    )
    ex = results["glickman_example"]
    return {
        "meta": meta,
        "systems": systems,
        "strategies": strategies,
        "x": x,
        "target": target,
        "ranked": ranked,
        "lead_all": lead_all,
        "first_lead": first_lead,
        "reach": reach,
        "boxes": boxes,
        "boot": boot,
        "flat": flat,
        "ex": ex,
        "ex_gap": max(abs(ex["paper"][k] - ex["computed"][k]) for k in ("rating", "rd")),
        "crossing": {k: c["reach"]["mean_curve_crossing"] for k, c in strategies.items()},
    }


def glickman_inputs() -> dict[str, Any]:
    """Inputs of the worked example, read from the script that runs it (never retyped here)."""
    src = (ROOT / "scripts" / "run_experiments.py").read_text(encoding="utf-8")
    body = src[src.index("def glickman_example") :]
    body = body[: body.index("return")]
    opponents = re.findall(r"Rating\((\d+), \d+\), [\d.]+\)", body)
    player = re.search(r"rate\(Rating\((\d+), (\d+), ([\d.]+)\), games, tau=([\d.]+)\)", body)
    if not player or len(opponents) != 3:
        raise ValueError("glickman_example() inputs not found in scripts/run_experiments.py")
    return {"player": player.groups()[:3], "tau": player.group(4), "opponents": opponents}


def z95() -> str:
    """The band multiplier used by the experiment, read from its source."""
    src = (ROOT / "src" / "pairwise_ratings" / "experiment.py").read_text(encoding="utf-8")
    match = re.search(r"^Z95 = ([\d.]+)$", src, re.M)
    if not match:
        raise ValueError("Z95 not found in src/pairwise_ratings/experiment.py")
    return match.group(1)


def git_last_commit(rel: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def line_of(rel: str, needle: str, end: str | None = None) -> tuple[int, int]:
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    first = next(i for i, line in enumerate(lines, 1) if needle in line)
    if end is None:
        return first, first
    return first, next(i for i, line in enumerate(lines, 1) if i >= first and end in line)


# ---------------------------------------------------------------- charts


def tau_chart(
    d: dict[str, Any], uid: str, width: int, height: int, keys: list[str], step: int = 1
) -> Any:
    """Mean Kendall tau by comparisons with the 95% band of the mean; one accent series."""
    # step > 1 thins the checkpoints (always keeping the last) for small variants
    idx = sorted(set(range(step - 1, len(d["x"]), step)) | {len(d["x"]) - 1})
    x = [d["x"][i] for i in idx]
    series = [
        (k, {b: [d["systems"][k]["tau"][b][i] for i in idx] for b in ("mean", "lo", "hi")})
        for k in keys
    ]
    narrow = width < 420
    label_w = max(rk.text_width(NAMES[k], 12 if narrow else 13) for k in keys) + 14
    ml, mr, mt, mb = 40, label_w, 30, 46
    xmax = x[-1]
    xt = rk.nice_ticks(0, xmax, 4 if narrow else 8)
    xt = [t for t in xt if t <= xmax]
    yt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def px(v: float) -> float:
        return float(ml + (width - ml - mr) * v / xmax)

    def py(v: float) -> float:
        return float(height - mb - (height - mt - mb) * v)

    parts = []
    for t in yt:
        parts.append(f'<path class="c-grid" d="M{ml} {rk.coord(py(t))}H{rk.coord(width - mr)}"/>')
        parts.append(rk.svg_text(ml - 8, py(t) + 4, f"{t:.1f}", "c-tick", "end"))
    for t in xt:
        parts.append(rk.svg_text(px(t), height - mb + 18, f"{t:,.0f}", "c-tick", "middle"))
    parts.append(f'<path class="c-axis" d="M{ml} {rk.coord(height - mb)}H{rk.coord(width - mr)}"/>')
    ends: list[tuple[float, str, bool]] = []
    for k, band in sorted(series, key=lambda s: s[0] == SUBJECT_SYSTEM):
        accent = k == SUBJECT_SYSTEM
        top = "".join(
            ("M" if i == 0 else "L") + f"{rk.coord(px(a))} {rk.coord(py(b))}"
            for i, (a, b) in enumerate(zip(x, band["hi"], strict=True))
        )
        bot = "".join(
            f"L{rk.coord(px(a))} {rk.coord(py(b))}"
            for a, b in reversed(list(zip(x, band["lo"], strict=True)))
        )
        fill = "var(--accent)" if accent else "var(--data-muted)"
        parts.append(
            f'<path d="{top}{bot}Z" style="fill:{fill};fill-opacity:{0.2 if accent else 0.3}"/>'
        )
        line = "".join(
            ("M" if i == 0 else "L") + f"{rk.coord(px(a))} {rk.coord(py(b))}"
            for i, (a, b) in enumerate(zip(x, band["mean"], strict=True))
        )
        cls = "c-line-accent" if accent else "c-stroke-muted"
        parts.append(f'<path class="{cls}" stroke-width="{2 if accent else 1.5}" d="{line}"/>')
        ends.append((py(band["mean"][-1]) + 4, k, accent))
    ends.sort(key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 15)
    for y, (_, k, accent) in zip(ys, ends, strict=True):
        parts.append(
            rk.svg_text(px(xmax) + 8, y, NAMES[k], "c-label c-hi" if accent else "c-label")
        )
    parts.append(
        rk.svg_text(
            ml,
            14,
            "Kendall τ vs true skill (mean, 95% band)"
            if not narrow
            else "Kendall τ (mean, 95% band)",
            "c-title",
        )
    )
    parts.append(rk.svg_text(width - mr, height - 8, "Comparisons played", "c-title", "end"))
    q = len(d["x"]) // 4 - 1  # a quarter of the budget, from the full series
    desc = "; ".join(
        f"{NAMES[k]}: τ {tau3(d['systems'][k]['tau']['mean'][q])} at {d['x'][q]}, "
        f"{tau3(d['systems'][k]['tau']['mean'][-1])} at {d['x'][-1]}"
        for k in keys
    )
    return rk.svg(
        uid,
        width,
        height,
        "Mean Kendall τ by comparisons, random pairs",
        desc + ".",
        "".join(parts),
    )


def box_chart(d: dict[str, Any], uid: str, width: int) -> Any:
    """Time to reach the target per strategy: one dot per run, box q1-q3, median tick; runs that
    never reach it are counted in a shaded column right of the budget."""
    budget = d["meta"]["budget"]
    order = [k for k in ("random", "info_gain", "closest", "uncertainty") if k in d["boxes"]]
    narrow = width < 420
    name_w = 0 if narrow else max(rk.text_width(NAMES[k]) for k in order) + 16
    never_w = 78 if not narrow else 58
    plot_l, plot_r = name_w or 12, width - never_w - 8
    row_h = 58 if narrow else 44
    top = 26
    xt = [t for t in rk.nice_ticks(0, budget, 4 if narrow else 8) if t <= budget]

    def x(v: float) -> float:
        return float(plot_l + (plot_r - plot_l) * v / budget)

    bottom = top + row_h * len(order)
    parts = [
        f'<rect class="c-band" x="{rk.coord(plot_r + 8)}" y="{top - 8}" width="{never_w}" '
        f'height="{rk.coord(bottom - top + 8)}" rx="6"/>',
        rk.svg_text(plot_r + 8 + never_w / 2, top - 12, "never", "c-title", "middle"),
    ]
    for t in xt:
        parts.append(f'<path class="c-grid" d="M{rk.coord(x(t))} {top - 8}V{rk.coord(bottom)}"/>')
        parts.append(rk.svg_text(x(t), bottom + 16, f"{t:,.0f}", "c-tick", "middle"))
    rng = random.Random(1)  # deterministic vertical jitter for the per-run dots
    for i, k in enumerate(order):
        b = d["boxes"][k]
        accent = k == SUBJECT_STRATEGY
        cy = top + i * row_h + (row_h - 14 if narrow else row_h / 2)
        if narrow:
            parts.append(rk.svg_text(0, cy - 17, NAMES[k], "c-label c-hi" if accent else "c-label"))
        else:
            parts.append(
                rk.svg_text(
                    name_w - 12, cy + 4.5, NAMES[k], "c-label c-hi" if accent else "c-label", "end"
                )
            )
        parts.append(f'<g data-row="{esc(k)}">')
        # one dot per run, drawn as zero-length round-capped strokes in one path (small page)
        dots = "".join(
            f"M{rk.coord(x(v))} {rk.coord(cy + rng.uniform(-7, 7))}h0"
            for v in d["reach"][k]
            if math.isfinite(v)
        )
        parts.append(
            f'<path d="{dots}" style="stroke:var(--data-muted);stroke-width:3.6;'
            'stroke-linecap:round;stroke-opacity:.6;fill:none"/>'
        )
        q3 = b["q3"] if math.isfinite(b["q3"]) else budget
        bw = max(x(q3) - x(b["q1"]), 1)
        stroke = "c-stroke-accent" if accent else "c-stroke-muted"
        dash = "" if math.isfinite(b["q3"]) else ' stroke-dasharray="3 2"'
        parts.append(
            f'<rect x="{rk.coord(x(b["q1"]))}" y="{rk.coord(cy - 10)}" '
            f'width="{rk.coord(bw)}" height="20" rx="3" class="{stroke}" stroke-width="1.5"{dash}/>'
        )
        if math.isfinite(b["median"]):
            parts.append(
                f'<path class="{stroke}" stroke-width="3" '
                f'd="M{rk.coord(x(b["median"]))} {rk.coord(cy - 10)}V{rk.coord(cy + 10)}"/>'
            )
        parts.append(
            rk.svg_text(
                plot_r + 8 + never_w / 2,
                cy + 4.5,
                f"{b['never']}/{b['n']}",
                "c-value" if accent else "c-label",
                "middle",
            )
        )
        parts.append("</g>")
    if not narrow:  # on a phone the labels sit above the rows, where an axis line would cross them
        parts.append(f'<path class="c-axis" d="M{rk.coord(plot_l)} {top - 8}V{rk.coord(bottom)}"/>')
    parts.append(
        rk.svg_text(
            plot_r,
            bottom + 36,
            f"Comparisons until Kendall τ ≥ {d['target']} (one dot per run)"
            if not narrow
            else f"Comparisons to τ ≥ {d['target']}",
            "c-title",
            "end",
        )
    )
    desc = "; ".join(
        f"{NAMES[k]}: median {fmt_reach(d['boxes'][k]['median'])}, quartiles "
        f"{fmt_reach(d['boxes'][k]['q1'])} to "
        f"{fmt_reach(d['boxes'][k]['q3'])}, never reached in {d['boxes'][k]['never']} of "
        f"{d['boxes'][k]['n']} runs"
        for k in order
    )
    return rk.svg(
        uid,
        width,
        bottom + 44,
        f"Comparisons needed to reach Kendall τ {d['target']}, by pair rule",
        desc + ".",
        "".join(parts),
    )


def fmt_reach(v: float) -> str:
    return "not reached" if math.isinf(v) else f"{v:,.0f}"


# ---------------------------------------------------------------- README


def readme_block(d: dict[str, Any]) -> str:
    m, s = d["meta"], d["systems"]
    boot, box = d["boot"], d["boxes"]
    lines = [
        README_START,
        "- **Bradley–Terry recovers the true ranking best**: Kendall τ "
        f"{tau3(s['bradley_terry']['final_tau'])} "
        f"after {m['budget']:,} random comparisons, vs {tau3(s['glicko2']['final_tau'])} for "
        "Glicko-2 and "
        f"{tau3(s['elo']['final_tau'])} for Elo (mean of {m['seeds']} simulated runs of "
        f"{m['n_players']} players; "
        f"outcomes come from the Bradley–Terry model itself).",
        (
            "- **Choosing pairs by information gain ends more accurate but not detectably "
            "sooner**: "
            if boot["lo"] <= 0 <= boot["hi"]
            else "- **Choosing pairs by information gain reaches the target sooner**: "
        )
        + f"a median of "
        f"{fmt_reach(box[SUBJECT_STRATEGY]['median'])} comparisons to reach τ {d['target']} vs "
        f"{fmt_reach(box['random']['median'])} for random pairs (95% bootstrap interval for the "
        "saving: "
        f"{num(boot['lo'])} to {num(boot['hi'])}); {box[SUBJECT_STRATEGY]['never']} vs "
        f"{box['random']['never']} "
        f"of {box['random']['n']} runs never get there, and final τ is "
        f"{tau3(d['strategies'][SUBJECT_STRATEGY]['final_tau'])} vs "
        f"{tau3(d['strategies']['random']['final_tau'])} "
        f"(Glicko-2 ratings).",
        f"- **The intuitive shortcuts stall**: pairing the least-known player leaves "
        f"{box['uncertainty']['never']} of {box['uncertainty']['n']} runs short of the target, the "
        "most even "
        f"match {box['closest']['never']}.",
        "- **Glicko-2 reproduces Glickman's worked example**: rating "
        f"{d['ex']['computed']['rating']:.2f} vs "
        f"{d['ex']['paper']['rating']:.2f} in the paper.",
        README_END,
    ]
    return "\n".join(lines)


def update_readme(path: Path, d: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        sys.exit(f"{path} has no {README_START} ... {README_END} block")
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    path.write_text(head + readme_block(d) + tail, encoding="utf-8")


# ---------------------------------------------------------------- page


def evidence_rows(
    d: dict[str, Any], src_sha: str, res_sha: str
) -> list[tuple[str, str, str, str, str]]:
    def blob(rel: str, sha: str, span: tuple[int, int] | None = None) -> str:
        anchor = ""
        if span:
            anchor = f"#L{span[0]}" + (f"-L{span[1]}" if span[1] != span[0] else "")
        return f"{REPO_URL}/blob/{sha}/{rel}{anchor}"

    m, s, box, boot = d["meta"], d["systems"], d["boxes"], d["boot"]
    res_href = blob("results/results.json", res_sha)
    sim = line_of("src/pairwise_ratings/simulate.py", "def simulate(", "return Run(")
    reach = line_of(
        "src/pairwise_ratings/experiment.py",
        "def comparisons_to_target",
        'out["mean_curve_crossing"]',
    )
    ig = line_of("src/pairwise_ratings/selection.py", "def info_gain_scores", "return out")
    gl = line_of("tests/test_glicko2.py", "def test_", "")
    return [
        (
            "E1",
            f"Final mean Kendall τ: Bradley–Terry {tau3(s['bradley_terry']['final_tau'])}, "
            "Glicko-2 "
            f"{tau3(s['glicko2']['final_tau'])}, Elo {tau3(s['elo']['final_tau'])}; {m['seeds']} "
            "runs × "
            f"{m['n_players']} players × {m['budget']:,} random comparisons",
            "results/results.json (systems[].final_tau)",
            res_href,
            "Simulated",
        ),
        (
            "E2",
            "Outcomes drawn from the Bradley–Terry model with skills from N(0, "
            f"{m['skill_sd']:g}); every condition with the same seed faces the same players",
            f"src/pairwise_ratings/simulate.py:{sim[0]}–{sim[1]}",
            blob("src/pairwise_ratings/simulate.py", src_sha, sim),
            "Code",
        ),
        (
            "E3",
            f"Median comparisons to τ {d['target']}: information gain "
            f"{fmt_reach(box[SUBJECT_STRATEGY]['median'])}, "
            f"random {fmt_reach(box['random']['median'])} (never-reached runs counted as later "
            "than the budget)",
            "results/results.json (strategies[].reach.per_seed); "
            f"src/pairwise_ratings/experiment.py:{reach[0]}–{reach[1]}",
            res_href,
            "Derived",
        ),
        (
            "E4",
            f"Saving {num(boot['point'])} comparisons, 95% bootstrap interval {num(boot['lo'])} to "
            f"{num(boot['hi'])} "
            f"({boot['samples']:,} paired resamples of seeds)",
            "scripts/build_report.py bootstrap_saved()",
            blob(
                "scripts/build_report.py",
                src_sha,
                line_of("scripts/build_report.py", "def bootstrap_saved", "return {"),
            ),
            "Derived",
        ),
        (
            "E5",
            "Never reached τ target: "
            + ", ".join(f"{NAMES[k]} {b['never']}/{b['n']}" for k, b in box.items()),
            "results/results.json (strategies[].reach.never)",
            res_href,
            "Simulated",
        ),
        (
            "E6",
            f"Glicko-2 rating {d['ex']['computed']['rating']:.2f} vs "
            f"{d['ex']['paper']['rating']:.2f} and RD "
            f"{d['ex']['computed']['rd']:.2f} vs {d['ex']['paper']['rd']:.2f} in Glickman's worked "
            "example",
            f"results/results.json (glickman_example); tests/test_glicko2.py:{gl[0]}",
            blob("tests/test_glicko2.py", src_sha, gl),
            "Test",
        ),
        (
            "E7",
            "Information gain = predicted drop in Glicko-2 posterior variance, known before the "
            "match",
            f"src/pairwise_ratings/selection.py:{ig[0]}–{ig[1]}",
            blob("src/pairwise_ratings/selection.py", src_sha, ig),
            "Code",
        ),
    ]


def prose(content: Any) -> Any:
    """Body paragraph held to the reading measure."""
    return rk.tag("p", content, class_="prose")


def toc(items: list[tuple[str, str]]) -> Any:
    """Tier B inline contents list: the kit's nav style, not sticky."""
    lis = rk.join(Raw(f"<li>{rk.tag('a', label, href='#' + sid)}</li>") for sid, label in items)
    return Raw(
        '<nav class="report-nav" aria-label="Contents" style="position:static">'
        f"<ul>{lis}</ul></nav>"
    )


def build(results: dict[str, Any], res_sha: str) -> str:
    d = derive(results)
    m, s, box, boot = d["meta"], d["systems"], d["boxes"], d["boot"]
    src_sha = m["sha"]
    ev = {r[0]: r for r in evidence_rows(d, src_sha, res_sha)}
    R = rk.ref
    res_href = ev["E1"][3]
    bt, gl, elo = (s[k]["final_tau"] for k in ("bradley_terry", "glicko2", "elo"))
    n_runs = f"{m['seeds']} simulated runs"
    sim_ctx = (
        f"Simulation, not real players: {m['n_players']} players with hidden skills drawn from "
        f"N(0, {m['skill_sd']:g}), {m['budget']:,} comparisons per run, outcomes generated by the "
        f"Bradley–Terry model itself; seeds 0–{m['seeds'] - 1}."
    )

    hero = rk.hero(
        eyebrow="Statistics · Python · Simulation study",
        title="pairwise-ratings",
        subtitle=Raw(
            "When all you can ask is “which of these two is better?”, how fast does a rating "
            "system "
            "recover the true ranking — and which pair should you ask about next? Elo, Glicko-2 "
            "and "
            "Bradley–Terry, with four rules for picking pairs."
        ),
        value=tau3(bt),
        label=Raw(
            f"Kendall τ for Bradley–Terry after {m['budget']:,} random comparisons, the highest of "
            "the "
            f"three systems{R('E1')}"
        ),
        support=Raw(
            f"vs {tau3(gl)} for Glicko-2 and {tau3(elo)} for Elo; mean of {n_runs} (1 = perfect "
            f"ranking){R('E1')}"
        ),
        context=sim_ctx,
        actions=[("Source", REPO_URL, True), ("Run the simulation", "#reproduce", False)],
        figure=rk.join(
            [
                rk.tag(
                    "p",
                    f"Ranking accuracy as comparisons accumulate, {n_runs}",
                    class_="card-label",
                ),
                rk.variants(
                    tau_chart(d, "hero-d", 398, 300, ["elo", "glicko2", "bradley_terry"], 2),
                    tau_chart(d, "hero-t", 640, 320, ["elo", "glicko2", "bradley_terry"], 2),
                    tau_chart(d, "hero-p", 300, 260, ["elo", "glicko2", "bradley_terry"], 4),
                ),
            ]
        ),
    )

    nav = toc(
        [
            ("findings", "Findings"),
            ("setup", "Setup"),
            ("systems", "Systems"),
            ("pairs", "Pair rules"),
            ("calibration", "Calibration"),
            ("glicko", "Correctness"),
            ("methodology", "Method"),
            ("limitations", "Limits"),
            ("reproduce", "Reproduce"),
            ("evidence", "Evidence"),
        ]
    )

    unclear = boot["lo"] <= 0 <= boot["hi"]
    bt_mae = s["bradley_terry"]["calibration"]["mean_abs_error_vs_true_p"]
    ig_final = d["strategies"][SUBJECT_STRATEGY]["final_tau"]
    rnd_final = d["strategies"]["random"]["final_tau"]
    findings = rk.bento(
        [
            (
                8,
                2,
                rk.join(
                    [
                        rk.tag(
                            "h3",
                            "Information-gain pairing: sooner on median, within noise"
                            if unclear
                            else "Information-gain pairing reaches the target sooner",
                            class_="card-heading",
                        ),
                        rk.comparison(
                            [
                                (
                                    NAMES[SUBJECT_STRATEGY],
                                    box[SUBJECT_STRATEGY]["median"],
                                    f"{fmt_reach(box[SUBJECT_STRATEGY]['median'])} comparisons",
                                    True,
                                ),
                                (
                                    "Random pairs",
                                    box["random"]["median"],
                                    f"{fmt_reach(box['random']['median'])} comparisons",
                                    False,
                                ),
                            ],
                            m["budget"],
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                "Median comparisons until a run first reaches Kendall τ "
                                f"{d['target']}, Glicko-2 "
                                f"ratings.{R('E3')} The saving of {num(boot['point'])} has a 95% "
                                "bootstrap interval of "
                                f"{num(boot['lo'])} to {num(boot['hi'])}.{R('E4')} "
                                '<a href="#pairs">Per-run '
                                f"distribution</a>"
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                1,
                rk.join(
                    [
                        rk.tag("h3", "Glicko-2 matches the paper", class_="card-heading"),
                        rk.metric(
                            "Rating after Glickman's worked example",
                            f"{d['ex']['computed']['rating']:.2f}",
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"The paper prints {d['ex']['paper']['rating']:.2f}; it rounds "
                                "intermediate "
                                f"steps.{R('E6')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                3,
                rk.join(
                    [
                        rk.tag("h3", "Shortcuts leave runs stuck", class_="card-heading"),
                        rk.metric(
                            "Runs never reaching the target, least-known player",
                            f"{box['uncertainty']['never']}/{box['uncertainty']['n']}",
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"Random pairs: {box['random']['never']}/{box['random']['n']}. "
                                "Most even match: "
                                f"{box['closest']['never']}/{box['closest']['n']}.{R('E5')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                4,
                rk.join(
                    [
                        rk.tag(
                            "h3", "Information-gain pairs end more accurate", class_="card-heading"
                        ),
                        rk.metric(
                            "Kendall τ, Glicko-2 with information-gain pairs", tau3(ig_final)
                        ),
                        rk.tag(
                            "p",
                            f"vs {tau3(rnd_final)} with random pairs, after {m['budget']:,} "
                            "comparisons.",
                        ),
                    ]
                ),
            ),
            (
                4,
                5,
                rk.join(
                    [
                        rk.tag("h3", "Elo's probabilities are furthest off", class_="card-heading"),
                        rk.metric(
                            "Mean |predicted − true| win probability, Elo",
                            f"{s['elo']['calibration']['mean_abs_error_vs_true_p']:.3f}",
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                "Bradley–Terry "
                                f"{bt_mae:.3f}. "
                                f'<a href="#calibration">Calibration</a>'
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    findings_sec = rk.section("findings", "What the simulations show", findings, eyebrow="Findings")

    # Setup
    params = rk.table(
        "Simulation parameters, from results.json",
        ["Parameter", "Value"],
        [
            ["Players per run", f"{m['n_players']}"],
            ["Hidden skill", f"N(0, {m['skill_sd']:g}) on the logit scale"],
            ["Outcome model", "P(i beats j) = 1 / (1 + exp(sⱼ − sᵢ))"],
            ["Comparisons per run", f"{m['budget']:,}, accuracy checked every {d['x'][0]}"],
            ["Runs per condition", f"{m['seeds']} (seeds 0–{m['seeds'] - 1})"],
            ["Accuracy target", f"Kendall τ ≥ {d['target']}"],
        ],
    )
    setup_body = rk.join(
        [
            prose(
                Raw(
                    f"Each run draws {m['n_players']} players with hidden skills, then plays "
                    f"{m['budget']:,} simulated matches. A rating system sees only who won, and "
                    "every "
                    f"{d['x'][0]} matches its ranking is scored against the hidden one with "
                    "Kendall τ: the "
                    "share of player pairs in the right order minus the share in the wrong order, "
                    "so 1 is a "
                    f"perfect ranking and 0 is chance.{R('E2')}"
                ),
            ),
            params,
            prose(
                Raw(
                    "Because the same seed gives the same players and outcome stream to every "
                    "condition, "
                    "differences between conditions are paired run by run."
                ),
            ),
        ]
    )
    setup_sec = rk.section(
        "setup",
        "Every run hides a true ranking, then scores how well it is recovered",
        setup_body,
        eyebrow="Setup",
    )

    # Systems
    lead = (
        "Bradley–Terry has the highest mean τ at every checkpoint"
        if d["lead_all"]
        else f"Bradley–Terry leads from {d['first_lead']:,} comparisons on"
        if d["first_lead"]
        else "Mean τ by comparisons"
    )
    fig1 = rk.figure(
        "fig-systems",
        lead,
        rk.variants(
            tau_chart(d, "sys-d", 1000, 400, ["elo", "glicko2", "bradley_terry"]),
            tau_chart(d, "sys-t", 640, 340, ["elo", "glicko2", "bradley_terry"], 2),
            tau_chart(d, "sys-p", 300, 280, ["elo", "glicko2", "bradley_terry"], 4),
        ),
        Raw(
            f"Mean Kendall τ over {n_runs}, random pairs, with the 95% normal band of the mean "
            f"(±{z95()} standard "
            "errors across runs). Simulated outcomes follow the Bradley–Terry model, so its fit "
            "is the "
            "correctly specified one here; that advantage may shrink on real "
            f"data.{R('E1')}{R('E2')}"
        ),
        question="Which rating system ranks players most accurately?",
        table=rk.data_table(
            "Final Kendall τ, random pairs",
            ["System", "Mean τ", "95% band"],
            [
                [
                    NAMES[k],
                    tau3(s[k]["final_tau"]),
                    f"{tau3(s[k]['tau']['lo'][-1])}–{tau3(s[k]['tau']['hi'][-1])}",
                ]
                for k in d["ranked"]
            ],
            numeric={1, 2},
        ),
        data_href=res_href,
    )
    systems_body = rk.join(
        [
            fig1,
            prose(
                Raw(
                    "Elo nudges two ratings after each match by a fixed step. Glicko-2 also tracks "
                    "how "
                    "uncertain each rating is and moves uncertain ratings further. Bradley–Terry "
                    "refits "
                    "every skill by maximum likelihood on all matches so far, so early results are "
                    "not "
                    "forgotten — consistent with its lead, though the simulation also hands it the "
                    "exact "
                    "outcome model."
                ),
            ),
        ]
    )
    systems_sec = rk.section(
        "systems",
        "Bradley–Terry recovers the hidden ranking most accurately",
        systems_body,
        eyebrow="Rating systems",
    )

    # Pair selection
    order = [k for k in ("random", "info_gain", "closest", "uncertainty") if k in box]
    ci_text = f"95% bootstrap interval {num(boot['lo'])} to {num(boot['hi'])}" + (
        "; the interval includes zero" if boot["lo"] <= 0 <= boot["hi"] else ""
    )
    fig2 = rk.figure(
        "fig-pairs",
        f"Information gain saves a median {num(boot['point'])} comparisons ({ci_text})",
        rk.variants(
            box_chart(d, "bx-d", 1000), box_chart(d, "bx-t", 640), box_chart(d, "bx-p", 300)
        ),
        Raw(
            f"Glicko-2 ratings, {n_runs} per rule. Each dot is one run's first checkpoint with τ ≥ "
            f"{d['target']}; boxes span the quartiles and the thick tick is the median, counting "
            "runs that never "
            f"get there as later than {m['budget']:,} (a dashed box edge means the upper quartile "
            "is never "
            f"reached). The interval resamples the {m['seeds']} seeds {boot['samples']:,} times in "
            "pairs, since "
            f"both rules face the same players seed by seed.{R('E3')}{R('E4')}{R('E5')}"
        ),
        question="Does choosing the next pair reach an accurate ranking sooner?",
        table=rk.data_table(
            "Comparisons to reach the target",
            ["Rule", "Q1", "Median", "Q3", "Never reached", "Median, reached runs only"],
            [
                [
                    NAMES[k],
                    fmt_reach(box[k]["q1"]),
                    fmt_reach(box[k]["median"]),
                    fmt_reach(box[k]["q3"]),
                    f"{box[k]['never']}/{box[k]['n']}",
                    fmt_reach(d["strategies"][k]["reach"]["median"] or math.inf),
                ]
                for k in order
            ],
            numeric={1, 2, 3, 4, 5},
        ),
        data_href=res_href,
    )
    flat = d["flat"]
    pair_text = [
        prose(
            Raw(
                "Information gain picks the pair whose result would shrink Glicko-2's total "
                "uncertainty the most; for Glicko-2 that amount is known before the match is "
                f"played.{R('E7')} Pairing the least-known player against a random opponent, or "
                "the two closest-rated players, sounds reasonable but leaves more runs short of "
                "the target than random pairing."
            ),
        ),
    ]
    if flat and d["crossing"].get("random") and d["crossing"].get(SUBJECT_STRATEGY):
        pair_text.append(
            prose(
                Raw(
                    f"Where the mean τ curve first crosses {d['target']} is a fragile summary: "
                    "random pairing's mean "
                    f"stays between {tau3(flat['lo'])} and {tau3(flat['hi'])} from "
                    f"{flat['from']:,} to {flat['to']:,} "
                    "comparisons, so a tiny change in the mean moves that crossing by hundreds of "
                    "comparisons. The "
                    f"per-run medians above do not have that problem."
                ),
            )
        )
    pairs_sec = rk.section(
        "pairs",
        "Information-gain pairing ends more accurate, but not detectably sooner"
        if unclear
        else "Choosing pairs by information gain reaches the target sooner",
        rk.join([fig2, *pair_text]),
        eyebrow="Pair selection",
    )

    # Calibration
    cal_rows = [
        [
            NAMES[k],
            f"{s[k]['calibration']['log_loss']:.3f}",
            f"{s[k]['calibration']['oracle_log_loss']:.3f}",
            f"{s[k]['calibration']['brier']:.3f}",
            f"{s[k]['calibration']['ece']:.3f}",
            f"{s[k]['calibration']['mean_abs_error_vs_true_p']:.3f}",
        ]
        for k in d["ranked"]
    ]
    cal_body = rk.join(
        [
            prose(
                Raw(
                    "Before each match, each system predicts the chance that one player wins. "
                    "Scored over "
                    f"all {s['elo']['calibration']['comparisons']:,} random-pair predictions per "
                    "system; the "
                    "oracle column scores the true simulated probabilities, which no system can "
                    "beat on "
                    f"average."
                ),
            ),
            rk.table(
                "Prediction quality, random pairs (lower is better)",
                ["System", "Log loss", "Oracle log loss", "Brier", "ECE", "Mean |p − true p|"],
                cal_rows,
                numeric={1, 2, 3, 4, 5},
            ),
        ]
    )
    best_mae = min(d["ranked"], key=lambda k: s[k]["calibration"]["mean_abs_error_vs_true_p"])
    cal_sec = rk.section(
        "calibration",
        f"{NAMES[best_mae]}'s win probabilities sit closest to the truth",
        cal_body,
        eyebrow="Calibration",
    )

    # Glicko-2 correctness
    ex = d["ex"]
    ex_rows = [
        [q, f"{ex['paper'][k]:g}", f"{ex['computed'][k]:.{dig}f}"]
        for q, k, dig in (
            ("Rating", "rating", 2),
            ("Rating deviation", "rd", 2),
            ("Volatility", "volatility", 6),
        )
    ]
    gx = glickman_inputs()
    glicko_body = rk.join(
        [
            prose(
                Raw(
                    "Glicko-2 is written from Glickman's paper and checked against its worked "
                    f"example: a {gx['player'][0]}-rated player (RD {gx['player'][1]}, volatility "
                    f"{gx['player'][2]}, system constant {gx['tau']}) beats a {gx['opponents'][0]} "
                    f"and loses to a {gx['opponents'][1]} and a {gx['opponents'][2]}. The code "
                    "matches the printed values to within "
                    f"{d['ex_gap']:.2f} rating "
                    f"points; the paper rounds its intermediate steps.{R('E6')}"
                ),
            ),
            rk.table(
                "Glickman's worked example",
                ["Quantity", "Paper", "This code"],
                ex_rows,
                numeric={1, 2},
            ),
        ]
    )
    glicko_sec = rk.section(
        "glicko",
        "Glicko-2 reproduces Glickman's worked example",
        glicko_body,
        eyebrow="Correctness",
    )

    method = rk.methodology(
        [
            ("Data", "Simulated players and outcomes; no real match data"),
            (
                "Runs",
                f"{m['seeds']} seeds per condition (0–{m['seeds'] - 1}); conditions share players "
                "seed by seed",
            ),
            (
                "Players and budget",
                f"{m['n_players']} players, {m['budget']:,} comparisons, τ checked every "
                f"{d['x'][0]}",
            ),
            ("Accuracy", "Kendall τ-b between estimated scores and hidden skills"),
            ("Bands", "95% normal interval of the mean across runs"),
            (
                "Time to target",
                "first checkpoint with τ ≥ target; runs that never reach it count as later than "
                "the "
                "budget",
            ),
            (
                "Bootstrap",
                f"{boot['samples']:,} paired resamples of seeds, percentile interval, Python "
                "random seed "
                f"{BOOT_SEED}"
                + (
                    f"; {boot['skipped']} resamples with an unreached median skipped"
                    if boot["skipped"]
                    else ""
                ),
            ),
            (
                "Code measured",
                Raw(
                    f'<a class="src" href="{REPO_URL}/tree/{esc(src_sha)}">{esc(src_sha[:7])}</a>'
                    + (" (uncommitted changes)" if m["dirty"] else "")
                ),
            ),
            (
                "Results committed in",
                Raw(f'<a class="src" href="{REPO_URL}/tree/{esc(res_sha)}">{esc(res_sha[:7])}</a>'),
            ),
            ("Command", Raw(f"<code>{esc(m['command'])}</code>")),
            ("Run date", m["generated_at"][:10]),
        ]
    )
    method_sec = rk.section("methodology", "Methodology", method, eyebrow="Conditions")

    limits = rk.limitations(
        [
            (
                "Simulated, correctly specified data.",
                "Outcomes follow the Bradley–Terry model, which favours the "
                "Bradley–Terry fit; real comparisons are noisier.",
            ),
            (
                "Fixed skills.",
                "No player improves or declines, so tracking changing skill is untested.",
            ),
            (
                "One pool size.",
                f"Only {m['n_players']} players; the value of smart pairing depends on pool size.",
            ),
            (
                "Pair rules use Glicko-2 only.",
                "Information gain and the other rules need Glicko-2's deviations; they "
                "were not combined with Bradley–Terry.",
            ),
            (
                "One accuracy target.",
                f"Time-to-target results are for τ {d['target']}; other targets were not run.",
            ),
        ]
    )
    limits_sec = rk.section(
        "limitations",
        "Results hold for simulated players with fixed skills",
        limits,
        eyebrow="Limitations",
    )

    steps_html = rk.reproduce(
        [
            ("Install the pinned dependencies (needs uv and make).", "make setup"),
            (
                "Run the tests, including the Glicko-2 worked example and the report checks.",
                "make test",
            ),
            (
                f"Run every condition over {m['seeds']} seeds and write results/results.json.",
                "make results",
            ),
            ("Rebuild this page and the README block from the results.", "make report"),
        ]
    )
    repro_sec = rk.section(
        "reproduce",
        "Reproduce it in four commands",
        steps_html,
        eyebrow="Reproduce",
        intro=Raw(f'Commands run from a clone of <a class="src" href="{REPO_URL}">{REPO}</a>.'),
    )

    ev_table = rk.table(
        "Evidence ledger",
        ["ID", "Claim", "Source", "Class"],
        [
            [
                Raw(f'<span id="evidence-{eid.lower()}">{eid}</span>'),
                claim,
                Raw(f'<a class="src" href="{esc(href)}">{esc(src)}</a>'),
                cls,
            ]
            for eid, claim, src, href, cls in ev.values()
        ],
        wrap_cols={1, 2},
    )
    ev_sec = rk.section(
        "evidence", "Every number traces to a committed file", ev_table, eyebrow="Sources"
    )

    main = rk.join(
        [
            hero,
            nav,
            findings_sec,
            setup_sec,
            systems_sec,
            pairs_sec,
            cal_sec,
            glicko_sec,
            method_sec,
            limits_sec,
            repro_sec,
            ev_sec,
        ]
    )
    footer = rk.footer(
        Raw(
            "Generated by <code>scripts/build_report.py</code> from "
            "<code>results/results.json</code> committed "
            f"in {esc(res_sha[:7])} · code {esc(src_sha[:7])} · simulated "
            f"{esc(m['generated_at'][:10])}"
        ),
        Raw(f'<a href="{HOME_URL}">Allen Wang</a> · <a href="{REPO_URL}">Source</a>'),
    )
    return str(
        rk.page(
            title="pairwise-ratings · Allen Wang",
            description=(
                "Simulation study of Elo, Glicko-2 and Bradley–Terry: Bradley–Terry recovers the "
                "hidden "
                f"ranking best (Kendall τ {tau3(bt)} vs {tau3(elo)} for Elo, {n_runs} of "
                f"{m['n_players']} players)."
            ),
            canonical=PAGE_URL,
            header=rk.report_header(HOME_URL, "← Allen Wang", [("Source", REPO_URL)]),
            main=main,
            footer=footer,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "site" / "index.html")
    parser.add_argument(
        "--readme", type=Path, help="rewrite the results block between the README markers"
    )
    args = parser.parse_args()
    if not args.results.exists():
        sys.exit(f"{args.results} not found; run `make results` first")
    results = json.loads(args.results.read_text())
    if not all("per_seed" in c["reach"] for c in results["strategies"]):
        sys.exit(f"{args.results} has no per-seed reach values; run `make results` first")
    # The commit holding these results is computed, never typed; uncommitted results fall back
    # to the commit that produced them until they are committed and the page is rebuilt.
    try:
        rel = str(args.results.resolve().relative_to(ROOT))
    except ValueError:
        rel = None
    res_sha = (git_last_commit(rel) if rel else None) or results["meta"]["sha"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(results, res_sha))
    if args.readme:
        update_readme(args.readme, derive(results))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

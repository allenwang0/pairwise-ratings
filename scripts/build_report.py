"""Build site/index.html from results/results.json: inline SVG, no external requests."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
W, H = 520, 320
MARGIN = {"left": 48, "right": 128, "top": 14, "bottom": 44}
DASHES = ["", "7 4", "2 3", "9 3 2 3"]
NAMES = {
    "elo": "Elo",
    "glicko2": "Glicko-2",
    "bradley_terry": "Bradley-Terry",
    "random": "Random",
    "uncertainty": "Uncertainty",
    "closest": "Closest",
    "info_gain": "Info gain",
}
MARKERS = ["circle", "square", "triangle"]


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


class Scale:
    def __init__(self, d0: float, d1: float, r0: float, r1: float) -> None:
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v: float) -> float:
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def axes(xs: Scale, ys: Scale, xt: list[float], yt: list[float], xlabel: str, ylabel: str) -> str:
    parts = []
    for t in yt:
        y = ys(t)
        parts.append(
            f'<line class="grid" x1="{xs.r0}" x2="{xs.r1}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<text class="tick" x="{xs.r0 - 6}" y="{y + 4:.1f}" text-anchor="end">{t:g}</text>'
        )
    for t in xt:
        x = xs(t)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{ys.r0 + 16}" text-anchor="middle">{t:g}</text>'
        )
    parts.append(f'<line class="axis" x1="{xs.r0}" x2="{xs.r1}" y1="{ys.r0}" y2="{ys.r0}"/>')
    parts.append(
        f'<text class="label" x="{(xs.r0 + xs.r1) / 2}" y="{H - 6}" '
        f'text-anchor="middle">{esc(xlabel)}</text>'
        f'<text class="label" transform="translate(12 {(ys.r0 + ys.r1) / 2}) rotate(-90)" '
        f'text-anchor="middle">{esc(ylabel)}</text>'
    )
    return "".join(parts)


def legend(names: list[str], marker: bool = False) -> str:
    items = []
    for i, n in enumerate(names):
        if marker:
            swatch = f'<svg width="14" height="14" aria-hidden="true">{shape(i, 7, 7, 5)}</svg>'
        else:
            swatch = (
                f'<svg width="28" height="10" aria-hidden="true"><line x1="0" x2="28" y1="5" '
                f'y2="5" class="s{i}" stroke-width="2.5" stroke-dasharray="{DASHES[i]}"/></svg>'
            )
        items.append(f"<span>{swatch}{esc(n)}</span>")
    return f'<div class="legend">{"".join(items)}</div>'


def shape(i: int, x: float, y: float, r: float) -> str:
    kind = MARKERS[i % len(MARKERS)]
    if kind == "circle":
        return f'<circle class="f{i}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>'
    if kind == "square":
        side = 2 * r
        return (
            f'<rect class="f{i}" x="{x - r:.1f}" y="{y - r:.1f}" '
            f'width="{side:.1f}" height="{side:.1f}"/>'
        )
    pts = " ".join(
        f"{x + dx * r:.1f},{y + dy * r:.1f}" for dx, dy in ((0, -1.2), (-1.1, 0.8), (1.1, 0.8))
    )
    return f'<polygon class="f{i}" points="{pts}"/>'


def spread_labels(ys: list[float], gap: float = 14.0) -> list[float]:
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [0.0] * len(ys)
    last = -1e9
    for i in order:
        placed[i] = max(ys[i], last + gap)
        last = placed[i]
    return placed


def line_chart(
    chart_id: str,
    x: list[int],
    series: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    target: float | None,
) -> str:
    lo = min(min(s[metric]["lo"]) for s in series)
    y0 = min(0.0, round(lo - 0.05, 1))
    xs = Scale(0, x[-1], MARGIN["left"], W - MARGIN["right"])
    ys = Scale(y0, 1.0, H - MARGIN["bottom"], MARGIN["top"])
    parts = [axes(xs, ys, ticks(0, x[-1], 4), ticks(y0, 1.0, 5), "Comparisons (count)", ylabel)]
    if target is not None:
        ty = ys(target)
        parts.append(
            f'<line class="ref" x1="{xs.r0}" x2="{xs.r1}" y1="{ty:.1f}" y2="{ty:.1f}"/>'
            f'<text class="tick" x="{xs.r0 + 4}" y="{ty - 4:.1f}">tau {target}</text>'
        )
    ends = []
    for i, s in enumerate(series):
        band = s[metric]
        top = " ".join(f"{xs(a):.1f},{ys(b):.1f}" for a, b in zip(x, band["hi"], strict=True))
        bottom = " ".join(
            f"{xs(a):.1f},{ys(b):.1f}"
            for a, b in zip(reversed(x), reversed(band["lo"]), strict=True)
        )
        parts.append(f'<polygon class="band f{i}" points="{top} {bottom}"/>')
        line = " ".join(f"{xs(a):.1f},{ys(b):.1f}" for a, b in zip(x, band["mean"], strict=True))
        parts.append(
            f'<polyline class="line s{i}" stroke-dasharray="{DASHES[i]}" points="{line}"/>'
        )
        ends.append(ys(band["mean"][-1]))
    for i, (s, ly) in enumerate(zip(series, spread_labels(ends), strict=True)):
        parts.append(
            f'<text class="direct" x="{xs.r1 + 6}" y="{ly + 4:.1f}">{esc(s["label"])}</text>'
            f'<line class="s{i}" x1="{xs.r1 + 1}" x2="{xs.r1 + 5}" y1="{ly:.1f}" y2="{ly:.1f}"/>'
        )
    parts.append(
        f'<line class="cross" x1="0" x2="0" y1="{ys.r1}" y2="{ys.r0}" visibility="hidden"/>'
        f'<rect class="hit" x="{xs.r0}" y="{ys.r1}" width="{xs.r1 - xs.r0}" '
        f'height="{ys.r0 - ys.r1}"/>'
    )
    data = {
        "x": x,
        "x0": xs.r0,
        "x1": xs.r1,
        "xmax": x[-1],
        "series": [
            {
                "name": s["label"],
                "mean": s[metric]["mean"],
                "lo": s[metric]["lo"],
                "hi": s[metric]["hi"],
            }
            for s in series
        ],
    }
    return (
        f'<div class="plot" data-chart="{chart_id}" data-metric="{metric}">'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(ylabel)} by comparisons">'
        f"{''.join(parts)}</svg>"
        f'<script type="application/json">{json.dumps(data)}</script></div>'
    )


def line_table(x: list[int], series: list[dict[str, Any]], metric: str, caption: str) -> str:
    step = max(1, len(x) // 16)
    head = "".join(f"<th>{esc(s['label'])} mean [95% CI]</th>" for s in series)
    rows = []
    for k in list(range(step - 1, len(x), step)):
        cells = "".join(
            f"<td>{fmt(s[metric]['mean'][k])} [{fmt(s[metric]['lo'][k])}, "
            f"{fmt(s[metric]['hi'][k])}]</td>"
            for s in series
        )
        rows.append(f"<tr><td>{x[k]}</td>{cells}</tr>")
    return (
        f"<details><summary>Data table</summary><table><caption>{esc(caption)}</caption>"
        f"<tr><th>Comparisons</th>{head}</tr>{''.join(rows)}</table></details>"
    )


def accuracy_section(
    section_id: str,
    title: str,
    takeaway: str,
    x: list[int],
    series: list[dict[str, Any]],
    target: float | None,
) -> str:
    names = [s["label"] for s in series]
    toggle = (
        f'<div class="toggle" role="radiogroup" aria-label="Metric">'
        f'<label><input type="radio" name="{section_id}-m" value="tau" checked> Kendall tau</label>'
        f'<label><input type="radio" name="{section_id}-m" value="rho"> Spearman rho</label></div>'
    )
    charts = line_chart(section_id, x, series, "tau", "Kendall tau vs true skill", target)
    charts += line_chart(section_id, x, series, "rho", "Spearman rho vs true skill", target)
    tables = line_table(x, series, "tau", f"{title}: Kendall tau")
    return (
        f'<section id="{section_id}"><h2>{esc(title)}</h2><p class="takeaway">{esc(takeaway)}</p>'
        f'{toggle}{legend(names)}<div class="chart">{charts}<div class="tip" hidden></div></div>'
        f'<p class="note">Lines are the mean over seeds; bands are 95% confidence intervals of '
        f"the mean.</p>{tables}</section>"
    )


def reach_chart(rows: list[dict[str, Any]], budget: int, target: float) -> str:
    row_h = 34
    h = MARGIN["top"] + row_h * len(rows) + 44
    left = 190
    xs = Scale(0, budget, left, W - 20)
    parts = []
    for t in ticks(0, budget, 4):
        x = xs(t)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{MARGIN["top"]}" y2="{h - 44}"/>'
            f'<text class="tick" x="{x:.1f}" y="{h - 28}" text-anchor="middle">{t:g}</text>'
        )
    for j, r in enumerate(rows):
        y = MARGIN["top"] + row_h * j + row_h / 2
        reach = r["reach"]
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y - 2:.1f}" text-anchor="end">'
            f"{esc(r['label'])}</text>"
            f'<text class="small" x="{left - 8}" y="{y + 11:.1f}" text-anchor="end">'
            f"{reach['never']} of {reach['seeds']} never reach</text>"
        )
        if reach["median"] is None:
            parts.append(f'<text class="tick" x="{left}" y="{y + 4:.1f}">no seed reached</text>')
            continue
        tip = (
            f"{r['label']}: median {reach['median']:g}, IQR {reach['q1']:g} to {reach['q3']:g}; "
            f"{reach['never']} of {reach['seeds']} seeds never reach tau {target}"
        )
        parts.append(
            f'<g class="mark" tabindex="0"><title>{esc(tip)}</title>'
            f'<line class="whisker" x1="{xs(reach["q1"]):.1f}" x2="{xs(reach["q3"]):.1f}" '
            f'y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<circle class="dot" cx="{xs(reach["median"]):.1f}" cy="{y:.1f}" r="5"/></g>'
        )
    parts.append(
        f'<text class="label" x="{(left + W - 20) / 2}" y="{h - 6}" text-anchor="middle">'
        f"Comparisons to first reach tau {target} (count)</text>"
    )
    return (
        f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Comparisons needed to reach tau '
        f'{target}">{"".join(parts)}</svg>'
    )


def reliability_chart(series: list[dict[str, Any]]) -> str:
    size = 340
    xs = Scale(0, 1, 48, size - 14)
    ys = Scale(0, 1, size - 44, 14)
    parts = [axes(xs, ys, ticks(0, 1, 5), ticks(0, 1, 5), "", "Observed win frequency")]
    parts.append(
        f'<text class="label" x="{(xs.r0 + xs.r1) / 2}" y="{size - 6}" text-anchor="middle">'
        "Predicted win probability</text>"
        f'<line class="ref" x1="{xs(0)}" y1="{ys(0)}" x2="{xs(1)}" y2="{ys(1)}"/>'
    )
    biggest = max(b["count"] for s in series for b in s["calibration"]["bins"])
    for i, s in enumerate(series):
        for b in s["calibration"]["bins"]:
            r = 3 + 5 * (b["count"] / biggest) ** 0.5
            tip = (
                f"{s['label']}: predicted {b['p_mean']:.3f}, observed {b['freq']:.3f}, "
                f"{b['count']} comparisons"
            )
            parts.append(
                f'<g class="mark" tabindex="0"><title>{esc(tip)}</title>'
                f"{shape(i, xs(b['p_mean']), ys(b['freq']), r)}</g>"
            )
    return (
        f'<svg class="square" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="Reliability diagram">{"".join(parts)}</svg>'
    )


def build(results: dict[str, Any]) -> str:
    meta = results["meta"]
    x = meta["checkpoints"]
    target = meta["tau_target"]
    systems = [dict(c, label=NAMES[c["system"]]) for c in results["systems"]]
    strategies = [dict(c, label=NAMES[c["strategy"]]) for c in results["strategies"]]

    final = sorted(systems, key=lambda s: -s["final_tau"])
    sys_take = (
        f"After {meta['budget']} random comparisons, mean Kendall tau is "
        + ", ".join(f"{s['label']} {s['final_tau']:.3f}" for s in final)
        + "."
    )
    crossing = {s["strategy"]: s["reach"]["mean_curve_crossing"] for s in strategies}
    reached = {k: v for k, v in crossing.items() if v is not None}
    if reached:
        best = min(reached, key=lambda k: reached[k])
        base = crossing["random"]
        strat_take = (
            f"{NAMES[best]} pairing lifts mean tau to {target} after {reached[best]} comparisons; "
            + (
                f"random pairing needs {base}."
                if base is not None
                else f"random pairing does not get there in {meta['budget']}."
            )
        )
    else:
        strat_take = f"No strategy's mean tau reaches {target} within {meta['budget']} comparisons."

    reach_rows = [dict(c, label=f"Glicko-2 + {NAMES[c['strategy']]}") for c in strategies] + [
        dict(c, label=f"{NAMES[c['system']]} + Random") for c in systems if c["system"] != "glicko2"
    ]
    medians = [r["reach"]["median"] for r in reach_rows if r["reach"]["median"] is not None]
    by_never = sorted(reach_rows, key=lambda r: r["reach"]["never"])
    reach_take = (
        f"Per-seed medians only range from {min(medians):g} to {max(medians):g} comparisons; "
        f"the share of seeds that never get there separates the methods: "
        f"{by_never[0]['label']} {by_never[0]['reach']['never']} of "
        f"{by_never[0]['reach']['seeds']}, {by_never[-1]['label']} "
        f"{by_never[-1]['reach']['never']}."
        if medians
        else f"No seed reaches tau {target}."
    )

    cal = sorted(systems, key=lambda s: s["calibration"]["ece"])
    cal_take = (
        f"{cal[0]['label']} has the lowest expected calibration error "
        f"({cal[0]['calibration']['ece']:.3f}); {cal[-1]['label']} the highest "
        f"({cal[-1]['calibration']['ece']:.3f})."
    )
    cal_rows = "".join(
        f"<tr><td>{esc(s['label'])}</td><td>{s['calibration']['log_loss']:.4f}</td>"
        f"<td>{s['calibration']['oracle_log_loss']:.4f}</td>"
        f"<td>{s['calibration']['brier']:.4f}</td><td>{s['calibration']['ece']:.4f}</td>"
        f"<td>{s['calibration']['mean_abs_error_vs_true_p']:.4f}</td></tr>"
        for s in systems
    )

    ex = results["glickman_example"]
    worst = max(abs(ex["paper"][k] - ex["computed"][k]) for k in ("rating", "rd"))
    ex_take = (
        f"This code matches the paper's printed values to within {worst:.4f} rating points; "
        "the paper rounds its intermediate steps, which accounts for the gap."
    )
    ex_rows = "".join(
        f"<tr><td>{k}</td><td>{ex['paper'][k]}</td><td>{ex['computed'][k]:.5f}</td></tr>"
        for k in ("rating", "rd", "volatility")
    )

    status = " (uncommitted changes)" if meta["dirty"] else ""
    body = f"""
<header>
<h1>Pairwise rating benchmark</h1>
<p>Elo, Glicko-2 and Bradley-Terry MLE, and four ways to choose the next pair, measured on
<strong>simulated</strong> comparisons between synthetic players with known skill.</p>
<p class="meta">{meta["seeds"]} seeds, {meta["n_players"]} players, skill drawn from
N(0, {meta["skill_sd"]:g}) on the logit scale, {meta["budget"]} comparisons per run.
Commit <code>{esc(meta["sha"][:12])}</code>{status}, generated {esc(meta["generated_at"])}.</p>
</header>
<main>
{
        accuracy_section(
            "systems",
            "Which rating system ranks best under random pairing?",
            sys_take,
            x,
            systems,
            None,
        )
    }
{
        accuracy_section(
            "strategies",
            "Does choosing the pair speed up ranking? (Glicko-2)",
            strat_take,
            x,
            strategies,
            target,
        )
    }
<section id="reach"><h2>How many comparisons until a run reaches tau {target}?</h2>
<p class="takeaway">{esc(reach_take)}</p>
<div class="chart">{reach_chart(reach_rows, meta["budget"], target)}</div>
<p class="note">Dot: median over seeds that reach the target. Whisker: 25th to 75th percentile.</p>
</section>
<section id="calibration"><h2>Are the predicted win probabilities honest?</h2>
<p class="takeaway">{esc(cal_take)}</p>
{legend([s["label"] for s in systems], marker=True)}
<div class="chart">{reliability_chart(systems)}</div>
<p class="note">Each mark is one of 10 equal-width bins of the probability a system gave before
the comparison; area grows with the number of comparisons in the bin. The diagonal is perfect
calibration. Random pairing, all {systems[0]["calibration"]["comparisons"]} comparisons
per system.</p>
<div class="scroll"><table><caption>Calibration under random pairing</caption>
<tr><th>System</th><th>Log loss</th><th>Oracle log loss</th><th>Brier</th><th>ECE</th>
<th>Mean |p - true p|</th></tr>{cal_rows}</table></div>
<p class="note">Oracle log loss scores the true simulated probabilities; no system can beat it
on average.</p>
</section>
<section id="glickman"><h2>Glicko-2 against Glickman's worked example</h2>
<p class="takeaway">{esc(ex_take)}</p>
<p class="note">Player 1500, RD 200, volatility 0.06, tau 0.5: a win against 1400 (RD 30) and
losses against 1550 (RD 100) and 1700 (RD 300).</p>
<table>
<tr><th>Quantity</th><th>Paper</th><th>This code</th></tr>{ex_rows}</table>
</section>
</main>
<footer><p>Generated by <code>scripts/build_report.py</code> from
<code>results/results.json</code>, written by <code>{esc(meta["command"])}</code>.</p></footer>
"""
    return PAGE.replace("{{BODY}}", body)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pairwise Rating Benchmark</title>
<style>
:root {
  color-scheme: light;
  --bg: #fcfcfb; --text: #0b0b0b; --muted: #52514e; --grid: #e4e3df; --axis: #8a8984;
  --s0: #2a78d6; --s1: #eb6834; --s2: #1baf7a; --s3: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --grid: #34332f; --axis: #77766f;
    --s0: #3987e5; --s1: #d95926; --s2: #199e70; --s3: #c98500;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1a1a19; --text: #f0efec; --muted: #c3c2b7; --grid: #34332f; --axis: #77766f;
  --s0: #3987e5; --s1: #d95926; --s2: #199e70; --s3: #c98500;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header, main, footer { max-width: 760px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 24px; }
h1 { font-size: 1.6rem; margin: 0 0 8px; }
h2 { font-size: 1.15rem; margin: 40px 0 4px; }
.meta, .note, footer { color: var(--muted); font-size: 0.85rem; }
.takeaway { margin: 0 0 12px; }
.chart { position: relative; }
svg { display: block; width: 100%; height: auto; font-family: inherit; }
svg.square { max-width: 420px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.ref { stroke: var(--muted); stroke-width: 1.2; stroke-dasharray: 5 4; }
.tick, .direct { fill: var(--muted); font-size: 12px; }
.small { fill: var(--muted); font-size: 10px; }
.label { fill: var(--text); font-size: 12.5px; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; }
.band { opacity: 0.16; stroke: none; }
.s0 { stroke: var(--s0); } .s1 { stroke: var(--s1); } .s2 { stroke: var(--s2); }
.s3 { stroke: var(--s3); }
.f0 { fill: var(--s0); } .f1 { fill: var(--s1); } .f2 { fill: var(--s2); } .f3 { fill: var(--s3); }
.mark circle, .mark rect, .mark polygon { stroke: var(--bg); stroke-width: 1.5; fill-opacity: 0.9; }
.whisker { stroke: var(--text); stroke-width: 2; }
.dot { fill: var(--s0); stroke: var(--bg); stroke-width: 2; }
.mark:focus { outline: none; } .mark:focus .dot { stroke: var(--text); }
.cross { stroke: var(--muted); stroke-width: 1; }
.hit { fill: transparent; cursor: crosshair; }
.plot[hidden] { display: none; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; margin: 6px 0; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: auto; display: inline; }
.toggle { display: flex; gap: 16px; font-size: 0.85rem; }
.tip { position: absolute; pointer-events: none; background: var(--bg); color: var(--text);
  border: 1px solid var(--grid); border-radius: 6px; padding: 6px 8px; font-size: 0.8rem;
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); white-space: nowrap; }
table { border-collapse: collapse; font-size: 0.85rem; margin: 8px 0; }
caption { text-align: left; color: var(--muted); padding-bottom: 4px; }
th, td { text-align: left; padding: 3px 10px 3px 0; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; }
details { font-size: 0.85rem; overflow-x: auto; }
.scroll { overflow-x: auto; }
code { font-size: 0.85em; }
footer { padding: 32px 16px 40px; }
@media (max-width: 640px) {
  .tick, .direct, .label { font-size: 15px; }
  .small { font-size: 12px; }
}
</style>
</head>
<body>
{{BODY}}
<script>
document.querySelectorAll("section").forEach((sec) => {
  const plots = sec.querySelectorAll(".plot");
  sec.querySelectorAll('.toggle input').forEach((input) => {
    input.addEventListener("change", () => {
      plots.forEach((p) => { p.hidden = p.dataset.metric !== input.value; });
    });
  });
  plots.forEach((p) => { p.hidden = p.dataset.metric !== "tau"; });
  const tip = sec.querySelector(".tip");
  plots.forEach((plot) => {
    const data = JSON.parse(plot.querySelector("script").textContent);
    const svg = plot.querySelector("svg");
    const cross = svg.querySelector(".cross");
    const hit = svg.querySelector(".hit");
    const show = (clientX) => {
      const box = svg.getBoundingClientRect();
      const vx = (clientX - box.left) * (svg.viewBox.baseVal.width / box.width);
      const frac = Math.min(1, Math.max(0, (vx - data.x0) / (data.x1 - data.x0)));
      let k = 0;
      data.x.forEach((v, i) => {
        if (Math.abs(v - frac * data.xmax) < Math.abs(data.x[k] - frac * data.xmax)) k = i;
      });
      const px = data.x0 + (data.x[k] / data.xmax) * (data.x1 - data.x0);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px);
      cross.setAttribute("visibility", "visible");
      const rows = data.series.map((s) =>
        `<div>${s.name}: ${s.mean[k].toFixed(3)} ` +
        `[${s.lo[k].toFixed(3)}, ${s.hi[k].toFixed(3)}]</div>`);
      tip.innerHTML = `<div><strong>${data.x[k]} comparisons</strong></div>` + rows.join("");
      tip.hidden = false;
      const left = (px / svg.viewBox.baseVal.width) * box.width;
      const wrap = tip.parentElement.getBoundingClientRect();
      tip.style.left = Math.min(Math.max(0, left + 12), wrap.width - tip.offsetWidth) + "px";
      tip.style.top = "8px";
    };
    hit.addEventListener("pointermove", (e) => show(e.clientX));
    hit.addEventListener("pointerleave", () => { tip.hidden = true;
      cross.setAttribute("visibility", "hidden"); });
  });
});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "site" / "index.html")
    args = parser.parse_args()
    if not args.results.exists():
        sys.exit(f"{args.results} not found; run `make results` first")
    results = json.loads(args.results.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(results))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

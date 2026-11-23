"""Build site/index.html from results/results.json with report-kit v1 inlined (no requests).

Every number on the page is read from the results file or computed from it here. The kit files in
site/kit/ are vendored (report-kit v1.0.0) and inlined so the page also works as a single file.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "site" / "kit"
REPO = "https://github.com/allenwang0/pairwise-ratings"
HUB = "https://allenwang0.github.io/"
# Line charts start here so the differences near the target are visible; the omitted early
# checkpoints are in each chart's table and in results.json.
Y_FLOOR = 0.6
NAMES = {
    "elo": "Elo",
    "glicko2": "Glicko-2",
    "bradley_terry": "Bradley-Terry",
    "random": "Random",
    "uncertainty": "Uncertainty",
    "closest": "Closest",
    "info_gain": "Info gain",
}
RULES = {
    "info_gain": "the pair whose result would shrink Glicko-2's total uncertainty the most",
    "closest": "the most even matchup by predicted win probability",
    "uncertainty": "the least-known player against a random opponent",
    "random": "two players uniformly at random",
}
COLORS = {
    "info_gain": "var(--series-1)",
    "random": "var(--series-2)",
    "closest": "var(--series-3)",
    "uncertainty": "var(--series-4)",
    "bradley_terry": "var(--series-1)",
    "glicko2": "var(--series-2)",
    "elo": "var(--series-3)",
}


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def num(v: float | None) -> str:
    """Counts and checkpoint values: thousands separator, at most one decimal (quartiles)."""
    if v is None:
        return "never"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def pct(fraction: float) -> str:
    return f"{fraction * 100:.0f} %" if abs(fraction) >= 0.1 else f"{fraction * 100:.1f} %"


def first_at(values: list[float], x: list[int], target: float) -> int | None:
    return next((x[i] for i, v in enumerate(values) if v >= target), None)


def lead_from(x: list[int], best: list[float], others: list[list[float]]) -> int | None:
    """First checkpoint from which `best` stays above every other curve to the end."""
    for i in range(len(x)):
        if all(best[j] > max(o[j] for o in others) for j in range(i, len(x))):
            return x[i]
    return None


def saving(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    return (base - cand) / base


def kit_file(name: str) -> str:
    path = KIT / name
    if not path.exists():
        sys.exit(f"{path} missing; vendor report-kit with sync-kit.sh site")
    # A literal "</script" inside an inlined script (the kit's usage comment has one) would end
    # the <script> element early.
    return path.read_text().replace("</script", "<\\/script")


def versions() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    out = {"Python": project["requires-python"]}
    for dep in project["dependencies"]:
        name, _, ver = dep.partition("==")
        out[name] = ver
    return out


def build(results: dict[str, Any]) -> str:
    meta = results["meta"]
    x: list[int] = meta["checkpoints"]
    target: float = meta["tau_target"]
    seeds: int = meta["seeds"]
    players: int = meta["n_players"]
    budget: int = meta["budget"]
    strat = {c["strategy"]: c for c in results["strategies"]}
    systems = {c["system"]: c for c in results["systems"]}
    ig, rnd = strat["info_gain"]["reach"], strat["random"]["reach"]
    bt = systems["bradley_terry"]["reach"]
    mean_save = saving(rnd["mean_curve_crossing"], ig["mean_curve_crossing"])
    med_save = saving(rnd["median"], ig["median"])

    def band_range(cond: dict[str, Any]) -> str:
        hi = first_at(cond["tau"]["hi"], x, target)
        lo = first_at(cond["tau"]["lo"], x, target)
        return f"{num(hi)}–{num(lo)}" if lo is not None else f"{num(hi)} to beyond {budget:,}"

    # ---------- copy (all numbers from results) ----------
    if mean_save is not None and med_save is not None:
        h1 = (
            f"Info-gain pairing reaches τ {target} sooner on average, "
            f"but the typical run saves about {med_save * 100:.0f} %"
        )
        hero_line = (
            f"Averaged over {seeds} runs, the Glicko-2 accuracy curve reaches τ {target} after "
            f"{num(ig['mean_curve_crossing'])} comparisons with info-gain pairing and "
            f"{num(rnd['mean_curve_crossing'])} with random pairs ({pct(mean_save)} fewer). "
            f"The median run gets there at {num(ig['median'])} vs {num(rnd['median'])}."
        )
    else:
        h1 = f"How many comparisons until a ranking reaches Kendall τ {target}?"
        hero_line = f"Not every condition reaches τ {target} within {budget:,} comparisons."
    final = sorted(systems.values(), key=lambda c: -c["final_tau"])
    ece = sorted(systems.values(), key=lambda c: c["calibration"]["ece"])
    ex = results["glickman_example"]
    worst = max(abs(ex["paper"][k] - ex["computed"][k]) for k in ("rating", "rd"))
    worst_round = max(0.01, round(worst + 0.005, 2))
    never = {k: c["reach"]["never"] for k, c in strat.items()}

    def lead_title(best: dict[str, Any], others: list[dict[str, Any]], who: str, vs: str) -> str:
        at = lead_from(x, best["tau"]["mean"], [o["tau"]["mean"] for o in others])
        if at is None:
            return f"{who}'s mean does not stay above {vs}"
        if at == x[0]:
            return f"{who}'s mean is above {vs} at every checkpoint"
        return f"{who}'s mean stays above {vs} from {at:,} comparisons on"

    strat_title = lead_title(strat["info_gain"], [strat["random"]], "Info gain", "random pairs'")
    best_sys = final[0]
    sys_title = lead_title(
        best_sys,
        [c for c in systems.values() if c is not best_sys],
        NAMES[best_sys["system"]],
        "the other raters'",
    )
    flat_from = first_at(strat["random"]["tau"]["mean"], x, target - 0.01)

    # ---------- data for the charts ----------
    start = next(
        (
            i
            for i in range(len(x))
            if all(
                c["tau"]["lo"][j] >= Y_FLOOR
                for c in [*strat.values(), *systems.values()]
                for j in range(i, len(x))
            )
        ),
        0,
    )
    start = min(start, len(x) - 2)

    def series(conds: dict[str, Any], order: list[str]) -> list[dict[str, Any]]:
        out = []
        for i, k in enumerate(order):
            if k not in conds:
                continue
            t = conds[k]["tau"]
            out.append(
                {
                    "key": k,
                    "name": NAMES[k],
                    "color": COLORS[k],
                    "dash": i % 2 == 1,
                    "y": t["mean"][start:],
                    "lo": t["lo"][start:],
                    "hi": t["hi"][start:],
                    "full": {"mean": t["mean"], "lo": t["lo"], "hi": t["hi"]},
                }
            )
        return out

    reach_rows = [
        {
            "label": NAMES[k],
            "reach": strat[k]["reach"],
            "group": "Glicko-2",
        }
        for k in ("info_gain", "closest", "uncertainty", "random")
        if k in strat
    ] + [
        {"label": f"{NAMES[k]}, random", "reach": systems[k]["reach"], "group": NAMES[k]}
        for k in ("bradley_terry", "elo")
        if k in systems
    ]
    data = {
        "target": target,
        "seeds": seeds,
        "yFloor": Y_FLOOR,
        "x": x[start:],
        "xFull": x,
        "strategies": series(strat, ["info_gain", "random", "closest", "uncertainty"]),
        "systems": series(systems, ["bradley_terry", "glicko2", "elo"]),
        "reach": [
            {
                "label": r["label"],
                "q1": r["reach"]["q1"],
                "median": r["reach"]["median"],
                "q3": r["reach"]["q3"],
                "crossing": r["reach"]["mean_curve_crossing"],
                "never": r["reach"]["never"],
                "seeds": r["reach"]["seeds"],
            }
            for r in reach_rows
        ],
        "hero": [
            {
                "label": f"Mean curve reaches τ {target}",
                "ig": ig["mean_curve_crossing"],
                "rnd": rnd["mean_curve_crossing"],
            },
            {"label": f"Median run reaches τ {target}", "ig": ig["median"], "rnd": rnd["median"]},
        ],
    }

    # ---------- bento ----------
    def card(span: str, label: str, metric: str, cmp: str, foot: str, href: str = "") -> str:
        tag, attr = ("a", f' href="{href}"') if href else ("div", "")
        return (
            f'<{tag} class="card {span}"{attr}><span class="label">{label}</span>'
            f'<span class="metric num">{metric}</span><span class="cmp">{cmp}</span>'
            f'<span class="foot">{foot}</span></{tag}>'
        )

    unit = '<span class="unit">fewer</span>'
    cards = [
        card(
            "span-8",
            f"Average curve to τ {target}, Glicko-2",
            f"{pct(mean_save)} {unit}" if mean_save is not None else "n/a",
            f"{num(ig['mean_curve_crossing'])} comparisons with info-gain pairing vs "
            f"{num(rnd['mean_curve_crossing'])} with random pairs",
            f"95 % band of the mean reaches {target} at {band_range(strat['info_gain'])} vs "
            f"{band_range(strat['random'])}",
            "#strategies",
        ),
        card(
            "span-4",
            "Typical run, Glicko-2",
            f"{med_save * 100:.0f} % {unit}" if med_save is not None else "n/a",
            f"Median first reach: {num(ig['median'])} vs {num(rnd['median'])} comparisons",
            f"Runs that reached {target}: {seeds - ig['never']} and {seeds - rnd['never']} of "
            f"{seeds}",
            "#reach",
        ),
        card(
            "span-4",
            "Swap the rater instead",
            f"{num(bt['mean_curve_crossing'])}",
            f"Bradley-Terry with random pairs reaches mean τ {target} here, no smart pairing",
            f"vs {num(rnd['mean_curve_crossing'])} for Glicko-2 with random pairs",
            "#systems",
        ),
        card(
            "span-4",
            f"Runs that never reach τ {target}",
            f"{never.get('info_gain')} vs {never.get('random')}",
            f"of {seeds}, info gain vs random. Closest: {never.get('closest')}, "
            f"uncertainty: {never.get('uncertainty')}",
            f"Within {budget:,} comparisons",
            "#reach",
        ),
        card(
            "span-4",
            "Glicko-2 vs Glickman's example",
            f'{worst_round:.2f} <span class="unit">points</span>',
            "Largest gap to the paper's printed rating and RD",
            "Worked example from Glickman (2012)",
            "#glickman",
        ),
    ]

    # ---------- tables ----------
    cal_rows = "".join(
        f'<tr><th scope="row">{NAMES[c["system"]]}</th>'
        f"<td>{c['calibration']['ece']:.3f}</td><td>{c['calibration']['log_loss']:.3f}</td>"
        f"<td>{c['calibration']['oracle_log_loss']:.3f}</td>"
        f"<td>{c['calibration']['brier']:.3f}</td>"
        f"<td>{c['calibration']['mean_abs_error_vs_true_p']:.3f}</td></tr>"
        for c in results["systems"]
    )
    ex_rows = "".join(
        f'<tr><th scope="row">{label}</th><td>{ex["paper"][k]}</td>'
        f"<td>{ex['computed'][k]:.{6 if k == 'volatility' else 4}f}</td></tr>"
        for k, label in (
            ("rating", "Rating"),
            ("rd", "Rating deviation"),
            ("volatility", "Volatility"),
        )
    )
    step = x[1] - x[0] if len(x) > 1 else x[0]
    env = {
        **versions(),
        "Seeds": f"0–{seeds - 1} (each seed gives every condition the same players)",
        "Players per run": f"{players}",
        "Comparisons per run": f"{budget:,}, checkpoint every {step}",
        "Skill distribution": f"N(0, {meta['skill_sd']:g}) on the logit scale",
        "Target": f"Kendall τ ≥ {target}",
        "Command": f'<span class="mono">{esc(meta["command"])}</span>',
        "Commit": f'<span class="mono">{esc(meta["sha"][:7])}</span>'
        + (" (uncommitted changes)" if meta["dirty"] else ""),
        "Generated": esc(meta["generated_at"]),
    }
    env_rows = "".join(f'<tr><th scope="row">{k}</th><td>{v}</td></tr>' for k, v in env.items())
    date = meta["generated_at"][:10]

    # ---------- section copy ----------
    if mean_save is not None:
        strat_h2 = (
            f"With Glicko-2, info-gain pairing lifts the average curve to τ {target} after "
            f"{num(ig['mean_curve_crossing'])} comparisons, not {num(rnd['mean_curve_crossing'])}"
        )
        strat_lede = (
            f"The random curve creeps from {target - 0.01:.2f} to {target} between "
            f"{num(flat_from)} and {num(rnd['mean_curve_crossing'])} comparisons, so a small "
            f"vertical gap becomes a wide horizontal one. That is where the {pct(mean_save)} "
            "comes from."
            if flat_from is not None and rnd["mean_curve_crossing"] is not None
            else f"Mean over {seeds} runs; the band is the 95 % confidence interval of the mean."
        )
    else:
        strat_h2 = f"Not every pair-selection rule reaches mean τ {target} within {budget:,}"
        strat_lede = f"Mean over {seeds} runs; the band is the 95 % confidence interval."
    if med_save is not None:
        reach_h2 = (
            f"The typical run gains far less: median first reach is {num(ig['median'])} "
            f"comparisons with info gain vs {num(rnd['median'])} with random pairs"
        )
    else:
        reach_h2 = f"How many comparisons each run needs to reach τ {target}"
    quart = [r["reach"] for r in reach_rows if r["reach"]["median"] is not None]
    reach_title = (
        "The middle half of runs overlaps across every condition"
        if quart and max(r["q1"] for r in quart) < min(r["q3"] for r in quart)
        else "Per-run spread of first reach, by condition"
    )
    sys_h2 = (
        f"With random pairs, {NAMES[final[0]['system']]} ranks best: mean τ "
        f"{final[0]['final_tau']:.3f} after {budget:,} comparisons, "
        f"{NAMES[final[-1]['system']]} {final[-1]['final_tau']:.3f}"
    )
    bt_cross = bt["mean_curve_crossing"]
    sys_lede = (
        f"Bradley-Terry refits every skill from all results so far. Its mean curve reaches τ "
        f"{target} at {num(bt_cross)} comparisons, against {num(rnd['mean_curve_crossing'])} "
        f"for Glicko-2 with the same random pairs."
        if bt_cross is not None
        else f"Mean over {seeds} runs with random pairs."
    )

    body = f"""
<a class="skip" href="#main">Skip to content</a>
<header class="bar">
  <div class="wrap">
    <a class="back" href="{HUB}">← Allen Wang</a>
    <span class="title" aria-hidden="true">Pairwise rating benchmark</span>
    <a class="pill" href="{REPO}">Code ↗</a>
  </div>
</header>
<main id="main">
<section class="hero wrap" aria-labelledby="h1">
  <p class="eyebrow">Simulation benchmark · Python</p>
  <h1 id="h1">{esc(h1)}</h1>
  <p class="lede">If all you can ask is “which of these two is better?”, how fast can a rating
  system recover the true order? Elo, Glicko-2 and Bradley-Terry, plus four rules for choosing
  the next pair, each run on the same {seeds} sets of {players} simulated players.</p>
  <p class="conditions"><span class="key">Simulated players</span><span>Pair rules tested with
  Glicko-2 only</span><span>{players} players</span><span>{seeds} runs per condition</span></p>
  <p class="hero-line">{esc(hero_line)}</p>
  <figure class="hero-fig" aria-labelledby="hero-cap">
    <div id="c-hero"></div>
    <figcaption id="hero-cap">Comparisons needed to reach Kendall τ {target} (fewer is better).
    “Mean curve” is where the {seeds}-run average crosses {target}; “median run” is the median
    first reach among runs that got there.</figcaption>
  </figure>
</section>

<section class="wrap" aria-label="Key results">
  <div class="bento">{"".join(cards)}</div>
</section>

<section class="s wrap" id="strategies" aria-labelledby="h-strategies">
  <p class="eyebrow">Choosing the next pair · Glicko-2 ratings</p>
  <h2 id="h-strategies">{esc(strat_h2)}</h2>
  <p class="lede takeaway">{esc(strat_lede)}</p>
  <figure class="figure" aria-labelledby="f-strategies">
    <div class="fig-head"><div><h3 id="f-strategies">{esc(strat_title)}</h3>
    <p class="sub">Mean Kendall τ over {seeds} runs ·
    95 % confidence band of the mean · {players} players</p></div>
    <div class="seg" role="radiogroup" aria-label="View" data-view-toggle><button role="radio"
    aria-checked="true" data-value="chart">Chart</button><button role="radio"
    aria-checked="false" data-value="table">Table</button></div></div>
    <div id="c-strategies"></div><div id="t-strategies" hidden></div>
  </figure>
  <figcaption>Kendall τ: the share of player pairs in the right order, rescaled so 1 is the true
  order and 0 is chance. The y axis starts at {Y_FLOOR} and the x axis at {x[start]:,}, where
  every curve is above {Y_FLOOR}; the table has every checkpoint. Pair rules: info gain picks
  {RULES["info_gain"]}; closest, {RULES["closest"]}; uncertainty,
  {RULES["uncertainty"]}.</figcaption>
  <p class="caveat"><b>An average curve, not a typical run.</b> The {
        pct(mean_save) if mean_save is not None else "gap"
    } is read where two averaged curves cross {target}. Runs that stall pull
  the random average down, so a single run usually saves much less (next section).</p>
</section>

<section class="s wrap" id="reach" aria-labelledby="h-reach">
  <p class="eyebrow">Per-run spread · first reach of τ {target}</p>
  <h2 id="h-reach">{esc(reach_h2)}</h2>
  <p class="lede takeaway">The rules differ more in how many runs stall: {never.get("uncertainty")}
  of {seeds} uncertainty runs and {never.get("closest")} closest runs never reach {target},
  against {never.get("random")} for random pairs and {never.get("info_gain")} for info gain.
  The intuitive shortcuts are riskier, not uniformly slower.</p>
  <figure class="figure" aria-labelledby="f-reach">
    <div class="fig-head"><div><h3 id="f-reach">{esc(reach_title)}</h3>
    <p class="sub">Comparisons until each run first reaches τ {target} · {seeds} runs
    per row</p></div>
    <div class="seg" role="radiogroup" aria-label="View" data-view-toggle><button role="radio"
    aria-checked="true" data-value="chart">Chart</button><button role="radio"
    aria-checked="false" data-value="table">Table</button></div></div>
    <div class="legend" aria-hidden="true">
      <span><svg width="10" height="10"><circle cx="5" cy="5" r="4"
      fill="var(--series-1)"/></svg>Median run</span>
      <span><svg width="22" height="10"><line x1="1" y1="5" x2="21" y2="5"
      stroke="var(--series-1)" stroke-width="2"/></svg>Middle half of runs (25th–75th
      percentile)</span>
      <span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="none"
      stroke="var(--series-1)" stroke-width="2"/></svg>Where the mean curve crosses</span>
    </div>
    <div id="c-reach"></div><div id="t-reach" hidden></div>
  </figure>
  <figcaption>The first four rows use Glicko-2 ratings; the last two use random pairs with
  another rater. Percentiles count only runs that reached {target} within {budget:,}
  comparisons; the table gives how many did not. Checkpoints are every
  {step} comparisons, so values are rounded up to the next
  checkpoint. Rows without a ring never reach {target} on the mean curve.</figcaption>
</section>

<section class="s wrap" id="systems" aria-labelledby="h-systems">
  <p class="eyebrow">Rating systems · random pairs</p>
  <h2 id="h-systems">{esc(sys_h2)}</h2>
  <p class="lede takeaway">{esc(sys_lede)}</p>
  <figure class="figure" aria-labelledby="f-systems">
    <div class="fig-head"><div><h3 id="f-systems">{esc(sys_title)}</h3>
    <p class="sub">Mean Kendall τ over {seeds} runs · 95 % confidence band
    of the mean · random pairs</p></div>
    <div class="seg" role="radiogroup" aria-label="View" data-view-toggle><button role="radio"
    aria-checked="true" data-value="chart">Chart</button><button role="radio"
    aria-checked="false" data-value="table">Table</button></div></div>
    <div id="c-systems"></div><div id="t-systems" hidden></div>
  </figure>
  <figcaption>Same runs and axes as the pair-rule chart. The Glicko-2 line here is the random
  pairs line there.</figcaption>
</section>

<section class="s wrap" id="calibration" aria-labelledby="h-calibration">
  <p class="eyebrow">Predicted win probabilities · random pairs</p>
  <h2 id="h-calibration">All three systems' win probabilities sit within
  {ece[-1]["calibration"]["ece"]:.2f} of observed win rates on average</h2>
  <p class="lede takeaway">Expected calibration error (ECE) is the average gap between predicted
  and observed win rates across 10 probability bins. Each system is scored on all
  {results["systems"][0]["calibration"]["comparisons"]:,} comparisons from its {seeds} runs.</p>
  <div class="tablewrap" tabindex="0" role="region" aria-label="Calibration table">
  <table><caption>Calibration under random pairing, pooled over {seeds} runs</caption>
  <thead><tr><th scope="col">System</th><th scope="col">ECE</th><th scope="col">Log loss</th>
  <th scope="col">Oracle log loss</th><th scope="col">Brier</th>
  <th scope="col">Mean |p − true p|</th></tr></thead><tbody>{cal_rows}</tbody></table></div>
  <p class="caveat"><b>One pooled estimate, no spread.</b> These are single numbers over all runs,
  so differences of about 0.01 are not ranked here. Oracle log loss scores the true simulated
  probabilities; no system can beat it on average.</p>
</section>

<section class="s wrap" id="glickman" aria-labelledby="h-glickman">
  <p class="eyebrow">Correctness check</p>
  <h2 id="h-glickman">Glicko-2 reproduces Glickman's worked example to within
  {worst_round:.2f} rating points</h2>
  <p class="lede takeaway">Player 1500 (RD 200, volatility 0.06, system constant 0.5) wins
  against 1400 (RD 30) and loses to 1550 (RD 100) and 1700 (RD 300). The paper rounds its
  intermediate steps, which accounts for the gap.</p>
  <div class="tablewrap" tabindex="0" role="region" aria-label="Glickman example table">
  <table><caption>Glickman (2012), “Example of the Glicko-2 system”</caption>
  <thead><tr><th scope="col">Quantity</th><th scope="col">Paper</th>
  <th scope="col">This code</th></tr></thead><tbody>{ex_rows}</tbody></table></div>
</section>

<section class="s wrap" id="method" aria-labelledby="h-method">
  <p class="eyebrow">Methodology</p>
  <h2 id="h-method">How this was measured</h2>
  <div class="prose" style="margin-top:24px">
    <p>Each run draws {players} players with hidden skills from N(0, {meta["skill_sd"]:g}) on the
    logit scale. Player i beats j with probability 1 / (1 + e<sup>s<sub>j</sub> −
    s<sub>i</sub></sup>), the same model the rating systems assume. Every condition plays
    {budget:,} comparisons per run, over seeds 0–{seeds - 1}; a seed gives every condition the
    same players, so the conditions are compared on identical pools.</p>
    <p>Every {step} comparisons the estimated skills are ranked and
    scored against the true order with Kendall τ. Curves are the mean over runs, with a 95 %
    normal-approximation interval of the mean (1.96 standard errors). “First reach” is, per run,
    the first checkpoint with τ ≥ {target}. results.json also records Spearman ρ for every
    curve.</p>
  </div>
  <div class="two">
    <div><h3>What this shows</h3><ul>
      <li>With Glicko-2, info-gain pairing reaches a ranking accuracy target sooner on the
      average curve, and leaves fewer runs short of it.</li>
      <li>The median run gains much less than the average curve suggests.</li>
      <li>Changing the rater to Bradley-Terry helps about as much as smart pairing.</li>
    </ul></div>
    <div><h3>What it doesn't show</h3><ul>
      <li>Real players: outcomes follow the exact model the raters assume, and skills never
      change.</li>
      <li>Other pool sizes: every run has {players} players.</li>
      <li>Pair rules with Elo or Bradley-Terry: they were tested with Glicko-2 only.</li>
      <li>Other targets: {target} is one threshold; the gap differs at other values.</li>
    </ul></div>
  </div>
</section>

<section class="s wrap" id="reproduce" aria-labelledby="h-reproduce">
  <p class="eyebrow">Reproduce</p>
  <h2 id="h-reproduce">Run it yourself</h2>
  <div class="codewrap"><pre><code>git clone {REPO}.git
cd pairwise-ratings
make setup     # install pinned dependencies (needs uv)
make test
make results   # {seeds} simulated runs per condition, about 1 minute
make report    # writes site/index.html</code></pre></div>
  <div class="tablewrap" tabindex="0" role="region" aria-label="Environment table">
  <table><caption>Run settings, from results/results.json and pyproject.toml</caption>
  <thead><tr><th scope="col">Field</th><th scope="col">Value</th></tr></thead>
  <tbody>{env_rows}</tbody></table></div>
</section>
</main>
<footer><div class="wrap"><span><a href="{HUB}">← Allen Wang</a><a href="{REPO}">Code ↗</a></span>
<span>Data: results/results.json from commit <span class="mono">{esc(meta["sha"][:7])}</span>
· {esc(date)}</span></div></footer>
<script type="application/json" id="rk-data">{json.dumps(data)}</script>
"""
    return (
        PAGE.replace("{{KIT_CSS}}", kit_file("report.css"))
        .replace("{{KIT_JS}}", kit_file("report.js") + "\n" + kit_file("charts.js"))
        .replace("{{DESCRIPTION}}", esc(hero_line))
        .replace("{{BODY}}", body)
    )


LOCAL_CSS = """
:root{--series-3:#1baf7a;--series-4:#eda100}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--series-3:#199e70;
--series-4:#c98500}}
:root[data-theme="dark"]{--series-3:#199e70;--series-4:#c98500}
.hero-line{margin-top:24px;max-width:var(--measure);font-size:17px;color:var(--ink-1)}
.hero-fig{margin-top:24px;max-width:760px}
.hero-fig figcaption{margin-top:8px}
.bento .card .metric{white-space:nowrap}
.bento .card .cmp{margin-bottom:12px}
.s>.tablewrap{margin-top:24px}
.tablewrap td{white-space:normal}
/* charts.js never draws narrower than 260 px and scales down below that; keep the plot >= 260 px
   at 320 px viewports so tick text stays >= 11 px. */
@media (max-width:359px){.figure{padding-inline:12px}}
#reproduce .tablewrap td{text-align:left}
"""

LOCAL_JS = """
addEventListener("DOMContentLoaded", function () {
  var D = RK.data("rk-data"), T = D.target;
  var n = function (v) { return v == null ? "never" :
    v.toLocaleString("en-US", { maximumFractionDigits: 1 }); };
  var f3 = function (v) { return v.toFixed(3); };
  var f2 = function (v) { return v.toFixed(2).replace(/0$/, ""); };
  RKCharts.hbar("#c-hero", {
    rows: D.hero.map(function (r) { return { label: r.label, values: { ig: r.ig, rnd: r.rnd } }; }),
    series: [{ key: "ig", name: "Info-gain pairing", color: "var(--series-1)" },
             { key: "rnd", name: "Random pairs", color: "var(--series-2)" }],
    unit: "comparisons", format: n,
    title: "Comparisons to reach Kendall tau " + T + ": info gain vs random pairs"
  });
  function curves(id, set, title) {
    RKCharts.line("#c-" + id, {
      x: D.x, series: set.map(function (s) {
        return { key: s.key, name: s.name, color: s.color, dash: s.dash,
          y: s.y, lo: s.lo, hi: s.hi };
      }),
      xLabel: "Comparisons", yLabel: "Kendall τ, mean of " + D.seeds + " runs (axis starts at " +
        D.yFloor + "; dark line = target " + T + ")",
      ymin: D.yFloor, zero: false, endLabels: false, format: f2,
      refs: [{ y: T, label: "" }], title: title
    });
    var cols = [{ key: "x", label: "Comparisons" }].concat(set.map(function (s) {
      return { key: s.key, label: s.name + " mean [95 % CI]" }; }));
    var rows = [];
    D.xFull.forEach(function (xv, i) {
      if ((i + 1) % 8 && i !== D.xFull.length - 1) return;
      var r = { x: xv.toLocaleString("en-US") };
      set.forEach(function (s) { r[s.key] = f3(s.full.mean[i]) + " [" + f3(s.full.lo[i]) + ", " +
        f3(s.full.hi[i]) + "]"; });
      rows.push(r);
    });
    RKCharts.table("#t-" + id, { caption: title + " (every 8th checkpoint)",
      columns: cols, rows: rows });
  }
  curves("strategies", D.strategies, "Mean Kendall tau by pair-selection rule, Glicko-2");
  curves("systems", D.systems, "Mean Kendall tau by rating system, random pairs");
  var rows = D.reach.filter(function (r) { return r.median != null; }).map(function (r) {
    return { label: r.label, series: { a: { lo: r.q1, mid: r.median, hi: r.q3,
      mark: r.crossing == null ? undefined : r.crossing,
      tip: "median " + n(r.median) + " · middle half " + n(r.q1) + "–" + n(r.q3) +
        "<br>mean curve " + (r.crossing == null ? "never crosses" : "crosses at " + n(r.crossing)) +
        "<br>" + r.never + " of " + r.seeds + " runs never reach " + T } } };
  });
  var lo = Math.min.apply(null, rows.map(function (r) { return r.series.a.lo; }));
  var x0 = Math.floor(lo / 100) * 100;
  var top = Math.max.apply(null, rows.map(function (r) {
    return Math.max(r.series.a.hi, r.series.a.mark == null ? 0 : r.series.a.mark); }));
  RKCharts.dotRange("#c-reach", {
    rows: rows, series: [{ key: "a", name: "Runs", color: "var(--series-1)" }],
    xLabel: "Comparisons to first reach τ " + T + " (axis starts at " + x0 + ")",
    xmin: x0, xmax: top + 20, ticks: { step: 100 }, format: n,
    title: "Comparisons until each run first reaches Kendall tau " + T
  });
  RKCharts.table("#t-reach", {
    caption: "First reach of tau " + T + " per condition",
    columns: [{ key: "label", label: "Condition" }, { key: "q1", label: "25th pct" },
      { key: "median", label: "Median" }, { key: "q3", label: "75th pct" },
      { key: "crossing", label: "Mean curve crosses" }, { key: "never", label: "Never reach" }],
    rows: D.reach.map(function (r) { return { label: r.label, q1: n(r.q1), median: n(r.median),
      q3: n(r.q3), crossing: n(r.crossing), never: r.never + " of " + r.seeds }; })
  });
});
"""

PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Pairwise rating benchmark · Allen Wang</title>
<meta name="description" content="{{DESCRIPTION}}">
<link rel="icon" href="data:,">
<meta name="theme-color" content="#f5f5f7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0b0b0b" media="(prefers-color-scheme: dark)">
<script>try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")"""
    """document.documentElement.setAttribute("data-theme",t)}catch(e){}</script>
<style>
{{KIT_CSS}}
/* report.local: simulated-data report additions */
"""
    + LOCAL_CSS
    + """</style>
</head>
<body>
{{BODY}}
<script>
{{KIT_JS}}
/* report.local */
"""
    + LOCAL_JS
    + """</script>
</body>
</html>
"""
)


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

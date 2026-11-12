# pairwise-ratings

Elo, Glicko-2 and Bradley-Terry maximum likelihood compared on simulated pairwise comparisons, plus four rules for choosing which pair to compare next, scored by how fast each recovers a known ranking and how honest its win probabilities are.

![Report: Kendall tau against comparisons for three rating systems and four pair-selection strategies](docs/report.png)

The report is `site/index.html`, a single static page with inline SVG charts, hover readouts, a Kendall tau / Spearman rho toggle and data tables. All data in it is simulated.

## Results

Setup: 32 synthetic players with skill drawn from N(0, 1) on the logit scale, outcomes drawn from the Bradley-Terry model, 1280 comparisons per run, seeds 0 to 99. Every condition with the same seed faces the same players. Numbers below come from `results/results.json` at commit `bc84bdc`.

Pair selection, all with Glicko-2 as the rating system:

| Strategy | Mean tau after 1280 | Mean curve reaches tau 0.8 at | Per-seed median to tau 0.8 (IQR) | Seeds that never reach 0.8 |
|---|---|---|---|---|
| Info gain | 0.857 | 740 | 670 (472.5 to 847.5) | 6 of 100 |
| Closest | 0.827 | 870 | 650 (447.5 to 912.5) | 20 of 100 |
| Random | 0.819 | 1010 | 700 (560 to 885) | 9 of 100 |
| Uncertainty | 0.794 | never | 640 (495 to 840) | 25 of 100 |

Info-gain pairing gets the mean Kendall tau to 0.8 after 740 comparisons; random pairing needs 1010, so it saves 270 comparisons (27%). At 1280 comparisons the 95% confidence intervals of the mean do not overlap: info gain 0.849 to 0.865, random 0.812 to 0.827. The per-seed medians barely differ; what separates the strategies is how many seeds stall. Uncertainty and closest-match pairing both leave more seeds short of 0.8 than random pairing does.

Rating systems under random pairing:

| System | Mean tau after 1280 | Log loss | Brier | ECE | Mean abs. error vs true p |
|---|---|---|---|---|---|
| Bradley-Terry MLE | 0.841 | 0.5764 | 0.1963 | 0.0266 | 0.0877 |
| Glicko-2 | 0.819 | 0.5798 | 0.1975 | 0.0367 | 0.0926 |
| Elo (K = 32) | 0.792 | 0.5814 | 0.1988 | 0.0261 | 0.1035 |

The oracle log loss, scoring the true simulated probabilities, is 0.5376, so no system can do better than that on average. The reliability diagram in the report shows Glicko-2 and Bradley-Terry are overconfident at the extremes (predicted 0.06, observed about 0.09 to 0.10), while Elo is slightly underconfident there.

Glicko-2 reproduces Glickman's worked example (player 1500 / 200 / 0.06, tau 0.5, one win and two losses): rating 1464.0507, RD 151.5165, volatility 0.059996 against the paper's printed 1464.06, 151.52 and 0.05999. The 0.0093-point gap in rating comes from the paper rounding its intermediate steps.

## How it works

Rating systems (`src/pairwise_ratings/`):

- `elo.py`: expected score 1 / (1 + 10^((Rb - Ra) / 400)), K = 32.
- `glicko2.py`: steps 1 to 8 of Glickman, "Example of the Glicko-2 system" (2013), with the Illinois regula falsi solver for volatility. Predictions use the combined deviation g(sqrt(phi_a^2 + phi_b^2)) from Glickman (1999).
- `bradley_terry.py`: maximum likelihood by Hunter's MM algorithm (Hunter 2004). Each player gets one virtual win and one virtual loss against a fixed player of strength 1, which keeps unbeaten players finite. In the simulation it is refit every 10 comparisons and predicts from the latest fit.

Pair selection (`selection.py`):

- random: a uniform unordered pair.
- uncertainty: the player with the largest Glicko-2 deviation against a random opponent. Pairing the two most uncertain players instead locks into a small set of repeated pairs, because after one round every deviation is nearly equal.
- closest: the pair with the largest p(1 - p), the most even predicted match.
- info gain: the pair whose comparison shrinks total Glicko-2 variance the most. For player i the update gives 1/phi'^2 = 1/phi*^2 + g(phi_j)^2 E(1 - E) whatever the outcome, so the reduction phi*^2 - phi'^2 is known before the comparison. The score is that reduction summed over both players. With equal deviations it picks the same pair as closest.

Accuracy is Kendall tau-b and Spearman rho of the estimated ratings against true skill, every 10 comparisons. Bands are mean plus or minus 1.96 standard errors over seeds. Calibration is computed over every pre-comparison prediction: log loss, Brier score and expected calibration error over 10 equal-width bins.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and make. uv installs Python 3.12 if needed.

```sh
make setup     # uv sync --frozen
make test      # 25 tests, including Hypothesis property tests
make lint      # ruff check and format check
make typecheck # mypy --strict
make results   # 100 seeds, about 1 minute on a 15-core laptop; writes results/results.json and site/card.json
make report    # writes site/index.html from results/results.json
make quick     # 4 seeds, 8 players, into build/quick, for a fast end-to-end check
```

`make results` refuses to run with uncommitted changes under `src/`, `scripts/`, `pyproject.toml` or `uv.lock`, so the commit recorded in the results is the code that produced them. Pass `--allow-dirty` to run anyway; the report then says so.

`site/card.json` is a summary for a portfolio page: the comparisons saved by info-gain pairing, a takeaway sentence and the tau curves, validated by a test against `tests/card.schema.json`.

## Library use

```python
import numpy as np

from pairwise_ratings.glicko2 import Glicko2
from pairwise_ratings.selection import info_gain_pair

rng = np.random.default_rng(0)
model = Glicko2(n_players=4)
for _ in range(12):
    a, b = info_gain_pair(model, rng)
    model.update(a, b)  # a beat b; pass your real result here

print(model.ratings().round(0), model.rds().round(0))
```

## Tests

- Glicko-2 against the paper's example, and the idle-player case.
- Property tests with Hypothesis: a Glicko-2 win never lowers a rating and a loss never raises one; the new deviation never exceeds sqrt(phi^2 + sigma'^2); all values stay finite; win probabilities lie in (0, 1) and p(a, b) + p(b, a) = 1; Elo conserves total rating for any K; a ranking compared with itself has tau 1.
- Bradley-Terry recovers 8 strengths from 20,000 comparisons with correlation above 0.99 and keeps an unbeaten player finite.
- Selection rules, seeded reproducibility, common random numbers across conditions, and counting seeds that never reach the target.
- The report builds with every view and no NaN, `card.json` validates against the schema, and the committed card agrees with the committed results.

## Limitations

- Skills are fixed. Glicko-2's volatility exists for skills that drift, which this simulation does not test.
- Outcomes follow the same logistic model all three systems assume. There is no test of misspecified noise.
- Each comparison is its own Glicko-2 rating period and only the two players update. Glickman intends periods of several games with deviation growth for absent players.
- Only Glicko-2 is paired with the active strategies; Elo has no deviation to drive them.
- One pool size and skill spread. Gains from active selection depend on both.
- Chart text is about 10 px on a 375 px phone because the SVGs scale to the page width.

## License

MIT. See [LICENSE](LICENSE).

# pairwise-ratings

If you can only ask "which of these two is better?", how fast can you recover the true ranking, and which pairs should you ask about? This project compares Elo, Glicko-2 and Bradley-Terry on simulated head-to-head data and tests four rules for picking the next pair.

![Ranking accuracy against number of comparisons for three rating systems and four pair-selection strategies](docs/report.png)

## Highlights

- Choosing pairs by expected information gain reaches the target ranking accuracy (Kendall tau 0.8) in **740 comparisons instead of 1,010** for random pairs. It saves 270 comparisons, 27% fewer.
- The intuitive alternatives backfire: pairing the most uncertain player, or the most even matchup, leaves more runs stuck below the target than random pairing does.
- Bradley-Terry maximum likelihood ranks best (tau 0.841) and has the most accurate win probabilities; Elo ranks worst (0.792).
- Glicko-2 is implemented from Glickman's paper and reproduces its worked example to within 0.01 rating points.
- 25 tests, including Hypothesis property tests; `mypy --strict` and ruff in CI.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and make.

```sh
make setup     # install pinned dependencies
make test
make results   # 100 seeds, about 1 minute on a 15-core laptop
make report    # writes site/index.html, an interactive version of the chart above
make quick     # 4 seeds, 8 players, for a fast end-to-end check
```

As a library:

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

## Results

Each run has 32 players with skill drawn from N(0, 1) on the logit scale, outcomes drawn from the Bradley-Terry model, and 1,280 comparisons. There are 100 seeds, and every condition with the same seed faces the same players. Numbers come from `results/results.json`, produced at commit `bc84bdc`.

**Pair selection** (all using Glicko-2):

| Strategy | Mean tau after 1280 | Mean curve reaches tau 0.8 at | Per-seed median to tau 0.8 (IQR) | Seeds that never reach 0.8 |
|---|---|---|---|---|
| Info gain | 0.857 | 740 | 670 (472.5 to 847.5) | 6 of 100 |
| Closest | 0.827 | 870 | 650 (447.5 to 912.5) | 20 of 100 |
| Random | 0.819 | 1010 | 700 (560 to 885) | 9 of 100 |
| Uncertainty | 0.794 | never | 640 (495 to 840) | 25 of 100 |

At 1,280 comparisons the 95% confidence intervals don't overlap: info gain 0.849 to 0.865, random 0.812 to 0.827. The per-seed medians are close. What separates the strategies is how many seeds stall.

**Rating systems** (all using random pairs):

| System | Mean tau after 1280 | Log loss | Brier | ECE | Mean abs. error vs true p |
|---|---|---|---|---|---|
| Bradley-Terry MLE | 0.841 | 0.5764 | 0.1963 | 0.0266 | 0.0877 |
| Glicko-2 | 0.819 | 0.5798 | 0.1975 | 0.0367 | 0.0926 |
| Elo (K = 32) | 0.792 | 0.5814 | 0.1988 | 0.0261 | 0.1035 |

The best possible log loss, scoring the true probabilities, is 0.5376. Glicko-2 and Bradley-Terry are overconfident at the extremes (predicted 0.06, observed about 0.09 to 0.10); Elo is slightly underconfident there.

**Glicko-2 check.** On Glickman's worked example (player 1500 / 200 / 0.06, tau 0.5, one win and two losses) this code gives rating 1464.0507, RD 151.5165, volatility 0.059996, against the paper's 1464.06, 151.52 and 0.05999. The 0.0093-point rating gap comes from the paper rounding intermediate steps.

## How it works

Rating systems (`src/pairwise_ratings/`):

- `elo.py`: expected score 1 / (1 + 10^((Rb - Ra) / 400)), K = 32.
- `glicko2.py`: steps 1 to 8 of Glickman, "Example of the Glicko-2 system" (2013), with the Illinois regula falsi solver for volatility. Predictions use the combined deviation g(sqrt(phi_a^2 + phi_b^2)) from Glickman (1999).
- `bradley_terry.py`: maximum likelihood via Hunter's MM algorithm (Hunter 2004). Each player gets one virtual win and one virtual loss against a fixed reference player, which keeps unbeaten players finite. Refit every 10 comparisons.

Pair selection (`selection.py`):

- **random**: a uniform unordered pair.
- **uncertainty**: the player with the largest Glicko-2 deviation against a random opponent.
- **closest**: the pair with the largest p(1 - p), the most even predicted match.
- **info gain**: the pair whose comparison shrinks total Glicko-2 variance the most. For player i the update gives 1/phi'^2 = 1/phi*^2 + g(phi_j)^2 E(1 - E) whatever the outcome, so the reduction is known before the match is played. The score sums it over both players. With equal deviations it picks the same pair as closest.

Accuracy is Kendall tau-b and Spearman rho of estimated ratings against true skill, measured every 10 comparisons; bands are mean ± 1.96 standard errors over seeds. Calibration uses every pre-comparison prediction: log loss, Brier score, and expected calibration error over 10 equal-width bins.

## Limitations

- Skills are fixed, so Glicko-2's volatility, which exists for drifting skill, goes untested.
- Outcomes follow the same logistic model all three systems assume; misspecified noise is not tested.
- Each comparison is its own Glicko-2 rating period, unlike Glickman's multi-game periods.
- One pool size (32) and skill spread; gains from active selection depend on both.

## License

MIT. See [LICENSE](LICENSE).

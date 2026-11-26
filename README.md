# pairwise-ratings

If all you can ask is "which of these two is better?", how fast can you recover the true ranking, and which pair should you ask next? Simulations of Elo, Glicko-2 and Bradley-Terry ratings and four pair-picking rules.

![Comparisons to reach Kendall tau 0.8: info-gain vs random pairing](docs/report.png)

## Highlights

Report: https://allenwang0.github.io/pairwise-ratings/

- With Glicko-2 ratings, averaged over 100 simulated runs of 32 players, the ranking-accuracy curve reaches Kendall tau 0.8 after **740 comparisons with information-gain pairing instead of 1,010** with random pairs: 270 (27 %) fewer. The typical run gains less: the median run gets there in 670 vs 700 comparisons, about 4 %.
- Changing the rater helps about as much: Bradley-Terry with random pairs reaches the same mean accuracy after 780 comparisons.
- The intuitive shortcuts are riskier, not uniformly slower: asking about the least-known player or the most even matchup leaves 25 and 20 of 100 runs short of the target, against 9 for random pairing.
- After 1,280 random comparisons, Bradley-Terry ranks best (rank agreement 0.841 out of 1) and Elo worst (0.792).
- Glicko-2 reproduces the worked example in Glickman's paper to within 0.01 rating points.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and make.

```sh
make setup     # install pinned dependencies
make test
make results   # 100 simulated runs, about 1 minute
make report    # interactive version of the chart above, written to site/index.html
```

## How it works

- Each run has 32 players with hidden skills and 1,280 simulated matches; each rater estimates the skills from the results.
- Accuracy is Kendall tau: pairs ranked in the right order, rescaled so 1 is perfect and 0 is chance. Target: 0.8.
- The information-gain rule picks the pair whose result would shrink Glicko-2's total uncertainty the most.

## Limitations

- Skills stay fixed; improving or declining players are untested.
- Match outcomes follow the same model the rating systems assume; real data is noisier.
- One pool size (32 players), and pair rules were tested with Glicko-2 only.

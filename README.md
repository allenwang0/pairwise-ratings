# pairwise-ratings

If all you can ask is "which of these two is better?", how fast can you recover the true ranking, and which pair should you ask about next? This project answers that with simulations of Elo, Glicko-2 and Bradley–Terry ratings and four rules for picking pairs. All players are simulated, not real.

**[Read the report →](https://allenwang0.github.io/pairwise-ratings/)** (per-run distributions, bootstrap interval, every number linked to its source)

![Ranking accuracy against number of comparisons for Elo, Glicko-2 and Bradley–Terry](docs/report.png)

## Highlights

<!-- results:start -->
- **Bradley–Terry recovers the true ranking best**: Kendall τ 0.841 after 1,280 random comparisons, vs 0.819 for Glicko-2 and 0.792 for Elo (mean of 100 simulated runs of 32 players; outcomes come from the Bradley–Terry model itself).
- **Choosing pairs by information gain ends more accurate but not detectably sooner**: a median of 695 comparisons to reach τ 0.8 vs 720 for random pairs (95% bootstrap interval for the saving: −50 to 125); 6 vs 9 of 100 runs never get there, and final τ is 0.857 vs 0.819 (Glicko-2 ratings).
- **The intuitive shortcuts stall**: pairing the least-known player leaves 25 of 100 runs short of the target, the most even match 20.
- **Glicko-2 reproduces Glickman's worked example**: rating 1464.05 vs 1464.06 in the paper.
<!-- results:end -->

## Run it

Requires [uv](https://docs.astral.sh/uv/) and make.

```sh
make setup     # install pinned dependencies
make test
make results   # 100 simulated runs, about 1 minute
make report    # rebuild site/index.html and this README's highlights
```

## How it works

- Each run hides a skill for every player, then simulates matches whose outcomes follow the Bradley–Terry model.
- Elo, Glicko-2 and Bradley–Terry each estimate the skills from who won.
- Ranking accuracy is Kendall τ against the hidden skills: 1 is perfect, 0 is chance.
- The information-gain rule picks the pair whose result would shrink Glicko-2's total uncertainty the most, an amount known before the match is played.

## Limitations

- Skills stay fixed, so tracking players who improve or decline is untested.
- Outcomes follow the model Bradley–Terry assumes, which favours it; real data is noisier.
- One pool size; the value of smart pairing depends on it.

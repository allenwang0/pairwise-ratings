import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pairwise_ratings import metrics
from pairwise_ratings.experiment import comparisons_to_target, run_condition
from pairwise_ratings.simulate import Config, simulate

SMALL = Config(n_players=8, budget=200, checkpoint_every=20)


def test_same_seed_gives_same_run() -> None:
    a = simulate("glicko2", "info_gain", SMALL, 5)
    b = simulate("glicko2", "info_gain", SMALL, 5)
    assert np.array_equal(a.tau, b.tau)
    assert np.array_equal(a.outcome, b.outcome)


def test_conditions_share_players_for_a_seed() -> None:
    # Common random numbers: every condition with seed s faces the same skills, so differences
    # between strategies are not differences between draws of players.
    assert np.array_equal(
        simulate("elo", "random", SMALL, 9).true_p, simulate("glicko2", "random", SMALL, 9).true_p
    )


def test_ranking_improves_with_comparisons() -> None:
    summary = run_condition("glicko2", "random", SMALL, range(10))
    tau = summary["tau"]["mean"]
    assert tau[-1] > tau[0]
    assert summary["final_tau"] > 0.5


def test_non_random_strategy_needs_deviations() -> None:
    with pytest.raises(ValueError):
        simulate("elo", "info_gain", SMALL, 0)


def test_seeds_that_never_reach_target_are_counted() -> None:
    tau = np.array([[0.5, 0.85], [0.5, 0.6]])
    out = comparisons_to_target(tau, np.array([10, 20]))
    assert out["never"] == 1
    assert out["median"] == 20.0
    assert out["mean_curve_crossing"] is None


@given(st.lists(st.floats(-5, 5), min_size=3, max_size=30, unique=True))
def test_truth_ranks_itself_perfectly(values: list[float]) -> None:
    x = np.array(values)
    assert metrics.kendall(x, x) == pytest.approx(1.0)
    assert metrics.kendall(-x, x) == pytest.approx(-1.0)


def test_calibrated_predictions_have_small_ece() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(float)
    assert metrics.ece(p, y) < 0.01
    assert metrics.ece(np.full_like(p, 0.9), y) > 0.3

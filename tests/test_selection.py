import numpy as np
import pytest

from pairwise_ratings.glicko2 import Glicko2
from pairwise_ratings.selection import (
    closest_pair,
    info_gain_pair,
    info_gain_scores,
    random_pair,
    select,
    uncertainty_pair,
)


def test_info_gain_prefers_uncertain_even_matches() -> None:
    model = Glicko2(3)
    model.mu[:] = [0.0, 0.1, 3.0]
    model.phi[:] = [0.3, 1.5, 1.5]
    scores = info_gain_scores(model)
    # Player 1 is uncertain and evenly matched with 0; player 2 is uncertain but far away.
    assert scores[0, 1] > scores[0, 2]
    assert info_gain_pair(model, np.random.default_rng(0)) == tuple(
        int(i) for i in np.unravel_index(np.argmax(np.triu(scores, 1)), scores.shape)
    )


def test_info_gain_equals_closest_when_deviations_are_equal() -> None:
    model = Glicko2(6)
    model.mu[:] = [0.0, 0.9, -1.4, 2.2, 0.35, -0.2]
    rng = np.random.default_rng(1)
    assert info_gain_pair(model, rng) == closest_pair(model.win_prob_matrix(), rng)


def test_uncertainty_always_includes_the_most_uncertain_player() -> None:
    model = Glicko2(5)
    model.phi[:] = [0.5, 0.5, 2.0, 0.5, 0.5]
    rng = np.random.default_rng(2)
    pairs = {uncertainty_pair(model, rng) for _ in range(50)}
    assert all(2 in p for p in pairs)
    assert len(pairs) > 1


def test_random_pair_is_ordered_and_distinct() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, b = random_pair(4, rng)
        assert 0 <= a < b < 4


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        select("greedy", Glicko2(3), np.random.default_rng(0))

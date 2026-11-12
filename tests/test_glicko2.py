import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pairwise_ratings.glicko2 import Glicko2, Rating, rate, update_scaled


def test_matches_glickman_worked_example() -> None:
    """Glickman (2013), section "Example": the reference numbers any implementation must hit."""
    player = Rating(1500, 200, 0.06)
    games = [(Rating(1400, 30), 1.0), (Rating(1550, 100), 0.0), (Rating(1700, 300), 0.0)]
    out = rate(player, games, tau=0.5)
    assert out.rating == pytest.approx(1464.06, abs=0.01)
    assert out.rd == pytest.approx(151.52, abs=0.01)
    assert out.volatility == pytest.approx(0.05999, abs=1e-5)


def test_idle_player_only_gains_deviation() -> None:
    mu, phi, sigma = update_scaled(0.3, 0.5, 0.06, [])
    assert (mu, sigma) == (0.3, 0.06)
    assert phi == pytest.approx(math.sqrt(0.5**2 + 0.06**2))


ratings = st.floats(-4.0, 4.0)
deviations = st.floats(0.05, 2.5)
volatilities = st.floats(0.01, 0.2)


@given(ratings, deviations, volatilities, ratings, deviations)
def test_winner_rises_loser_falls_and_deviation_is_bounded(
    mu: float, phi: float, sigma: float, mu_j: float, phi_j: float
) -> None:
    # A win can only be good news and a loss only bad news; the update can never add more
    # uncertainty than the period's inflation sqrt(phi^2 + sigma'^2).
    for score in (1.0, 0.0):
        new_mu, new_phi, new_sigma = update_scaled(mu, phi, sigma, [(mu_j, phi_j, score)])
        assert all(math.isfinite(x) for x in (new_mu, new_phi, new_sigma))
        assert new_mu >= mu if score == 1.0 else new_mu <= mu
        assert new_phi <= math.sqrt(phi**2 + new_sigma**2) + 1e-12
        assert new_sigma > 0


@given(st.lists(st.tuples(st.integers(0, 4), st.integers(0, 4)), max_size=60))
def test_predictions_are_complementary_probabilities(games: list[tuple[int, int]]) -> None:
    model = Glicko2(5)
    for a, b in games:
        if a != b:
            model.update(a, b)
    p = model.win_prob_matrix()
    assert ((p > 0) & (p < 1)).all()
    assert p + p.T == pytest.approx(1.0)
    assert model.predict(0, 1) == pytest.approx(p[0, 1])

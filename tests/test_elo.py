import pytest
from hypothesis import given
from hypothesis import strategies as st

from pairwise_ratings.elo import Elo


def test_even_match_moves_half_of_k() -> None:
    model = Elo(2)
    model.update(0, 1)
    assert model.r.tolist() == [1516.0, 1484.0]


@given(
    st.floats(1.0, 64.0),
    st.lists(st.tuples(st.integers(0, 3), st.integers(0, 3)), max_size=80),
)
def test_total_rating_is_conserved(k: float, games: list[tuple[int, int]]) -> None:
    # Elo is zero-sum: points the winner gains come from the loser, so the pool mean never drifts.
    model = Elo(4, k=k)
    for a, b in games:
        if a != b:
            model.update(a, b)
    assert model.r.sum() == pytest.approx(4 * 1500.0)
    p = model.win_prob_matrix()
    assert p + p.T == pytest.approx(1.0)

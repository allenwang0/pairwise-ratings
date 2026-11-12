import numpy as np

from pairwise_ratings.bradley_terry import BradleyTerry, fit


def test_recovers_strengths_from_many_comparisons() -> None:
    rng = np.random.default_rng(7)
    skill = rng.normal(0, 1, 8)
    model = BradleyTerry(8)
    for _ in range(20_000):
        a, b = rng.choice(8, 2, replace=False)
        a_wins = rng.random() < 1 / (1 + np.exp(skill[b] - skill[a]))
        model.update(a, b) if a_wins else model.update(b, a)
    model.refit()
    est = model.scores()
    assert np.corrcoef(est - est.mean(), skill - skill.mean())[0, 1] > 0.99


def test_undefeated_player_stays_finite() -> None:
    # Plain MLE sends an unbeaten player to +infinity; the virtual games keep it finite and on top.
    wins = np.zeros((3, 3))
    wins[0, 1] = wins[0, 2] = 5
    theta = fit(wins)
    assert np.isfinite(theta).all()
    assert theta.argmax() == 0

import numpy as np
import pytest

from emoscript.axis import (
    Calibrator,
    accuracy,
    auc,
    brier,
    expected_calibration_error,
    fit_diff_axis,
    fit_mean_axis,
    fit_multi_axis,
    kmeans,
    l2_normalize,
    log_loss,
    score,
    score_per_axis,
)


def test_l2_normalize_rows_are_unit_norm():
    X = np.array([[3.0, 4.0], [0.0, 5.0]])
    assert np.allclose(np.linalg.norm(l2_normalize(X), axis=1), 1.0)
    assert np.isclose(np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))), 1.0)


def test_diff_axis_separates_clusters():
    rng = np.random.default_rng(0)
    pos = rng.normal(loc=[1, 0, 0, 0], scale=0.05, size=(30, 4))
    neg = rng.normal(loc=[-1, 0, 0, 0], scale=0.05, size=(30, 4))
    axis = fit_diff_axis(l2_normalize(pos), l2_normalize(neg))
    s_pos, s_neg = score(l2_normalize(pos), axis), score(l2_normalize(neg), axis)
    assert s_pos.mean() > s_neg.mean()
    assert auc(s_pos, s_neg) == 1.0


def test_mean_axis_is_unit_norm():
    E = l2_normalize(np.random.default_rng(1).normal(size=(10, 8)))
    assert np.isclose(np.linalg.norm(fit_mean_axis(E)), 1.0)


def test_auc_matches_brute_force_and_handles_ties():
    pos = np.array([0.9, 0.8, 0.5])
    neg = np.array([0.5, 0.1])
    wins = sum(p > n for p in pos for n in neg)
    ties = sum(p == n for p in pos for n in neg)
    assert auc(pos, neg) == pytest.approx((wins + 0.5 * ties) / 6)
    with pytest.raises(ValueError):
        auc(np.array([]), neg)


def test_kmeans_recovers_well_separated_clusters():
    rng = np.random.default_rng(0)
    centers = l2_normalize(rng.normal(size=(3, 16)))
    X = l2_normalize(np.vstack([c + rng.normal(scale=0.05, size=(20, 16)) for c in centers]))
    truth = np.repeat(np.arange(3), 20)
    C, labels, inertia = kmeans(X, 3, seed=0)
    assert C.shape == (3, 16)
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0)
    # same partition up to relabeling
    assert len({tuple(labels[truth == g]) for g in range(3)}) == 3
    for g in range(3):
        assert len(set(labels[truth == g])) == 1
    assert inertia < 20 * 3 * 16 * 0.05**2 * 2
    with pytest.raises(ValueError):
        kmeans(X, 0)


def test_multi_axis_k1_equals_diff_axis_and_k2_rescues_second_mode():
    rng = np.random.default_rng(3)
    d = 24
    neg_c = l2_normalize(rng.normal(size=d))
    mode_a = l2_normalize(rng.normal(size=d))
    mode_b = l2_normalize(rng.normal(size=d))
    neg = l2_normalize(neg_c + rng.normal(scale=0.1, size=(40, d)))
    pos_a = l2_normalize(mode_a + rng.normal(scale=0.1, size=(40, d)))
    pos_b = l2_normalize(mode_b + rng.normal(scale=0.1, size=(10, d)))  # minority flavour
    pos = np.vstack([pos_a, pos_b])

    axes1, labels1 = fit_multi_axis(pos, neg, 1)
    assert axes1.shape == (1, d) and set(labels1) == {0}
    assert np.allclose(axes1[0], fit_diff_axis(pos, neg))

    axes2, labels2 = fit_multi_axis(pos, neg, 2, seed=0)
    assert axes2.shape == (2, d) and set(labels2) == {0, 1}
    # the minority flavour scores better once it has its own axis
    s1 = score(pos_b, axes1)
    s2 = score(pos_b, axes2)
    assert s2.mean() > s1.mean()
    assert score_per_axis(pos_b, axes2).shape == (10, 2)
    assert np.allclose(score(pos_b, axes2), score_per_axis(pos_b, axes2).max(axis=1))


def test_calibrator_is_monotone_bounded_and_serializable():
    rng = np.random.default_rng(0)
    s_pos = rng.normal(0.3, 0.05, 100)
    s_neg = rng.normal(0.1, 0.05, 100)
    cal = Calibrator().fit(s_pos, s_neg)
    grid = np.linspace(-0.2, 0.6, 50)
    p = cal.predict_proba(grid)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= 0)  # non-decreasing in the score
    mid = (grid > 0.05) & (grid < 0.35)
    assert np.all(np.diff(p[mid]) > 0)  # strictly increasing where the data lives
    assert cal.predict_proba(np.array([0.3]))[0] > 0.9
    assert cal.predict_proba(np.array([0.1]))[0] < 0.1
    thr = cal.score_at(0.5)
    assert 0.1 < thr < 0.3
    assert cal.predict_proba(np.array([thr]))[0] == pytest.approx(0.5, abs=1e-6)
    blob = cal.to_arrays()
    cal2 = Calibrator.from_arrays(blob)
    assert np.allclose(cal2.predict_proba(grid), p)


def test_calibrator_survives_perfectly_separable_data():
    s_pos = np.linspace(0.5, 0.9, 20)
    s_neg = np.linspace(-0.2, 0.2, 20)
    cal = Calibrator().fit(s_pos, s_neg)
    p = cal.predict_proba(np.concatenate([s_pos, s_neg]))
    assert np.all(np.isfinite(p))
    assert accuracy(p, np.r_[np.ones(20), np.zeros(20)]) == 1.0


def test_metric_sanity():
    y = np.array([1, 1, 0, 0])
    perfect = np.array([1.0, 1.0, 0.0, 0.0])
    coin = np.full(4, 0.5)
    assert log_loss(perfect, y) < log_loss(coin, y)
    assert brier(perfect, y) == 0.0 and brier(coin, y) == 0.25
    assert accuracy(perfect, y) == 1.0
    assert expected_calibration_error(perfect, y) == 0.0
    assert expected_calibration_error(coin, y) == pytest.approx(0.0)

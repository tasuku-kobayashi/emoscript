import numpy as np

from copyaxis.axis import auc, fit_diff_axis, fit_mean_axis, l2_normalize, score


def test_l2_normalize_rows_are_unit_norm():
    X = np.array([[3.0, 4.0], [0.0, 5.0]])
    assert np.allclose(np.linalg.norm(l2_normalize(X), axis=1), 1.0)


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

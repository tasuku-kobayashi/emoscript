"""Semantic-axis construction, scoring, calibration and metrics.

The core idea: if "great ad copy" occupies a coherent direction in embedding
space, you can estimate that direction from a corpus of famous taglines and
score new copy by cosine similarity against it.

Estimators
----------
- fit_mean_axis: centroid of the positive (great-copy) embeddings.
  Simple, but mostly captures "well-formed marketing English" and fails to
  separate great copy from *plausible-sounding bland* copy.
- fit_diff_axis: positive centroid minus negative centroid.
  Subtracting the negative direction isolates what is *distinctive* about
  great copy. This is the recommended default (k = 1).
- fit_multi_axis: spherical k-means over the positives, then one diff axis per
  cluster. A line passes if it is close to *any* flavour of great copy
  (terse, informational, humorous...). Scoring takes the max over axes.

Calibration
-----------
- Calibrator: a one-feature logistic regression that maps a raw cosine score
  to P(great copy). Fit it on *out-of-fold* scores (see emoscript.pipeline) so
  the probabilities are honest.

Everything here is plain numpy; no model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- basics
def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows (or a single vector)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return X / (np.linalg.norm(X) + _EPS)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + _EPS)


def fit_mean_axis(E_pos: np.ndarray) -> np.ndarray:
    """Unit vector pointing at the centroid of positive examples."""
    return l2_normalize(np.asarray(E_pos).mean(axis=0))


def fit_diff_axis(E_pos: np.ndarray, E_neg: np.ndarray) -> np.ndarray:
    """Unit vector: positive centroid minus negative centroid.

    Works best when negatives include "hard negatives": fluent, ad-like but
    generic copy that a naive centroid would incorrectly score highly.
    """
    return l2_normalize(np.asarray(E_pos).mean(axis=0) - np.asarray(E_neg).mean(axis=0))


def score(E: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Cosine of each row of E against the axis.

    ``axes`` may be a single unit vector ``(d,)`` or a stack ``(k, d)``; with a
    stack the score of a row is the maximum over axes ("close to any flavour").
    E must be L2-normalized.
    """
    E = np.atleast_2d(np.asarray(E, dtype=np.float64))
    axes = np.asarray(axes, dtype=np.float64)
    if axes.ndim == 1:
        return E @ axes
    return (E @ axes.T).max(axis=1)


def score_per_axis(E: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Cosine of each row of E against every axis: shape ``(n, k)``."""
    E = np.atleast_2d(np.asarray(E, dtype=np.float64))
    axes = np.atleast_2d(np.asarray(axes, dtype=np.float64))
    return E @ axes.T


# --------------------------------------------------------------------------- clustering
def kmeans(
    X: np.ndarray,
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 100,
    seed: int = 0,
    spherical: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Small k-means (k-means++ init, several restarts, best inertia wins).

    With ``spherical=True`` the centroids are re-normalized after every update,
    which is the right choice for unit-norm embeddings compared by cosine.

    Returns ``(centroids (k, d), labels (n,), inertia)``.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None

    for _ in range(n_init):
        # k-means++ seeding
        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rng.integers(n)]
        for j in range(1, k):
            d2 = ((X[:, None, :] - centers[None, :j, :]) ** 2).sum(-1).min(axis=1)
            total = d2.sum()
            if total <= 0:
                centers[j] = X[rng.integers(n)]
            else:
                centers[j] = X[rng.choice(n, p=d2 / total)]
        if spherical:
            centers = l2_normalize(centers)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = d2.argmin(axis=1)
            for j in range(k):
                members = X[new_labels == j]
                if len(members) == 0:  # re-seed an empty cluster at the farthest point
                    members = X[[d2.min(axis=1).argmax()]]
                    new_labels[d2.min(axis=1).argmax()] = j
                centers[j] = members.mean(axis=0)
            if spherical:
                centers = l2_normalize(centers)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (centers.copy(), labels.copy(), inertia)

    assert best is not None
    return best


def fit_multi_axis(
    E_pos: np.ndarray,
    E_neg: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """k diff-axes, one per k-means cluster of the positives.

    Returns ``(axes (k, d), labels (n_pos,))``. ``k=1`` reduces exactly to
    :func:`fit_diff_axis`.
    """
    E_pos = np.asarray(E_pos, dtype=np.float64)
    if k == 1:
        return fit_diff_axis(E_pos, E_neg)[None, :], np.zeros(len(E_pos), dtype=int)
    centers, labels, _ = kmeans(E_pos, k, seed=seed, n_init=n_init)
    neg_centroid = np.asarray(E_neg, dtype=np.float64).mean(axis=0)
    axes = l2_normalize(centers - neg_centroid[None, :])
    return axes, labels


# --------------------------------------------------------------------------- metrics
def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Rank AUC: P(random positive scores above random negative), ties count half."""
    p = np.asarray(pos_scores, dtype=np.float64)[:, None]
    n = np.asarray(neg_scores, dtype=np.float64)[None, :]
    if p.size == 0 or n.size == 0:
        raise ValueError("auc needs at least one positive and one negative score")
    return float((p > n).mean() + 0.5 * (p == n).mean())


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(((p - y) ** 2).mean())


def accuracy(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(((p >= threshold) == (y >= 0.5)).mean())


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 5) -> float:
    """ECE with equal-width probability bins (lower is better)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


# --------------------------------------------------------------------------- calibration
@dataclass
class Calibrator:
    """Logistic regression on one or more raw axis scores -> P(great copy).

    Features are standardized with the training mean/std before the logistic
    fit (Newton's method with a small L2 penalty, so perfectly separable data
    does not blow up).
    """

    w: np.ndarray = field(default_factory=lambda: np.zeros(1))
    b: float = 0.0
    mu: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sd: np.ndarray = field(default_factory=lambda: np.ones(1))
    l2: float = 1e-2

    @staticmethod
    def _as_2d(S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=np.float64)
        return S[:, None] if S.ndim == 1 else S

    def fit(self, S_pos: np.ndarray, S_neg: np.ndarray, *, max_iter: int = 100) -> Calibrator:
        Xp, Xn = self._as_2d(S_pos), self._as_2d(S_neg)
        X = np.vstack([Xp, Xn])
        y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + _EPS
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        d = Z.shape[1]
        theta = np.zeros(d)
        reg = np.full(d, self.l2)
        reg[-1] = 0.0  # no penalty on the intercept
        for _ in range(max_iter):
            p = _sigmoid(Z @ theta)
            g = Z.T @ (p - y) + reg * theta
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg) + 1e-9 * np.eye(d)
            step = np.linalg.solve(H, g)
            theta -= step
            if np.abs(step).max() < 1e-9:
                break
        self.w, self.b = theta[:-1], float(theta[-1])
        return self

    def predict_proba(self, S: np.ndarray) -> np.ndarray:
        Z = (self._as_2d(S) - self.mu) / self.sd
        return _sigmoid(Z @ self.w + self.b)

    def score_at(self, prob: float) -> float:
        """Raw (one-feature) score at which the calibrated probability equals ``prob``."""
        if self.w.shape[0] != 1 or self.w[0] == 0:
            raise ValueError("score_at is only defined for a one-feature calibrator with non-zero slope")
        logit = np.log(prob / (1 - prob))
        return float((logit - self.b) / self.w[0] * self.sd[0] + self.mu[0])

    # ---- (de)serialization helpers for npz ----
    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "calib_w": np.asarray(self.w, dtype=np.float64),
            "calib_b": np.asarray(self.b, dtype=np.float64),
            "calib_mu": np.asarray(self.mu, dtype=np.float64),
            "calib_sd": np.asarray(self.sd, dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, blob) -> Calibrator:
        return cls(
            w=np.asarray(blob["calib_w"], dtype=np.float64).reshape(-1),
            b=float(blob["calib_b"]),
            mu=np.asarray(blob["calib_mu"], dtype=np.float64).reshape(-1),
            sd=np.asarray(blob["calib_sd"], dtype=np.float64).reshape(-1),
        )

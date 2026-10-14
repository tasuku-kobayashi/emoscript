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


def weighted_mean(E: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    if weights is None:
        return E.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(E),) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be a non-negative vector with one entry per row and a positive sum")
    return (E * w[:, None]).sum(axis=0) / w.sum()


def family_weights(families: list) -> np.ndarray:
    """Per-row weights that give every family (e.g. negative kind) equal total weight.

    The diff axis subtracts a plain mean, so whichever negative family has
    the most lines decides the direction. Balancing families keeps a large
    new family from drowning out the others.
    """
    fam = np.asarray(families)
    counts = {f: int((fam == f).sum()) for f in np.unique(fam)}
    return np.array([1.0 / (len(counts) * counts[f]) for f in fam]) * len(fam)


def fit_diff_axis(E_pos: np.ndarray, E_neg: np.ndarray, neg_weights: np.ndarray | None = None) -> np.ndarray:
    """Unit vector: positive centroid minus (optionally weighted) negative centroid.

    Works best when negatives include "hard negatives": fluent, ad-like but
    generic copy that a naive centroid would incorrectly score highly.
    """
    return l2_normalize(np.asarray(E_pos).mean(axis=0) - weighted_mean(E_neg, neg_weights))


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
            # re-seed each empty cluster with a distinct point that is far from its centre
            dmin = d2.min(axis=1).copy()
            for j in range(k):
                if not np.any(new_labels == j):
                    idx = int(dmin.argmax())
                    new_labels[idx] = j
                    dmin[idx] = -np.inf
            for j in range(k):
                members = X[new_labels == j]
                if len(members):
                    centers[j] = members.mean(axis=0)
                else:  # only with fewer distinct points than k: park the centre on a random point
                    centers[j] = X[rng.integers(n)]
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
    neg_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """k diff-axes, one per k-means cluster of the positives.

    Each axis is ``normalize(mean(cluster) - mean(negatives))`` using the *raw*
    cluster mean, not the unit-length spherical centre: unit-norm embeddings
    share a large common direction (mean pairwise cosine ~0.8 for e5), so a
    unit centre minus a raw mean of length ~0.9 would leave every axis leaning
    along that common direction and inflate all scores. ``k=1`` reduces
    exactly to :func:`fit_diff_axis`.

    Returns ``(axes (k, d), labels (n_pos,))``.
    """
    E_pos = np.asarray(E_pos, dtype=np.float64)
    E_neg = np.asarray(E_neg, dtype=np.float64)
    if k == 1:
        return fit_diff_axis(E_pos, E_neg, neg_weights)[None, :], np.zeros(len(E_pos), dtype=int)
    _, labels, _ = kmeans(E_pos, k, seed=seed, n_init=n_init)
    neg_centroid = weighted_mean(E_neg, neg_weights)
    global_mean = E_pos.mean(axis=0)
    means = np.vstack(
        [E_pos[labels == c].mean(axis=0) if np.any(labels == c) else global_mean for c in range(k)]
    )
    axes = l2_normalize(means - neg_centroid[None, :])
    return axes, labels


# --------------------------------------------------------------------------- metrics
def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Rank AUC: P(random positive scores above random negative), ties count half."""
    p = np.asarray(pos_scores, dtype=np.float64)[:, None]
    n = np.asarray(neg_scores, dtype=np.float64)[None, :]
    if p.size == 0 or n.size == 0:
        raise ValueError("auc needs at least one positive and one negative score")
    if np.isnan(p).any() or np.isnan(n).any():
        raise ValueError("auc: scores contain NaN")
    return float((p > n).mean() + 0.5 * (p == n).mean())


def _resample(idx_groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap: resample whole groups with replacement, return row indices."""
    pick = rng.integers(len(idx_groups), size=len(idx_groups))
    return np.concatenate([idx_groups[g] for g in pick])


def _groups_of(labels, n: int) -> list[np.ndarray]:
    if labels is None:
        return [np.array([i]) for i in range(n)]
    labels = np.asarray(labels)
    return [np.flatnonzero(labels == g) for g in np.unique(labels)]


def auc_bootstrap_ci(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    *,
    pos_groups=None,
    neg_groups=None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for :func:`auc`.

    ``pos_groups`` / ``neg_groups`` (e.g. brand per tagline) switch to a cluster
    bootstrap that resamples whole groups, so several lines from one brand do
    not count as independent evidence. With a saturated AUC of 1.0 the
    percentile interval collapses to [1, 1]; read that as "no error seen in
    this sample", not as certainty.
    """
    p = np.asarray(pos_scores, dtype=np.float64)
    n = np.asarray(neg_scores, dtype=np.float64)
    pg, ng = _groups_of(pos_groups, len(p)), _groups_of(neg_groups, len(n))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = auc(p[_resample(pg, rng)], n[_resample(ng, rng)])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_auc_bootstrap_ci(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for AUC when positives and negatives come in matched pairs
    (row i of each array belongs together): pairs are resampled as units."""
    p = np.asarray(pos_scores, dtype=np.float64)
    n = np.asarray(neg_scores, dtype=np.float64)
    if len(p) != len(n):
        raise ValueError("paired bootstrap needs equally long, aligned score arrays")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(len(p), size=len(p))
        vals[i] = auc(p[idx], n[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (e.g. a paired win rate)."""
    if n <= 0:
        raise ValueError("wilson_interval needs n > 0")
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


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
    keep = ~np.isnan(p)
    p, y = p[keep], y[keep]
    if p.size == 0:
        return float("nan")
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
    fit (Newton's method with an L2 penalty on the slope, so perfectly
    separable data does not blow up). With ``balanced=True`` (default) the two
    classes get equal total weight, so ``p = 0.5`` means "as likely great as
    bland under a 50/50 prior" regardless of how many positives and negatives
    the corpus happens to contain; move the decision threshold to encode a
    different prior. The default penalty was chosen by cross-validation on the
    training split (``experiments/select_settings.py``).
    """

    w: np.ndarray = field(default_factory=lambda: np.zeros(1))
    b: float = 0.0
    mu: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sd: np.ndarray = field(default_factory=lambda: np.ones(1))
    l2: float = 1.0
    balanced: bool = True

    @staticmethod
    def _as_2d(S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=np.float64)
        return S[:, None] if S.ndim == 1 else S

    def fit(self, S_pos: np.ndarray, S_neg: np.ndarray, *, max_iter: int = 100) -> Calibrator:
        Xp, Xn = self._as_2d(S_pos), self._as_2d(S_neg)
        if len(Xp) < 2 or len(Xn) < 2:
            raise ValueError("Calibrator.fit needs at least two positive and two negative scores")
        if np.isnan(Xp).any() or np.isnan(Xn).any():
            raise ValueError("Calibrator.fit: scores contain NaN")
        X = np.vstack([Xp, Xn])
        y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
        if self.balanced:
            wts = np.where(y == 1, len(X) / (2 * len(Xp)), len(X) / (2 * len(Xn)))
        else:
            wts = np.ones(len(X))
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + _EPS
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        d = Z.shape[1]
        theta = np.zeros(d)
        reg = np.full(d, self.l2)
        reg[-1] = 0.0  # no penalty on the intercept
        for _ in range(max_iter):
            p = _sigmoid(Z @ theta)
            g = Z.T @ (wts * (p - y)) + reg * theta
            H = (Z * (wts * p * (1 - p))[:, None]).T @ Z + np.diag(reg) + 1e-9 * np.eye(d)
            step = np.linalg.solve(H, g)
            theta -= step
            if np.abs(step).max() < 1e-9:
                break
        self.w, self.b = theta[:-1], float(theta[-1])
        return self

    def predict_proba(self, S: np.ndarray) -> np.ndarray:
        Z = (self._as_2d(S) - self.mu) / self.sd
        return _sigmoid(Z @ self.w + self.b)

    @property
    def slope(self) -> float:
        """Slope on the standardized score; must be > 0 for 'higher score = more great'."""
        return float(self.w[0]) if self.w.shape[0] == 1 else float("nan")

    def score_at(self, prob: float) -> float:
        """Raw (one-feature) score at which the calibrated probability equals ``prob``.

        Returns NaN when the calibrator is degenerate (zero or negative slope,
        e.g. identical scores for every line) instead of raising.
        """
        if not 0 < prob < 1:
            raise ValueError("prob must be strictly between 0 and 1")
        if self.w.shape[0] != 1 or not self.w[0] > 1e-9:
            return float("nan")
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

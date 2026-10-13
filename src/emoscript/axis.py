"""Semantic-axis construction and scoring.

The core idea: if 'great ad copy' occupies a coherent direction in embedding
space, you can estimate that direction from a corpus of famous taglines and
score new copy by cosine similarity against it.

Two estimators are provided:

- fit_mean_axis: centroid of the positive (great-copy) embeddings.
  Simple, but mostly captures 'well-formed marketing English' and fails to
  separate great copy from *plausible-sounding bland* copy.
- fit_diff_axis: positive centroid minus negative centroid.
  Subtracting the negative direction isolates what is *distinctive* about
  great copy. This is the recommended default.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def fit_mean_axis(E_pos: np.ndarray) -> np.ndarray:
    """Unit vector pointing at the centroid of positive examples."""
    m = E_pos.mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-12)


def fit_diff_axis(E_pos: np.ndarray, E_neg: np.ndarray) -> np.ndarray:
    """Unit vector: positive centroid minus negative centroid.

    Works best when negatives include 'hard negatives' — fluent, ad-like but
    generic copy that a naive centroid would incorrectly score highly.
    """
    d = E_pos.mean(axis=0) - E_neg.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def score(E: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of E against the axis (E must be L2-normalized)."""
    return E @ axis


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Rank AUC: P(random positive scores above random negative)."""
    wins = sum(1 for p in pos_scores for n in neg_scores if p > n)
    ties = sum(1 for p in pos_scores for n in neg_scores if p == n)
    return (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))

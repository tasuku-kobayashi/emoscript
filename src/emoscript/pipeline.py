"""End-to-end fitting, scoring and persistence.

    embed_fn = make_embed_fn()                    # or any texts -> (n, dim) callable
    model = fit_from_corpus(embed_fn, k=3)        # axes + cross-fitted calibrator
    scores, probs = score_texts(model, ["Think fresh."], embed_fn)
    model.save("axis.npz"); AxisModel.load("axis.npz")

``embed_fn`` is injected so the whole pipeline can be exercised with a fake
embedder in tests and swapped for any backend in production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .axis import (
    Calibrator,
    accuracy,
    auc,
    brier,
    fit_multi_axis,
    l2_normalize,
    log_loss,
    score,
    score_per_axis,
)
from .dataio import DATA_DIR, load_corpus, texts
from .embed import DEFAULT_MODEL, EmbedFn, prefix_for
from .text import normalizer

NegativeKinds = tuple[str, ...]


@dataclass
class AxisModel:
    """A fitted scorer: one or more unit axes plus an optional calibrator."""

    axes: np.ndarray  # (k, dim)
    model_name: str = DEFAULT_MODEL
    prefix: str = ""
    calibrator: Calibrator | None = None
    meta: dict = field(default_factory=dict)
    casing: str = "none"  # text normalization applied before embedding (see emoscript.text)

    @property
    def k(self) -> int:
        return int(self.axes.shape[0])

    def embed(self, texts_: list[str], embed_fn: EmbedFn) -> np.ndarray:
        """Normalize casing the way the model was fitted, embed, L2-normalize."""
        return l2_normalize(embed_fn(normalizer(self.casing)(list(texts_))))

    def score(self, E: np.ndarray) -> np.ndarray:
        """Headline score: max cosine over axes (E must be L2-normalized)."""
        return score(E, self.axes)

    def score_per_axis(self, E: np.ndarray) -> np.ndarray:
        return score_per_axis(E, self.axes)

    def predict_proba(self, scores: np.ndarray) -> np.ndarray | None:
        if self.calibrator is None:
            return None
        return self.calibrator.predict_proba(np.asarray(scores))

    def threshold_for(self, prob: float = 0.5) -> float | None:
        """Raw score at which calibrated P(great) crosses ``prob`` (None if uncalibrated)."""
        if self.calibrator is None:
            return None
        return self.calibrator.score_at(prob)

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "axes": np.asarray(self.axes, dtype=np.float64),
            "model": np.asarray(self.model_name),
            "prefix": np.asarray(self.prefix),
            "casing": np.asarray(self.casing),
            "meta": np.asarray(json.dumps({**self.meta, "emoscript_version": __version__})),
        }
        if self.calibrator is not None:
            arrays.update(self.calibrator.to_arrays())
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> AxisModel:
        blob = np.load(path, allow_pickle=False)
        if "axes" in blob:
            axes = np.asarray(blob["axes"], dtype=np.float64)
        elif "axis" in blob:  # v0.1 artifact: a single axis vector
            axes = np.asarray(blob["axis"], dtype=np.float64)[None, :]
        else:
            raise ValueError(f"{path}: not an emoscript axis file")
        model_name = str(blob["model"]) if "model" in blob else DEFAULT_MODEL
        prefix = str(blob["prefix"]) if "prefix" in blob else prefix_for(model_name)
        meta = json.loads(str(blob["meta"])) if "meta" in blob else {}
        calibrator = Calibrator.from_arrays(blob) if "calib_w" in blob else None
        casing = str(blob["casing"]) if "casing" in blob else "none"
        return cls(
            axes=axes, model_name=model_name, prefix=prefix, calibrator=calibrator, meta=meta, casing=casing
        )


# --------------------------------------------------------------------------- fitting
def _folds(n: int, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(n)
    return [idx[i::n_folds] for i in range(n_folds)]


def cross_fit_scores(
    E_pos: np.ndarray,
    E_neg: np.ndarray,
    k: int,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold raw scores for every training positive and negative.

    Each fold's axes are fitted on the other folds only, so these scores are
    not optimistic and can be used to calibrate and to estimate the threshold.
    """
    rng = np.random.default_rng(seed)
    n_folds = max(2, min(n_folds, len(E_pos), len(E_neg)))
    pos_folds = _folds(len(E_pos), n_folds, rng)
    neg_folds = _folds(len(E_neg), n_folds, rng)
    oof_pos = np.empty(len(E_pos))
    oof_neg = np.empty(len(E_neg))
    for f in range(n_folds):
        pos_hold, neg_hold = pos_folds[f], neg_folds[f]
        pos_mask = np.ones(len(E_pos), bool)
        pos_mask[pos_hold] = False
        neg_mask = np.ones(len(E_neg), bool)
        neg_mask[neg_hold] = False
        k_f = min(k, int(pos_mask.sum()))
        axes, _ = fit_multi_axis(E_pos[pos_mask], E_neg[neg_mask], k_f, seed=seed)
        oof_pos[pos_hold] = score(E_pos[pos_hold], axes)
        oof_neg[neg_hold] = score(E_neg[neg_hold], axes)
    return oof_pos, oof_neg


def fit_from_embeddings(
    E_pos: np.ndarray,
    E_neg: np.ndarray,
    *,
    k: int = 1,
    calibrate: bool = True,
    n_folds: int = 5,
    seed: int = 0,
    model_name: str = DEFAULT_MODEL,
    prefix: str = "",
    casing: str = "none",
    meta: dict | None = None,
) -> AxisModel:
    """Fit axes (and a cross-fitted calibrator) from already-normalized embeddings."""
    E_pos = l2_normalize(E_pos)
    E_neg = l2_normalize(E_neg)
    axes, labels = fit_multi_axis(E_pos, E_neg, k, seed=seed)
    info: dict = {
        "k": int(k),
        "n_pos": int(len(E_pos)),
        "n_neg": int(len(E_neg)),
        "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
        **(meta or {}),
    }
    calibrator = None
    if calibrate:
        oof_pos, oof_neg = cross_fit_scores(E_pos, E_neg, k, n_folds=n_folds, seed=seed)
        calibrator = Calibrator().fit(oof_pos, oof_neg)
        p = calibrator.predict_proba(np.concatenate([oof_pos, oof_neg]))
        y = np.concatenate([np.ones(len(oof_pos)), np.zeros(len(oof_neg))])
        info["oof"] = {
            "auc": auc(oof_pos, oof_neg),
            "log_loss": log_loss(p, y),
            "brier": brier(p, y),
            "accuracy@0.5": accuracy(p, y),
            "threshold@0.5": calibrator.score_at(0.5),
            "pos_score_mean": float(oof_pos.mean()),
            "neg_score_mean": float(oof_neg.mean()),
        }
    return AxisModel(
        axes=axes, model_name=model_name, prefix=prefix, calibrator=calibrator, meta=info, casing=casing
    )


def fit_from_corpus(
    embed_fn: EmbedFn,
    data_dir: str | Path = DATA_DIR,
    *,
    k: int = 1,
    negatives: NegativeKinds = ("hard", "generic"),
    calibrate: bool = True,
    n_folds: int = 5,
    seed: int = 0,
    model_name: str = DEFAULT_MODEL,
    prefix: str | None = None,
    casing: str = "none",
) -> AxisModel:
    """Fit from the bundled (or a custom) corpus: train splits only."""
    corpus = load_corpus(data_dir)
    norm = normalizer(casing)
    pos = norm(texts(corpus.pos_train))
    neg = norm(texts(corpus.negatives_train(negatives)))
    if not pos or not neg:
        raise ValueError("fit_from_corpus: need at least one positive and one negative")
    E_pos = embed_fn(pos)
    E_neg = embed_fn(neg)
    return fit_from_embeddings(
        E_pos,
        E_neg,
        k=k,
        calibrate=calibrate,
        n_folds=n_folds,
        seed=seed,
        model_name=model_name,
        prefix=prefix_for(model_name) if prefix is None else prefix,
        casing=casing,
        meta={"negatives": list(negatives), "data_dir": str(Path(data_dir))},
    )


# --------------------------------------------------------------------------- scoring
def score_texts(
    model: AxisModel, texts_: list[str], embed_fn: EmbedFn
) -> tuple[np.ndarray, np.ndarray | None]:
    """Raw scores and calibrated probabilities (None when uncalibrated)."""
    s = model.score(model.embed(texts_, embed_fn))
    return s, model.predict_proba(s)


# --------------------------------------------------------------------------- evaluation
def evaluate(model: AxisModel, embed_fn: EmbedFn, data_dir: str | Path = DATA_DIR) -> dict:
    """Held-out evaluation on the *_test splits (never used for fitting)."""
    corpus = load_corpus(data_dir)
    groups = {
        "pos": texts(corpus.pos_test),
        "hard": texts(corpus.hard_test),
        "generic": texts(corpus.generic_test),
        "topic_bland": [r["bland"] for r in corpus.contrast],
    }
    S = {g: model.score(model.embed(t, embed_fn)) for g, t in groups.items() if t}
    out: dict = {
        "n": {g: len(t) for g, t in groups.items()},
        "auc": {g: auc(S["pos"], S[g]) for g in ("hard", "generic", "topic_bland") if g in S},
        "score_mean": {g: float(s.mean()) for g, s in S.items()},
    }
    if corpus.contrast and "topic_bland" in S:
        famous_score = dict(zip(groups["pos"], S["pos"], strict=True))
        wins = [famous_score[r["famous"]] > s for r, s in zip(corpus.contrast, S["topic_bland"], strict=True)]
        out["contrast_pair_win_rate"] = float(np.mean(wins))
    # length breakdown of held-out positives (word count)
    lengths = np.array([len(t.split()) for t in groups["pos"]])
    buckets = (
        ("short<=4w", lengths <= 4),
        ("mid5-7w", (lengths > 4) & (lengths < 8)),
        ("long>=8w", lengths >= 8),
    )
    for name, mask in buckets:
        if mask.any() and "hard" in S:
            out.setdefault("auc_vs_hard_by_length", {})[name] = {
                "n": int(mask.sum()),
                "auc": auc(S["pos"][mask], S["hard"]),
            }
    if model.calibrator is not None:
        negs = np.concatenate([S[g] for g in ("hard", "generic") if g in S])
        p = model.predict_proba(np.concatenate([S["pos"], negs]))
        y = np.concatenate([np.ones(len(S["pos"])), np.zeros(len(negs))])
        out["calibration"] = {
            "log_loss": log_loss(p, y),
            "brier": brier(p, y),
            "accuracy@0.5": accuracy(p, y),
            "threshold@0.5": model.threshold_for(0.5),
        }
    return out

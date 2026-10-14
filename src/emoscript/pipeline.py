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
    auc_bootstrap_ci,
    brier,
    fit_multi_axis,
    l2_normalize,
    log_loss,
    paired_auc_bootstrap_ci,
    score,
    score_per_axis,
    wilson_interval,
)
from .dataio import DATA_DIR, load_corpus, texts
from .embed import DEFAULT_MODEL, EmbedFn, prefix_for
from .text import canonicalize, has_letters, normalizer

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
    # Training positives kept for explanations: (n, dim) unit vectors, their texts and cluster ids.
    exemplars: np.ndarray | None = None
    exemplar_texts: list[str] = field(default_factory=list)
    exemplar_labels: np.ndarray | None = None
    # Every famous line the corpus knows (train and test) with its brand, for the
    # lexical near-duplicate check: a candidate that copies or parodies one of
    # these should be flagged, however well it scores.
    famous_texts: list[str] = field(default_factory=list)
    famous_brands: list[str] = field(default_factory=list)

    @property
    def k(self) -> int:
        return int(self.axes.shape[0])

    def best_axis(self, E: np.ndarray) -> np.ndarray:
        """Index of the axis each row matches best (all zeros for k=1)."""
        return self.score_per_axis(E).argmax(axis=1)

    def nearest(self, E: np.ndarray, n: int = 3) -> list[list[tuple[str, float]]]:
        """For each row of E, the n most similar training taglines as (text, cosine)."""
        if self.exemplars is None or not self.exemplar_texts:
            return [[] for _ in range(np.atleast_2d(E).shape[0])]
        sims = np.atleast_2d(np.asarray(E, dtype=np.float64)) @ self.exemplars.T
        out = []
        for row in sims:
            idx = np.argsort(-row)[:n]
            out.append([(self.exemplar_texts[i], float(row[i])) for i in idx])
        return out

    def explain(self, E: np.ndarray, n: int = 3) -> list[dict]:
        """Per row: headline score, best axis, per-axis scores and nearest famous taglines."""
        E = np.atleast_2d(np.asarray(E, dtype=np.float64))
        per_axis = self.score_per_axis(E)
        near = self.nearest(E, n)
        return [
            {
                "score": float(per_axis[i].max()),
                "axis": int(per_axis[i].argmax()),
                "per_axis": [float(x) for x in per_axis[i]],
                "nearest": near[i],
            }
            for i in range(len(E))
        ]

    def embed(self, texts_: list[str], embed_fn: EmbedFn) -> np.ndarray:
        """Normalize casing the way the model was fitted, embed, L2-normalize."""
        return l2_normalize(embed_fn(normalizer(self.casing)(list(texts_))))

    def score(self, E: np.ndarray) -> np.ndarray:
        """Headline score: max cosine over axes (E must be L2-normalized)."""
        return score(E, self.axes)

    def score_per_axis(self, E: np.ndarray) -> np.ndarray:
        return score_per_axis(E, self.axes)

    def predict_proba(self, scores: np.ndarray, prior: float = 0.5) -> np.ndarray | None:
        """Calibrated P(great copy).

        The calibrator is fitted with balanced classes, so its output assumes a
        50/50 prior. ``prior`` shifts the logit so that p reflects a different
        base rate: with hundreds of machine-generated candidates of which maybe
        one in twenty is usable, ``prior=0.05`` gives probabilities (and a
        PASS threshold) that mean what they say.
        """
        if self.calibrator is None:
            return None
        p = self.calibrator.predict_proba(np.asarray(scores))
        if prior == 0.5:
            return p
        if not 0 < prior < 1:
            raise ValueError("prior must be strictly between 0 and 1")
        eps = 1e-12
        logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1)) + np.log(prior / (1 - prior))
        return 1.0 / (1.0 + np.exp(-logit))

    def near_famous(
        self, texts_: list[str], *, max_edits: int = 2, trigram: float = 0.5
    ) -> list[dict | None]:
        """Lexical near-duplicate check against every famous line in the corpus.

        Flags verbatim copies and one- or two-word parodies ("Just Chew It.",
        "Kills Kids Dead") that the embedding scores like the originals. A
        match is a famous line with token edit distance <= ``max_edits``, capped
        at ``(tokens - 1) // 2`` (one- and two-word lines only match exactly,
        three- and four-word lines allow one swapped word), or character-trigram
        Jaccard >= ``trigram``.
        Returns, per input, ``{"text", "brand", "edits", "trigram"}`` or None.
        """
        if not self.famous_texts:
            return [None] * len(texts_)
        canon = [canonicalize(t) for t in self.famous_texts]
        toks = [c.split() for c in canon]
        tris = [_trigrams(c) for c in canon]
        out: list[dict | None] = []
        for t in texts_:
            c = canonicalize(t)
            ct, ctri = c.split(), _trigrams(c)
            best = None
            for i, (ft, ftri) in enumerate(zip(toks, tris, strict=True)):
                if abs(len(ft) - len(ct)) > max_edits:
                    jac = len(ctri & ftri) / len(ctri | ftri) if (ctri or ftri) else 0.0
                    if jac < trigram:
                        continue
                    edits = _token_edit_distance(ct, ft)
                else:
                    edits = _token_edit_distance(ct, ft)
                    jac = len(ctri & ftri) / len(ctri | ftri) if (ctri or ftri) else 0.0
                # allowed edits grow with the famous line: 1-2 tokens exact only, 3-4 one
                # word, 5+ two words. "Eat Fresh" must not flag every "<verb> fresh".
                allowed = min(max_edits, (len(ft) - 1) // 2)
                hit = edits <= allowed or jac >= trigram
                if hit and (best is None or (edits, -jac) < (best["edits"], -best["trigram"])):
                    best = {
                        "text": self.famous_texts[i],
                        "brand": self.famous_brands[i] if self.famous_brands else "",
                        "edits": int(edits),
                        "trigram": float(jac),
                    }
            out.append(best)
        return out

    def threshold_for(self, prob: float = 0.5) -> float | None:
        """Raw score at which calibrated P(great) crosses ``prob``.

        None when uncalibrated or when the calibrator is degenerate (no usable slope).
        """
        if self.calibrator is None:
            return None
        t = self.calibrator.score_at(prob)
        return None if np.isnan(t) else t

    # ---- persistence ----
    def save(self, path: str | Path) -> Path:
        """Write the model; numpy appends ``.npz`` if missing, so the real path is returned."""
        path = Path(path)
        if path.suffix != ".npz":
            path = Path(str(path) + ".npz")
        arrays: dict[str, np.ndarray] = {
            "axes": np.asarray(self.axes, dtype=np.float64),
            "model": np.asarray(self.model_name),
            "prefix": np.asarray(self.prefix),
            "casing": np.asarray(self.casing),
            "meta": np.asarray(json.dumps({**self.meta, "emoscript_version": __version__})),
        }
        if self.calibrator is not None:
            arrays.update(self.calibrator.to_arrays())
        if self.famous_texts:
            arrays["famous_texts"] = np.asarray(self.famous_texts)
            arrays["famous_brands"] = np.asarray(self.famous_brands or [""] * len(self.famous_texts))
        if self.exemplars is not None and self.exemplar_texts:
            arrays["ex_E"] = np.asarray(self.exemplars, dtype=np.float32)
            arrays["ex_texts"] = np.asarray(self.exemplar_texts)
            labels = self.exemplar_labels
            arrays["ex_labels"] = np.asarray(
                labels if labels is not None else np.zeros(len(self.exemplar_texts)), dtype=np.int32
            )
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> AxisModel:
        with np.load(path, allow_pickle=False) as blob:
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
            exemplars = exemplar_labels = None
            exemplar_texts: list[str] = []
            if "ex_E" in blob:
                exemplars = np.asarray(blob["ex_E"], dtype=np.float64)
                exemplar_texts = [str(t) for t in blob["ex_texts"]]
                exemplar_labels = np.asarray(blob["ex_labels"], dtype=int)
            famous_texts = [str(t) for t in blob["famous_texts"]] if "famous_texts" in blob else []
            famous_brands = [str(t) for t in blob["famous_brands"]] if "famous_brands" in blob else []
        return cls(
            axes=axes,
            model_name=model_name,
            prefix=prefix,
            calibrator=calibrator,
            meta=meta,
            casing=casing,
            exemplars=exemplars,
            exemplar_texts=exemplar_texts,
            exemplar_labels=exemplar_labels,
            famous_texts=famous_texts,
            famous_brands=famous_brands,
        )


def _trigrams(s: str) -> set[str]:
    s = s.replace(" ", "_")
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else ({s} if s else set())


def _token_edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


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
    pos_texts: list[str] | None = None,
    keep_exemplars: bool = True,
) -> AxisModel:
    """Fit axes (and a cross-fitted calibrator) from already-normalized embeddings.

    When ``pos_texts`` is given the training positives are kept in the model as
    exemplars (for ``explain`` / nearest-tagline output) and each cluster is
    summarised in ``meta["clusters"]`` by the taglines closest to its centroid.
    """
    E_pos = l2_normalize(E_pos)
    E_neg = l2_normalize(E_neg)
    if pos_texts is not None and len(pos_texts) != len(E_pos):
        raise ValueError("pos_texts must align with E_pos")
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(E_pos) < max(2, k) or len(E_neg) < 2:
        raise ValueError(
            f"need at least max(2, k)={max(2, k)} positives and 2 negatives, "
            f"got {len(E_pos)} and {len(E_neg)}"
        )
    axes, labels = fit_multi_axis(E_pos, E_neg, k, seed=seed)
    info: dict = {
        "k": int(k),
        "n_pos": int(len(E_pos)),
        "n_neg": int(len(E_neg)),
        "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
        **(meta or {}),
    }
    if pos_texts is not None:
        clusters = []
        for c in range(k):
            members = np.flatnonzero(labels == c)
            centroid = l2_normalize(E_pos[members].mean(axis=0))
            order = members[np.argsort(-(E_pos[members] @ centroid))]
            clusters.append(
                {
                    "size": int(len(members)),
                    "mean_words": float(np.mean([len(pos_texts[i].split()) for i in members])),
                    "exemplars": [pos_texts[i] for i in order[:5]],
                }
            )
        info["clusters"] = clusters
    calibrator = None
    if calibrate:
        oof_pos, oof_neg = cross_fit_scores(E_pos, E_neg, k, n_folds=n_folds, seed=seed)
        calibrator = Calibrator().fit(oof_pos, oof_neg)
        p = calibrator.predict_proba(np.concatenate([oof_pos, oof_neg]))
        y = np.concatenate([np.ones(len(oof_pos)), np.zeros(len(oof_neg))])
        threshold = calibrator.score_at(0.5)
        info["oof"] = {
            "auc": auc(oof_pos, oof_neg),
            "log_loss": log_loss(p, y),
            "brier": brier(p, y),
            "accuracy@0.5": accuracy(p, y),
            "threshold@0.5": None if np.isnan(threshold) else threshold,
            "calibrator_slope": calibrator.slope,
            "pos_score_mean": float(oof_pos.mean()),
            "neg_score_mean": float(oof_neg.mean()),
        }
        if not calibrator.slope > 1e-9:
            info["warnings"] = info.get("warnings", []) + [
                "calibrator is degenerate (non-positive slope): probabilities are not usable"
            ]
    keep = keep_exemplars and pos_texts is not None
    return AxisModel(
        axes=axes,
        model_name=model_name,
        prefix=prefix,
        calibrator=calibrator,
        meta=info,
        casing=casing,
        exemplars=E_pos if keep else None,
        exemplar_texts=list(pos_texts) if keep else [],
        exemplar_labels=labels if keep else None,
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
    casing: str = "canonical",
    keep_exemplars: bool = True,
) -> AxisModel:
    """Fit from the bundled (or a custom) corpus: train splits only.

    Defaults (k=1, casing="canonical", calibrator L2=1) were chosen by
    brand-grouped nested cross-validation on the training split only
    (``experiments/select_settings.py``); the held-out test split was not
    consulted. With the short and Title-Case negatives of v0.2 a single diff
    axis beats the multi-axis variants on out-of-fold log-loss and AUC.
    """
    corpus = load_corpus(data_dir)
    norm = normalizer(casing)
    pos_raw = texts(corpus.pos_train)
    pos = norm(pos_raw)
    neg = norm(texts(corpus.negatives_train(negatives)))
    if not pos or not neg:
        raise ValueError("fit_from_corpus: need at least one positive and one negative")
    E_pos = embed_fn(pos)
    E_neg = embed_fn(neg)
    model = fit_from_embeddings(
        E_pos,
        E_neg,
        k=k,
        calibrate=calibrate,
        n_folds=n_folds,
        seed=seed,
        model_name=model_name,
        prefix=prefix_for(model_name) if prefix is None else prefix,
        casing=casing,
        meta={
            "negatives": list(negatives),
            "data_dir": "bundled" if Path(data_dir).resolve() == DATA_DIR.resolve() else str(data_dir),
        },
        pos_texts=pos_raw,
        keep_exemplars=keep_exemplars,
    )
    famous = corpus.pos_train + corpus.pos_test
    model.famous_texts = [r["text"] for r in famous]
    model.famous_brands = [str(r.get("brand", "")) for r in famous]
    return model


# --------------------------------------------------------------------------- scoring
def score_texts(
    model: AxisModel, texts_: list[str], embed_fn: EmbedFn, *, prior: float = 0.5
) -> tuple[np.ndarray, np.ndarray | None]:
    """Raw scores and calibrated probabilities (None when uncalibrated).

    Inputs with no letters at all (empty, punctuation, numbers, emoji) are not
    copy and get NaN instead of a score. ``prior`` is passed to
    :meth:`AxisModel.predict_proba`.
    """
    texts_ = list(texts_)
    ok = np.array([has_letters(t) for t in texts_], dtype=bool)
    s = np.full(len(texts_), np.nan)
    if ok.any():
        s[ok] = model.score(model.embed([t for t, o in zip(texts_, ok, strict=True) if o], embed_fn))
    p = model.predict_proba(s, prior=prior) if model.calibrator is not None else None
    return s, p


def brief_relevance(model: AxisModel, texts_: list[str], brief: str, embed_fn: EmbedFn) -> np.ndarray:
    """Cosine between each line and a short product / campaign brief.

    The great-copy axis is deliberately topic-blind, so nonsense that *sounds*
    like a tagline ("Chase the Gravy") can outscore on-brief copy. Reporting
    relevance next to the score lets a ranking require both. Lines without
    letters get NaN.
    """
    texts_ = list(texts_)
    ok = np.array([has_letters(t) for t in texts_], dtype=bool)
    out = np.full(len(texts_), np.nan)
    if ok.any() and has_letters(brief):
        B = model.embed([brief], embed_fn)[0]
        E = model.embed([t for t, o in zip(texts_, ok, strict=True) if o], embed_fn)
        out[ok] = E @ B
    return out


# --------------------------------------------------------------------------- evaluation
def evaluate(
    model: AxisModel,
    embed_fn: EmbedFn,
    data_dir: str | Path = DATA_DIR,
    *,
    n_boot: int = 1000,
    threshold: float = 0.5,
) -> dict:
    """Held-out evaluation on the *_test splits (never used for fitting).

    - ``auc[g]``: famous taglines vs group g. ``auc_ci95`` uses a brand-cluster
      bootstrap for the taglines and a paired bootstrap for the topic-matched
      rewrites.
    - ``auc["quotes"]`` is the quotability control: famous taglines vs famous
      non-advertising lines. Near 0.5 means the axis measures memorable writing
      in general rather than anything specific to advertising.
    - ``pass_rate[g]``: share of each group with calibrated p >= ``threshold``;
      ``junk`` (placeholder / UI strings) should be close to 0.
    - ``calibration`` is computed over taglines vs hard + generic + topic-matched
      negatives (every bland group), not just the easy ones.
    """
    corpus = load_corpus(data_dir)
    groups = {
        "pos": texts(corpus.pos_test),
        "hard": texts(corpus.hard_test),
        "generic": texts(corpus.generic_test),
        "topic_bland": [r["bland"] for r in corpus.contrast],
        "quotes": texts(corpus.quotes),
        "junk": texts(corpus.junk),
    }
    S = {g: model.score(model.embed(t, embed_fn)) for g, t in groups.items() if t}
    pos_brands = [r.get("brand", i) for i, r in enumerate(corpus.pos_test)]
    neg_groups = [g for g in ("hard", "generic", "topic_bland", "quotes", "junk") if g in S]
    out: dict = {
        "n": {g: len(t) for g, t in groups.items()},
        "auc": {g: auc(S["pos"], S[g]) for g in neg_groups},
        "auc_ci95": {},
        "score_mean": {g: float(s.mean()) for g, s in S.items()},
    }
    for g in neg_groups:
        if g == "topic_bland" and len(S[g]) == len(S["pos"]):
            famous_by_text = dict(zip(groups["pos"], S["pos"], strict=True))
            fam = np.array([famous_by_text[r["famous"]] for r in corpus.contrast])
            out["auc_ci95"][g] = paired_auc_bootstrap_ci(fam, S[g], n_boot=n_boot)
            wins = int((fam > S[g]).sum())
            out["contrast_pair_win_rate"] = wins / len(fam)
            out["contrast_pair_win_ci95"] = wilson_interval(wins, len(fam))
        else:
            out["auc_ci95"][g] = auc_bootstrap_ci(S["pos"], S[g], pos_groups=pos_brands, n_boot=n_boot)
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
        P = {g: model.predict_proba(s) for g, s in S.items()}
        out["pass_rate"] = {g: float((p >= threshold).mean()) for g, p in P.items()}
        bland = [g for g in ("hard", "generic", "topic_bland") if g in S]
        negs = np.concatenate([S[g] for g in bland])
        p = model.predict_proba(np.concatenate([S["pos"], negs]))
        y = np.concatenate([np.ones(len(S["pos"])), np.zeros(len(negs))])
        out["calibration"] = {
            "negatives": bland,
            "log_loss": log_loss(p, y),
            "brier": brier(p, y),
            "accuracy@0.5": accuracy(p, y, threshold),
            "threshold@0.5": model.threshold_for(0.5),
        }
    return out

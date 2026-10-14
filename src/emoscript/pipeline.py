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
    family_weights,
    fit_multi_axis,
    l2_normalize,
    log_loss,
    paired_auc_bootstrap_ci,
    score,
    score_per_axis,
    weighted_mean,
    wilson_interval,
)
from .dataio import DATA_DIR, load_corpus, texts
from .embed import DEFAULT_MODEL, EmbedFn, prefix_for
from .text import BLOCKING_FLAGS, canonicalize, has_letters, input_flags, normalizer, word_tokens

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
    # Optional second direction: famous lines minus the "not a message" negatives
    # (nonsense, UI strings). The calibrator then reads [style, sense] and a line
    # must look like great copy AND like a message to score a high probability.
    sense_axis: np.ndarray | None = None
    # One-feature calibrator on the sense score alone (famous vs nonsense + UI, balanced,
    # out-of-fold). P(message) >= 0.5 is the sense GATE: the two-feature calibrator gave the
    # sense axis almost no weight, so a soft feature changed nothing; the gate does.
    sense_gate: Calibrator | None = None
    _famous_cache: tuple | None = field(default=None, repr=False, compare=False)

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

    @property
    def n_features(self) -> int:
        return 2 if self.sense_axis is not None else 1

    def sense_score(self, E: np.ndarray) -> np.ndarray | None:
        if self.sense_axis is None:
            return None
        return np.atleast_2d(np.asarray(E, dtype=np.float64)) @ self.sense_axis

    def features(self, E: np.ndarray) -> np.ndarray:
        """Calibrator input per row: ``[style]`` or ``[style, sense]``."""
        s = self.score(E)
        if self.sense_axis is None:
            return s[:, None]
        return np.column_stack([s, self.sense_score(E)])

    def sense_proba(self, E: np.ndarray) -> np.ndarray | None:
        """P(this is a message, not nonsense) from the sense score; None without a gate."""
        if self.sense_axis is None or self.sense_gate is None:
            return None
        return self.sense_gate.predict_proba(self.sense_score(E))

    def passes_sense_gate(self, E: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Boolean per row; all True when the model has no sense gate."""
        p = self.sense_proba(E)
        return np.ones(np.atleast_2d(E).shape[0], dtype=bool) if p is None else p >= threshold

    def proba_from_embeddings(self, E: np.ndarray, prior: float = 0.5) -> np.ndarray | None:
        """Calibrated P(great copy) from embeddings (works for 1- and 2-feature models)."""
        if self.calibrator is None:
            return None
        return self._shift_prior(self.calibrator.predict_proba(self.features(E)), prior)

    @staticmethod
    def _shift_prior(p: np.ndarray, prior: float) -> np.ndarray:
        if prior == 0.5:
            return p
        if not 0 < prior < 1:
            raise ValueError("prior must be strictly between 0 and 1")
        eps = 1e-12
        logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1)) + np.log(prior / (1 - prior))
        return 1.0 / (1.0 + np.exp(-logit))

    def predict_proba(self, scores: np.ndarray, prior: float = 0.5) -> np.ndarray | None:
        """Calibrated P(great copy) from the style score alone.

        For a two-feature model (``sense_axis`` set) the style score is not
        enough; use :meth:`proba_from_embeddings`. This method then raises.

        The calibrator is fitted with balanced classes, so its output assumes a
        50/50 prior. ``prior`` shifts the logit so that p reflects a different
        base rate: with hundreds of machine-generated candidates of which maybe
        one in twenty is usable, ``prior=0.05`` gives probabilities (and a
        PASS threshold) that mean what they say.
        """
        if self.calibrator is None:
            return None
        if self.sense_axis is not None:
            raise ValueError("this model calibrates on [style, sense]; use proba_from_embeddings(E)")
        return self._shift_prior(self.calibrator.predict_proba(np.asarray(scores)), prior)

    def near_famous(
        self, texts_: list[str], *, max_edits: int = 2, trigram: float = 0.5
    ) -> list[dict | None]:
        """Lexical check against every famous line in the corpus (train and test).

        Comparison is on word tokens with punctuation, apostrophes, case and
        wrappers removed, so "Just. Do. It.", "I’m lovin’ it" and
        '1. Tagline: "Just Do It."' all count as copies.

        Severity:
        - ``copy``      identical tokens. The CLI refuses to PASS these.
        - ``parody``    token edit distance <= ``max_edits`` and at most
                        ``(tokens - 1) // 2`` of the famous line changed
                        ("Just Chew It.", "Kills Kids Dead"), or (both lines
                        3+ tokens) character-trigram Jaccard >= ``trigram``.
        - ``possible``  a two-word famous line with its first word kept and the
                        second swapped ("Got Mold?").
        Parodies and possibles are warnings, not blocks: short ordinary copy
        ("Do the math.", "Like a pro.") sits one word away from some famous
        line far too often for a lexical rule to be allowed to veto it.
        Returns, per input, ``{"text", "brand", "edits", "trigram", "severity"}`` or None.
        """
        if not self.famous_texts:
            return [None] * len(texts_)
        if self._famous_cache is None or self._famous_cache[0] is not self.famous_texts:
            toks = [word_tokens(canonicalize(t)) for t in self.famous_texts]
            self._famous_cache = (self.famous_texts, toks, [_trigrams(" ".join(t)) for t in toks])
        _, toks, tris = self._famous_cache
        out: list[dict | None] = []
        for t in texts_:
            ct = word_tokens(canonicalize(t))
            ctri = _trigrams(" ".join(ct))
            best = None
            for i, (ft, ftri) in enumerate(zip(toks, tris, strict=True)):
                if not ft or not ct:
                    continue
                if abs(len(ft) - len(ct)) > max_edits:
                    if len(ft) < 3 or len(ct) < 3:
                        continue
                    jac = len(ctri & ftri) / len(ctri | ftri) if (ctri or ftri) else 0.0
                    if jac < trigram:
                        continue
                    edits = _token_edit_distance(ct, ft)
                else:
                    edits = _token_edit_distance(ct, ft)
                    jac = len(ctri & ftri) / len(ctri | ftri) if (ctri or ftri) else 0.0
                allowed = min(max_edits, (len(ft) - 1) // 2)
                if edits == 0:
                    severity = "copy"
                elif edits <= allowed or (len(ft) >= 3 and len(ct) >= 3 and jac >= trigram):
                    severity = "parody"
                elif len(ft) == 2 and len(ct) == 2 and edits == 1 and ct[0] == ft[0]:
                    severity = "possible"  # "Got Mold?" rides on "Got Milk?"; lower confidence
                else:
                    continue
                rank = ({"copy": 0, "parody": 1, "possible": 2}[severity], edits, -jac)
                if best is None or rank < best["_rank"]:
                    best = {
                        "text": self.famous_texts[i],
                        "brand": self.famous_brands[i] if self.famous_brands else "",
                        "edits": int(edits),
                        "trigram": float(jac),
                        "severity": severity,
                        "_rank": rank,
                    }
            if best is not None:
                best.pop("_rank")
            out.append(best)
        return out

    def threshold_for(self, prob: float = 0.5) -> float | None:
        """Raw score at which calibrated P(great) crosses ``prob``.

        None when uncalibrated or when the calibrator is degenerate (no usable slope).
        """
        if self.calibrator is None or self.sense_axis is not None:
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
        if self.sense_axis is not None:
            arrays["sense_axis"] = np.asarray(self.sense_axis, dtype=np.float64)
        if self.sense_gate is not None:
            arrays.update({"gate_" + k: v for k, v in self.sense_gate.to_arrays().items()})
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
            sense_axis = np.asarray(blob["sense_axis"], dtype=np.float64) if "sense_axis" in blob else None
            sense_gate = None
            if "gate_calib_w" in blob:
                sense_gate = Calibrator.from_arrays(
                    {k[5:]: blob[k] for k in blob.files if k.startswith("gate_")}
                )
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
            sense_axis=sense_axis,
            sense_gate=sense_gate,
        )


def web_export(
    model: AxisModel, web_model: str | None = None, pooling: str | None = None, revision: str | None = None
) -> dict:
    """Everything the browser app (``web/``) needs, as plain JSON types.

    The embedding model itself is not exported: the app loads the transformers.js
    port named here from the Hugging Face hub. ``web_model`` / ``pooling``
    default to the known mapping for the axis's fastembed model.
    """
    from .embed import WEB_MODELS, WEB_REVISIONS

    if web_model is None or pooling is None:
        if model.model_name not in WEB_MODELS:
            raise ValueError(
                f"no transformers.js mapping for {model.model_name!r}; pass web_model and pooling "
                f"(known: {', '.join(sorted(WEB_MODELS))})"
            )
        wm, pl = WEB_MODELS[model.model_name]
        web_model, pooling = web_model or wm, pooling or pl
    if pooling not in ("cls", "mean"):
        raise ValueError("pooling must be 'cls' or 'mean'")
    if revision is None:
        revision = WEB_REVISIONS.get(web_model, "main")

    def calib(c: Calibrator | None) -> dict | None:
        if c is None:
            return None
        return {
            "w": [float(x) for x in c.w],
            "b": float(c.b),
            "mu": [float(x) for x in c.mu],
            "sd": [float(x) for x in c.sd],
        }

    meta = {
        k: v
        for k, v in model.meta.items()
        if k in ("k", "n_pos", "n_neg", "oof", "sense_gate_oof", "data_dir")
    }
    return {
        "format": "emoscript-web/1",
        "emoscript_version": __version__,
        "embedding": {
            "fastembed_model": model.model_name,
            "web_model": web_model,
            "pooling": pooling,
            "revision": revision,
            "prefix": model.prefix,
            "dim": int(model.axes.shape[1]),
        },
        "casing": model.casing,
        "axes": [[float(x) for x in row] for row in model.axes],
        "calibrator": calib(model.calibrator),
        "sense_axis": [float(x) for x in model.sense_axis] if model.sense_axis is not None else None,
        "sense_gate": calib(model.sense_gate),
        "famous": [
            {"text": t, "brand": b}
            for t, b in zip(
                model.famous_texts, model.famous_brands or [""] * len(model.famous_texts), strict=True
            )
        ],
        "meta": meta,
    }


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


SENSE_FAMILIES = frozenset({"nonsense", "ui"})  # "not a message" negatives for the sense axis


def negative_family(row: dict) -> str:
    """Coarse family of a negative row from its ``note`` (see CONTRIBUTING)."""
    note = str(row.get("note", "")).lower()
    if note == "short-factual":
        return "prose"
    if note.startswith("generic-"):
        return "ad-speak"
    if note.startswith("short-") or note.startswith("mid-"):
        return "short"
    if note.startswith("ui-"):
        return "ui"
    if note.startswith("topic-"):
        return "topic"
    if note.startswith("question") or note.startswith("nonsense"):
        return note.split("-")[0]
    return "prose"  # transactional / factual / support / policy / instructional / short-factual


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
    neg_weights: np.ndarray | None = None,
    sense_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold raw scores for every training positive and negative.

    Each fold's axes are fitted on the other folds only, so these scores are
    not optimistic and can be used to calibrate and to estimate the threshold.
    With ``sense_weights`` the arrays have two columns, ``[style, sense]``.
    """
    rng = np.random.default_rng(seed)
    n_folds = max(2, min(n_folds, len(E_pos), len(E_neg)))
    pos_folds = _folds(len(E_pos), n_folds, rng)
    neg_folds = _folds(len(E_neg), n_folds, rng)
    nf = 2 if sense_weights is not None else 1
    oof_pos = np.empty((len(E_pos), nf))
    oof_neg = np.empty((len(E_neg), nf))
    for f in range(n_folds):
        pos_hold, neg_hold = pos_folds[f], neg_folds[f]
        pos_mask = np.ones(len(E_pos), bool)
        pos_mask[pos_hold] = False
        neg_mask = np.ones(len(E_neg), bool)
        neg_mask[neg_hold] = False
        k_f = min(k, int(pos_mask.sum()))
        w_f = None if neg_weights is None else np.asarray(neg_weights)[neg_mask]
        axes, _ = fit_multi_axis(E_pos[pos_mask], E_neg[neg_mask], k_f, seed=seed, neg_weights=w_f)
        oof_pos[pos_hold, 0] = score(E_pos[pos_hold], axes)
        oof_neg[neg_hold, 0] = score(E_neg[neg_hold], axes)
        if sense_weights is not None:
            sw = np.asarray(sense_weights)[neg_mask]
            s_axis = fit_sense_axis(E_pos[pos_mask], E_neg[neg_mask], sw)
            oof_pos[pos_hold, 1] = E_pos[pos_hold] @ s_axis
            oof_neg[neg_hold, 1] = E_neg[neg_hold] @ s_axis
    if nf == 1:
        return oof_pos[:, 0], oof_neg[:, 0]
    return oof_pos, oof_neg


def fit_sense_axis(E_pos: np.ndarray, E_neg: np.ndarray, sense_weights: np.ndarray) -> np.ndarray:
    """Famous centroid minus the centroid of the "not a message" negatives (weights > 0)."""
    w = np.asarray(sense_weights, dtype=np.float64)
    if not (w > 0).any():
        raise ValueError("sense_weights selects no negatives")
    return l2_normalize(np.asarray(E_pos).mean(axis=0) - weighted_mean(E_neg, w))


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
    neg_weights: np.ndarray | None = None,
    sense_weights: np.ndarray | None = None,
) -> AxisModel:
    """Fit axes (and a cross-fitted calibrator) from already-normalized embeddings.

    ``neg_weights`` (one per negative row) weights the negative centroid; see
    :func:`emoscript.axis.family_weights` for equal weight per negative family.
    ``sense_weights`` (one per negative row, zero outside the "not a message"
    families) adds a second axis and a two-feature calibrator; see
    :class:`AxisModel.sense_axis`.

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
    axes, labels = fit_multi_axis(E_pos, E_neg, k, seed=seed, neg_weights=neg_weights)
    info: dict = {
        "k": int(k),
        "n_pos": int(len(E_pos)),
        "n_neg": int(len(E_neg)),
        "neg_weighted": neg_weights is not None,
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
    sense_axis = fit_sense_axis(E_pos, E_neg, sense_weights) if sense_weights is not None else None
    info["sense_axis"] = sense_axis is not None
    calibrator = None
    sense_gate = None
    if calibrate:
        oof_pos, oof_neg = cross_fit_scores(
            E_pos, E_neg, k, n_folds=n_folds, seed=seed, neg_weights=neg_weights, sense_weights=sense_weights
        )
        calibrator = Calibrator().fit(oof_pos, oof_neg)
        if sense_axis is not None:
            in_sense = np.asarray(sense_weights) > 0
            sense_gate = Calibrator().fit(oof_pos[:, 1], oof_neg[in_sense, 1])
            gate_p = sense_gate.predict_proba(np.concatenate([oof_pos[:, 1], oof_neg[:, 1]]))
            info["sense_gate_oof"] = {
                "famous_pass_rate": float((gate_p[: len(oof_pos)] >= 0.5).mean()),
                "nonsense_pass_rate": float((gate_p[len(oof_pos) :][in_sense] >= 0.5).mean()),
                "other_negative_pass_rate": float((gate_p[len(oof_pos) :][~in_sense] >= 0.5).mean()),
                "threshold_sense_score": sense_gate.score_at(0.5),
            }
        p = calibrator.predict_proba(np.concatenate([oof_pos, oof_neg]))
        y = np.concatenate([np.ones(len(oof_pos)), np.zeros(len(oof_neg))])
        style_pos = oof_pos if oof_pos.ndim == 1 else oof_pos[:, 0]
        style_neg = oof_neg if oof_neg.ndim == 1 else oof_neg[:, 0]
        threshold = calibrator.score_at(0.5) if sense_axis is None else float("nan")
        info["oof"] = {
            "auc": auc(style_pos, style_neg),
            "auc_proba": auc(p[: len(oof_pos)], p[len(oof_pos) :]),
            "log_loss": log_loss(p, y),
            "brier": brier(p, y),
            "accuracy@0.5": accuracy(p, y),
            "threshold@0.5": None if np.isnan(threshold) else threshold,
            "calibrator_slope": calibrator.slope if sense_axis is None else None,
            "calibrator_weights": [float(x) for x in calibrator.w],
            "pos_score_mean": float(style_pos.mean()),
            "neg_score_mean": float(style_neg.mean()),
        }
        if sense_axis is None and not calibrator.slope > 1e-9:
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
        sense_axis=sense_axis,
        sense_gate=sense_gate,
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
    balance_negatives: bool = True,
    sense: bool = True,
) -> AxisModel:
    """Fit from the bundled (or a custom) corpus: train splits only.

    ``sense`` (default) adds the second, "is this even a message" axis fitted
    against the nonsense and UI-boilerplate families, and a two-feature
    calibrator. It exists because the style axis alone cannot tell "Chase the
    Gravy" from a tagline.

    ``balance_negatives`` (default) gives every negative family (by the
    ``note`` prefix: generic ad-speak, short, UI, topic-descriptive, nonsense,
    prose, ...) equal total weight in the negative centroid instead of one
    vote per line, so the largest family cannot decide the direction. Chosen
    by train-only cross-validation (``experiments/select_settings.py``).

    Defaults (k=1, casing="canonical", calibrator L2=1, balanced negative
    families) were chosen by
    brand-grouped nested cross-validation on the training split only
    (``experiments/select_settings.py``); the held-out test split was not
    consulted. With the short and Title-Case negatives of v0.2 a single diff
    axis beats the multi-axis variants on out-of-fold log-loss and AUC.
    """
    corpus = load_corpus(data_dir)
    norm = normalizer(casing)
    pos_raw = texts(corpus.pos_train)
    pos = norm(pos_raw)
    neg_rows = corpus.negatives_train(negatives)
    neg = norm(texts(neg_rows))
    families = [negative_family(r) for r in neg_rows]
    neg_weights = family_weights(families) if balance_negatives else None
    sense_weights = None
    if sense:
        in_sense = np.array([f in SENSE_FAMILIES for f in families])
        if in_sense.any():
            sense_weights = np.where(
                in_sense, family_weights([f if f in SENSE_FAMILIES else "x" for f in families]), 0.0
            )
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
        neg_weights=neg_weights,
        sense_weights=sense_weights,
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
    ok, E = embed_valid(model, texts_, embed_fn)
    s = np.full(len(texts_), np.nan)
    p = np.full(len(texts_), np.nan) if model.calibrator is not None else None
    if ok.any():
        s[ok] = model.score(E)
        if p is not None:
            p[ok] = model.proba_from_embeddings(E, prior=prior)
    return s, p


def embed_valid(model: AxisModel, texts_: list[str], embed_fn: EmbedFn) -> tuple[np.ndarray, np.ndarray]:
    """Embed the lines that have letters once; returns ``(ok_mask, E)`` with E aligned to ok rows."""
    texts_ = list(texts_)
    ok = np.array([has_letters(t) for t in texts_], dtype=bool)
    E = (
        model.embed([t for t, o in zip(texts_, ok, strict=True) if o], embed_fn)
        if ok.any()
        else np.zeros((0, 1))
    )
    return ok, E


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
      ``junk`` (placeholder / UI strings) should be close to 0 and ``good_copy``
      (original usable lines) close to 1: together they are the cost/benefit of
      every junk-suppressing measure.
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
        "good_copy": texts(corpus.good_copy),
    }
    EMB = {g: model.embed(t, embed_fn) for g, t in groups.items() if t}
    S = {g: model.score(e) for g, e in EMB.items()}
    pos_brands = [r.get("brand", i) for i, r in enumerate(corpus.pos_test)]
    # held-out lines whose brand belongs to the same company as a training brand (disclosed in the data)
    sister = np.array([bool(r.get("sister_brand")) for r in corpus.pos_test])
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
    if sister.any() and (~sister).any():
        keep = ~sister
        fam_by_text = dict(zip(groups["pos"], S["pos"], strict=True))
        pair_keep = np.array([not r.get("sister_brand") for r in corpus.pos_test])
        fam = np.array([fam_by_text[r["famous"]] for r in corpus.contrast])
        contrast_keep = np.array(
            [
                not next(x for x in corpus.pos_test if x["text"] == r["famous"]).get("sister_brand")
                for r in corpus.contrast
            ]
        )
        out["excluding_sister_brands"] = {
            "n_pos": int(keep.sum()),
            "n_sister": int(sister.sum()),
            "auc": {g: auc(S["pos"][keep], S[g]) for g in neg_groups if g != "topic_bland"},
            "auc_topic_bland_paired": auc(fam[contrast_keep], S["topic_bland"][contrast_keep])
            if "topic_bland" in S
            else None,
        }
        del pair_keep
    if "good_copy" in S:
        out["good_copy"] = {
            "auc_vs_junk": auc(S["good_copy"], S["junk"]) if "junk" in S else None,
            "auc_vs_hard": auc(S["good_copy"], S["hard"]) if "hard" in S else None,
            "auc_famous_vs_good": auc(S["pos"], S["good_copy"]),
        }
    if model.calibrator is not None:
        P = {g: model.proba_from_embeddings(EMB[g]) for g in S}
        G = {g: model.passes_sense_gate(EMB[g]) for g in S}
        out["pass_rate"] = {g: float(((p >= threshold) & G[g]).mean()) for g, p in P.items()}
        if model.sense_gate is not None:
            out["sense_gate_pass_rate"] = {g: float(G[g].mean()) for g in S}
            out["pass_rate_without_gate"] = {g: float((p >= threshold).mean()) for g, p in P.items()}
        if "good_copy" in P and "junk" in P:
            out["good_copy"]["auc_proba_vs_junk"] = auc(P["good_copy"], P["junk"])
            # the scorer is built for tagline-length lines; show the register split explicitly
            gw = np.array([len(t.split()) for t in groups["good_copy"]])
            short = gw <= 6
            ok_g = (P["good_copy"] >= threshold) & G["good_copy"]
            out["good_copy"]["by_register"] = {
                "tagline_length<=6w": {"n": int(short.sum()), "pass_rate": float(ok_g[short].mean())},
                "longer>6w": {"n": int((~short).sum()), "pass_rate": float(ok_g[~short].mean())},
            }
        # what a CLI user actually sees: a line with an input flag (placeholder, single
        # word, wrapped, ...) or a near-famous warning is not an unqualified PASS
        guarded = {}
        for g, t in groups.items():
            if g not in P:
                continue
            near = model.near_famous(t) if g != "pos" else [None] * len(t)  # the famous lines ARE famous
            flagged = np.array(
                [
                    bool(BLOCKING_FLAGS & set(input_flags(x)))
                    or (nf is not None and nf["severity"] == "copy")
                    for x, nf in zip(t, near, strict=True)
                ]
            )
            guarded[g] = float(((P[g] >= threshold) & G[g] & ~flagged).mean())
        out["pass_rate_after_guards"] = guarded
        bland = [g for g in ("hard", "generic", "topic_bland") if g in S]
        p = np.concatenate([P["pos"]] + [P[g] for g in bland])
        y = np.concatenate([np.ones(len(S["pos"]))] + [np.zeros(len(S[g])) for g in bland])
        out["calibration"] = {
            "negatives": bland,
            "log_loss": log_loss(p, y),
            "brier": brier(p, y),
            "accuracy@0.5": accuracy(p, y, threshold),
            "threshold@0.5": model.threshold_for(0.5),
        }
    return out

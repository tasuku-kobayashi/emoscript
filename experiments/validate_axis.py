"""Held-out validation of the great-copy axis.

Everything is fitted on the *_train splits and evaluated on the *_test splits
(never used for fitting). Variants:

  M1  positive centroid only
  M2  diff axis: positives - (hard + generic negatives)        [default, k=1]
  M3  diff axis: positives - hard negatives only
  Mk  multi-axis: k-means over positives, one diff axis each, max over axes
  kNN reference: mean cosine to the 5 nearest training positives minus the
      5 nearest training negatives (not an axis; a sanity ceiling for the
      corpus, used to judge how much the single direction gives up)
  surface reference: a logistic regression on six cheap features of the raw
      text (length, words, final full stop, apostrophe, word length, Title
      Case ratio). If the axis does not beat this row, it has learned
      typography, not writing.

Also reported: the PASS rate per group for calibrated variants (junk must be
near 0), and the quotability control (famous taglines vs famous non-ad lines).

Reports, per variant: AUC of held-out taglines vs hard negatives, vs generic
prose, vs *topic-matched* bland rewrites of the same taglines (the style-vs-
topic control), the paired win rate on those rewrites, an AUC breakdown by
tagline length, and (via cross-fitted calibration on train) probability
quality on the held-out set.

    python experiments/validate_axis.py [--model NAME] [--k 1,2,3,4] [--casing lower] [--json out.json]

``--casing lower`` embeds everything lower-cased: famous taglines are written
in Title Case and the negatives in sentence case, so this run shows how much
of the separation survives once that typographic cue is removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from emoscript.axis import (  # noqa: E402
    Calibrator,
    accuracy,
    auc,
    auc_bootstrap_ci,
    brier,
    family_weights,  # noqa: E402
    fit_diff_axis,
    fit_mean_axis,
    l2_normalize,
    log_loss,
    paired_auc_bootstrap_ci,
    score,
    wilson_interval,
)
from emoscript.dataio import DATA_DIR, load_corpus, texts  # noqa: E402
from emoscript.embed import DEFAULT_MODEL, make_embed_fn  # noqa: E402
from emoscript.pipeline import fit_from_embeddings, negative_family  # noqa: E402
from emoscript.text import BLOCKING_FLAGS, CASINGS, input_flags, normalizer  # noqa: E402

# Probes are illustrative only (not metrics). They are new lines, not paraphrases of corpus entries.
PROBES = [
    ("Sleep like you mean it.", "coined, punchy"),
    ("The sofa that forgives.", "coined, concrete image"),
    ("Comfort, redefined for modern living.", "bland ad-speak"),
    ("We sell high quality furniture at low prices.", "bland ad-speak"),
    ("Lorem Ipsum", "placeholder"),
]


def length_bucket(t: str) -> str:
    n = len(t.split())
    return "short<=4w" if n <= 4 else ("mid5-7w" if n < 8 else "long>=8w")


def run(
    model_name: str,
    ks: list[int],
    data_dir: Path,
    embed_fn=None,
    casing: str = "none",
    n_boot: int = 1000,
    balance: bool = True,
) -> dict:
    corpus = load_corpus(data_dir)
    embed_fn = embed_fn or make_embed_fn(model_name)
    norm = normalizer(casing)

    groups = {
        "pos_tr": texts(corpus.pos_train),
        "hard_tr": texts(corpus.hard_train),
        "gen_tr": texts(corpus.generic_train),
        "pos_te": texts(corpus.pos_test),
        "hard_te": texts(corpus.hard_test),
        "gen_te": texts(corpus.generic_test),
        "topic_te": [r["bland"] for r in corpus.contrast],
        "quotes": texts(corpus.quotes),
        "junk": texts(corpus.junk),
        "probe": [p for p, _ in PROBES],
    }
    groups = {g: ts for g, ts in groups.items() if ts}
    pos_brands = [r["brand"] for r in corpus.pos_test]
    all_texts = norm([t for g in groups.values() for t in g])
    E_all = l2_normalize(embed_fn(all_texts))
    E: dict[str, np.ndarray] = {}
    i = 0
    for g, ts in groups.items():
        E[g] = E_all[i : i + len(ts)]
        i += len(ts)

    N_all = np.vstack([E["hard_tr"], E["gen_tr"]])
    neg_w = (
        family_weights([negative_family(r) for r in corpus.hard_train + corpus.generic_train])
        if balance
        else None
    )
    variants: dict[str, dict] = {}

    def add_variant(name: str, axes: np.ndarray, calibrator=None, oof=None, scorer=None):
        scorer = scorer or (lambda X: score(X, axes))
        s = {
            g: scorer(E[g])
            for g in ("pos_te", "hard_te", "gen_te", "topic_te", "quotes", "junk", "probe")
            if g in E
        }
        famous_score = dict(zip(groups["pos_te"], s["pos_te"], strict=True))
        fam_paired = np.array([famous_score[r["famous"]] for r in corpus.contrast])
        wins = [famous_score[r["famous"]] > sc for r, sc in zip(corpus.contrast, s["topic_te"], strict=True)]
        buckets = {}
        for b in ("short<=4w", "mid5-7w", "long>=8w"):
            m = np.array([length_bucket(t) == b for t in groups["pos_te"]])
            if m.any():
                buckets[b] = {"n": int(m.sum()), "auc_vs_hard": auc(s["pos_te"][m], s["hard_te"])}
        v = {
            "k": int(np.atleast_2d(axes).shape[0]) if axes is not None else 0,
            "auc_vs_hard": auc(s["pos_te"], s["hard_te"]),
            "auc_vs_generic": auc(s["pos_te"], s["gen_te"]),
            "auc_vs_topic_bland": auc(s["pos_te"], s["topic_te"]),
            "auc_vs_quotes": auc(s["pos_te"], s["quotes"]) if "quotes" in s else None,
            "auc_vs_junk": auc(s["pos_te"], s["junk"]) if "junk" in s else None,
            "auc_ci95": {
                **{
                    g: auc_bootstrap_ci(s["pos_te"], s[g], pos_groups=pos_brands, n_boot=n_boot)
                    for g in ("hard_te", "gen_te", "quotes", "junk")
                    if g in s
                },
                "topic_te": paired_auc_bootstrap_ci(fam_paired, s["topic_te"], n_boot=n_boot),
            },
            "topic_pair_win_rate": float(np.mean(wins)),
            "topic_pair_win_ci95": wilson_interval(int(np.sum(wins)), len(wins)),
            "by_length": buckets,
            "mean": {g: float(sc.mean()) for g, sc in s.items()},
            "min_pos": float(s["pos_te"].min()),
            "max_hard": float(s["hard_te"].max()),
            "probes": {p: float(sc) for p, sc in zip(groups["probe"], s["probe"], strict=True)},
            "lowest_famous": [
                {"text": groups["pos_te"][j], "score": float(s["pos_te"][j])}
                for j in np.argsort(s["pos_te"])[:5]
            ],
            "highest_bland": [
                {"text": groups["hard_te"][j], "score": float(s["hard_te"][j])}
                for j in np.argsort(-s["hard_te"])[:3]
            ],
        }
        if calibrator is not None:
            negs = np.concatenate([s["hard_te"], s["gen_te"], s["topic_te"]])
            p = calibrator.predict_proba(np.concatenate([s["pos_te"], negs]))
            y = np.concatenate([np.ones(len(s["pos_te"])), np.zeros(len(negs))])
            v["calibration"] = {
                "oof_train": oof,
                "negatives": "hard+generic+topic",
                "test_log_loss": log_loss(p, y),
                "test_brier": brier(p, y),
                "test_accuracy@0.5": accuracy(p, y),
                "threshold@0.5": calibrator.score_at(0.5),
            }
            v["pass_rate@0.5"] = {
                g: float((calibrator.predict_proba(sc) >= 0.5).mean()) for g, sc in s.items() if g != "probe"
            }
            if "junk" in s:
                flagged = np.array([bool(BLOCKING_FLAGS & set(input_flags(t))) for t in groups["junk"]])
                v["junk_pass_rate_after_flags"] = float(
                    ((calibrator.predict_proba(s["junk"]) >= 0.5) & ~flagged).mean()
                )
        variants[name] = v
        return s

    def surface_features(ts: list[str]) -> np.ndarray:
        rows_ = []
        for t in ts:
            words = t.split()
            n = max(len(words), 1)
            rows_.append(
                [
                    len(t) / 100,
                    len(words) / 20,
                    float(t.rstrip().endswith(".")),
                    float("'" in t),
                    (sum(len(w) for w in words) / n) / 10,
                    sum(w[:1].isupper() for w in words) / n,
                ]
            )
        return np.array(rows_)

    surface = Calibrator(l2=1.0).fit(
        surface_features(groups["pos_tr"]),
        surface_features(groups["hard_tr"] + groups["gen_tr"]),
    )
    group_of_matrix = {id(E[g]): g for g in E}

    def surface_scorer(X: np.ndarray) -> np.ndarray:
        # scored on the RAW text of the same group (before any normalization)
        return surface.predict_proba(surface_features(groups[group_of_matrix[id(X)]]))

    add_variant("surface features on raw text (not an axis)", None, scorer=surface_scorer)
    add_variant("M1 centroid", fit_mean_axis(E["pos_tr"]))

    def knn_scorer(X: np.ndarray, kk: int = 5) -> np.ndarray:
        sp = np.sort(X @ E["pos_tr"].T, axis=1)[:, -kk:].mean(axis=1)
        sn = np.sort(X @ N_all.T, axis=1)[:, -kk:].mean(axis=1)
        return sp - sn

    add_variant("kNN-5 reference (not an axis)", None, scorer=knn_scorer)
    add_variant("M3 diff (hard only)", fit_diff_axis(E["pos_tr"], E["hard_tr"]))
    ranking_scores = None
    for k in ks:
        m = fit_from_embeddings(
            E["pos_tr"], N_all, k=k, calibrate=True, model_name=model_name, neg_weights=neg_w
        )
        tag = ", balanced families" if balance else ""
        name = f"M2 diff (hard+generic{tag})" if k == 1 else f"M2 multi-axis k={k}{tag}"
        s = add_variant(name, m.axes, m.calibrator, m.meta.get("oof"))
        variants[name]["cluster_sizes"] = m.meta["cluster_sizes"]
        if ranking_scores is None:
            ranking_scores = s

    return {
        "model": model_name,
        "casing": casing,
        "balanced_negatives": balance,
        "n": {g: len(ts) for g, ts in groups.items()},
        "variants": variants,
        "_ranking": {
            "pos_te": list(zip(groups["pos_te"], map(float, ranking_scores["pos_te"]), strict=True)),
            "hard_te": list(zip(groups["hard_te"], map(float, ranking_scores["hard_te"]), strict=True)),
            "gen_te": list(zip(groups["gen_te"], map(float, ranking_scores["gen_te"]), strict=True)),
            "topic_te": list(zip(groups["topic_te"], map(float, ranking_scores["topic_te"]), strict=True)),
            "quotes": list(
                zip(groups.get("quotes", []), map(float, ranking_scores.get("quotes", [])), strict=True)
            ),
            "junk": list(
                zip(groups.get("junk", []), map(float, ranking_scores.get("junk", [])), strict=True)
            ),
            "probe": list(zip(groups["probe"], map(float, ranking_scores["probe"]), strict=True)),
        },
    }


def print_report(res: dict, show_ranking: bool = True) -> None:
    n = res["n"]
    print(
        f"model: {res['model']}   casing: {res.get('casing', 'none')}   "
        f"negatives weighted per {'family' if res.get('balanced_negatives', False) else 'line'}"
    )
    print(
        f"train: {n['pos_tr']} taglines / {n['hard_tr']} hard / {n['gen_tr']} generic   "
        f"test: {n['pos_te']} taglines / {n['hard_te']} hard / {n['gen_te']} generic / "
        f"{n['topic_te']} topic-matched"
    )
    for name, v in res["variants"].items():
        print(f"\n[{name}]")
        ci = v["auc_ci95"]

        def fmt(key, g, v=v, ci=ci):
            lo, hi = ci[g]
            return f"{v[key]:.3f} [{lo:.2f},{hi:.2f}]"

        line = (
            f"  AUC vs hard {fmt('auc_vs_hard', 'hard_te')}   generic {fmt('auc_vs_generic', 'gen_te')}   "
            f"topic-matched {fmt('auc_vs_topic_bland', 'topic_te')}   pair win {v['topic_pair_win_rate']:.3f}"
        )
        if v.get("auc_vs_quotes") is not None:
            line += f"\n  quotability control (vs famous non-ad quotes) {fmt('auc_vs_quotes', 'quotes')}"
        if v.get("auc_vs_junk") is not None:
            line += f"   vs junk {fmt('auc_vs_junk', 'junk')}"
        lo, hi = v["topic_pair_win_ci95"]
        line += f"\n  pair win rate 95% Wilson [{lo:.2f},{hi:.2f}]"
        print(line)
        if "pass_rate@0.5" in v:
            rates = "  ".join(f"{g} {r:.2f}" for g, r in v["pass_rate@0.5"].items())
            extra = (
                f"   junk after input flags {v['junk_pass_rate_after_flags']:.2f}"
                if "junk_pass_rate_after_flags" in v
                else ""
            )
            print(f"  PASS rate at p>=0.5: {rates}{extra}")
        print(
            "  by length: "
            + "   ".join(f"{b} n={d['n']} AUC {d['auc_vs_hard']:.3f}" for b, d in v["by_length"].items())
        )
        print(
            f"  held-out taglines mean {v['mean']['pos_te']:+.4f} (min {v['min_pos']:+.4f})   "
            f"hard mean {v['mean']['hard_te']:+.4f} (max {v['max_hard']:+.4f})"
        )
        if "cluster_sizes" in v:
            print(f"  cluster sizes: {v['cluster_sizes']}")
        if "calibration" in v:
            c = v["calibration"]
            print(
                f"  calibration (test): log-loss {c['test_log_loss']:.3f}  brier {c['test_brier']:.3f}  "
                f"acc@0.5 {c['test_accuracy@0.5']:.3f}  threshold {c['threshold@0.5']:+.4f}"
            )
        lowest = "; ".join(f"{d['score']:+.3f} {d['text'][:50]}" for d in v["lowest_famous"][:3])
        print(f"  lowest-scoring famous: {lowest}")
        print("  probes: " + "; ".join(f"{s:+.3f} {p}" for p, s in v["probes"].items()))

    if show_ranking:
        r = res["_ranking"]
        label = {
            "pos_te": "FAMOUS",
            "hard_te": "bland ",
            "gen_te": "prose ",
            "topic_te": "topic ",
            "quotes": "quote ",
            "junk": "junk  ",
            "probe": "probe ",
        }
        rows = sorted(((t, g, s) for g, rows_ in r.items() for t, s in rows_), key=lambda x: -x[2])
        print("\nHeld-out ranking by the default M2 axis (highest first):")
        for rank, (t, g, s) in enumerate(rows, 1):
            print(f"  {rank:3d}. [{label[g]}] {s:+.4f}  {t}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", default="1,2,3,4", help="comma list of multi-axis sizes to evaluate")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--json", help="write the full report to this path")
    ap.add_argument("--no-ranking", action="store_true")
    ap.add_argument("--casing", choices=CASINGS, default="none", help="normalize case before embedding")
    ap.add_argument("--no-balance", action="store_true", help="weight negatives per line, not per family")
    args = ap.parse_args(argv)
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    res = run(args.model, ks, Path(args.data_dir), casing=args.casing, balance=not args.no_balance)
    print_report(res, show_ranking=not args.no_ranking)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

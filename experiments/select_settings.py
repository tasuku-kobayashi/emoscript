"""Choose k, text normalization and calibrator penalty on the TRAIN split only.

The held-out test split must not be used to pick settings, so this script runs
a brand-grouped 5-fold cross-validation over the training positives and
negatives. For every (casing, k, l2) it reports the out-of-fold AUC and the
log-loss of probabilities produced by a calibrator that was itself fitted on
inner out-of-fold scores of the training folds (nested, so nothing leaks).

    python experiments/select_settings.py [--model NAME] [--casings none,lower,lower_nopunct]
                                          [--k 1,2,3] [--l2 0.1,1,10] [--json out.json]

Pick the row with the best log-loss (ties: higher AUC, then smaller k), then
freeze that choice as the library default and evaluate it once on the test split.
Outer folds are brand-grouped; the inner folds that fit each fold's calibrator
are random (a brand-grouped inner split changed the threshold by 0.0001).
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
    auc,
    family_weights,
    fit_multi_axis,
    l2_normalize,
    log_loss,
    score,
)
from emoscript.dataio import DATA_DIR, load_corpus, texts  # noqa: E402
from emoscript.embed import DEFAULT_MODEL, make_embed_fn  # noqa: E402
from emoscript.pipeline import cross_fit_scores, negative_family  # noqa: E402
from emoscript.text import normalizer  # noqa: E402


def grouped_folds(groups: list, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Assign whole groups (brands) to folds, balancing fold sizes greedily."""
    uniq = list(dict.fromkeys(groups))
    rng.shuffle(uniq)
    sizes = [0] * n_folds
    fold_of: dict = {}
    for g in sorted(uniq, key=lambda g: -groups.count(g)):
        f = int(np.argmin(sizes))
        fold_of[g] = f
        sizes[f] += groups.count(g)
    return [np.array([i for i, g in enumerate(groups) if fold_of[g] == f]) for f in range(n_folds)]


def run(
    model_name: str,
    casings: list[str],
    ks: list[int],
    l2s: list[float],
    data_dir: Path,
    n_folds: int = 5,
    seed: int = 0,
    embed_fn=None,
    balances: tuple[bool, ...] = (False,),
) -> list[dict]:
    corpus = load_corpus(data_dir)
    embed_fn = embed_fn or make_embed_fn(model_name)
    pos_raw = texts(corpus.pos_train)
    neg_rows = corpus.negatives_train()
    neg_raw = texts(neg_rows)
    fam_w = family_weights([negative_family(r) for r in neg_rows])
    brands = [r["brand"] for r in corpus.pos_train]
    rng = np.random.default_rng(seed)
    pos_folds = grouped_folds(brands, n_folds, rng)
    neg_idx = rng.permutation(len(neg_raw))
    neg_folds = [neg_idx[i::n_folds] for i in range(n_folds)]
    rows: list[dict] = []
    for casing in casings:
        norm = normalizer(casing)
        E_pos = l2_normalize(embed_fn(norm(pos_raw)))
        E_neg = l2_normalize(embed_fn(norm(neg_raw)))
        for k, balance in [(k, b) for k in ks for b in balances]:
            W = fam_w if balance else None
            oof_s_pos = np.full(len(E_pos), np.nan)
            oof_s_neg = np.full(len(E_neg), np.nan)
            oof_p = {l2: (np.full(len(E_pos), np.nan), np.full(len(E_neg), np.nan)) for l2 in l2s}
            for f in range(n_folds):
                hp, hn = pos_folds[f], neg_folds[f]
                tp = np.setdiff1d(np.arange(len(E_pos)), hp)
                tn = np.setdiff1d(np.arange(len(E_neg)), hn)
                w_tn = None if W is None else W[tn]
                axes, _ = fit_multi_axis(E_pos[tp], E_neg[tn], k, seed=seed, neg_weights=w_tn)
                oof_s_pos[hp] = score(E_pos[hp], axes)
                oof_s_neg[hn] = score(E_neg[hn], axes)
                # inner cross-fit on the training part only -> calibrator for this fold
                in_p, in_n = cross_fit_scores(
                    E_pos[tp], E_neg[tn], k, n_folds=n_folds, seed=seed, neg_weights=w_tn
                )
                for l2 in l2s:
                    cal = Calibrator(l2=l2).fit(in_p, in_n)
                    oof_p[l2][0][hp] = cal.predict_proba(score(E_pos[hp], axes))
                    oof_p[l2][1][hn] = cal.predict_proba(score(E_neg[hn], axes))
            cv_auc = auc(oof_s_pos, oof_s_neg)
            y = np.r_[np.ones(len(E_pos)), np.zeros(len(E_neg))]
            for l2 in l2s:
                p = np.r_[oof_p[l2][0], oof_p[l2][1]]
                # balanced log-loss: mean of the per-class log-losses, so the class ratio does not dominate
                ll_pos = log_loss(oof_p[l2][0], np.ones(len(E_pos)))
                ll_neg = log_loss(oof_p[l2][1], np.zeros(len(E_neg)))
                rows.append(
                    {
                        "casing": casing,
                        "k": k,
                        "balance": balance,
                        "l2": l2,
                        "cv_auc": cv_auc,
                        "cv_log_loss": log_loss(p, y),
                        "cv_balanced_log_loss": 0.5 * (ll_pos + ll_neg),
                        "cv_pos_pass_rate": float((oof_p[l2][0] >= 0.5).mean()),
                        "cv_neg_pass_rate": float((oof_p[l2][1] >= 0.5).mean()),
                    }
                )
    return rows


def print_table(rows: list[dict]) -> None:
    print(
        f"{'casing':14s} {'k':>2s} {'bal':>3s} {'l2':>5s} {'AUC':>6s} {'logloss':>8s} {'bal-ll':>7s} "
        f"{'pos pass':>9s} {'neg pass':>9s}"
    )
    for r in sorted(rows, key=lambda r: (r["cv_balanced_log_loss"], -r["cv_auc"], r["k"])):
        print(
            f"{r['casing']:14s} {r['k']:2d} {'y' if r.get('balance') else 'n':>3s} {r['l2']:5g} "
            f"{r['cv_auc']:6.3f} {r['cv_log_loss']:8.3f} "
            f"{r['cv_balanced_log_loss']:7.3f} {r['cv_pos_pass_rate']:9.3f} {r['cv_neg_pass_rate']:9.3f}"
        )
    best = min(rows, key=lambda r: (r["cv_balanced_log_loss"], -r["cv_auc"], r["k"]))
    print(
        f"\nbest by balanced log-loss: casing={best['casing']} k={best['k']} "
        f"balance={best.get('balance', False)} l2={best['l2']}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--casings", default="none,lower,canonical")
    ap.add_argument("--k", default="1,2,3")
    ap.add_argument("--l2", default="0.1,1,10")
    ap.add_argument("--balance", default="0,1", help="comma list of 0/1: equal weight per negative family")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rows = run(
        args.model,
        [c.strip() for c in args.casings.split(",") if c.strip()],
        [int(x) for x in args.k.split(",") if x.strip()],
        [float(x) for x in args.l2.split(",") if x.strip()],
        Path(args.data_dir),
        balances=tuple(bool(int(x)) for x in args.balance.split(",") if x.strip()),
    )
    print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

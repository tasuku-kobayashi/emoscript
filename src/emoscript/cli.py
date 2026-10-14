"""Command-line interface.

emoscript fit --out axis.npz              # fit the axis (+ calibrator) from the bundled corpus
emoscript score "Just ship it." --axis axis.npz
emoscript score "Just ship it." --explain      # + nearest famous taglines, matched axis
emoscript score --file c.txt --prior 0.05 --brief "meal-kit delivery for busy parents"
emoscript score --file candidates.txt --sort --json
emoscript eval --axis axis.npz            # held-out AUC / calibration report
emoscript corpus-check                    # corpus integrity (no model needed)
emoscript info --axis axis.npz            # what is in an axis file
emoscript fetch-axis                      # download a prebuilt axis from GitHub Releases
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .dataio import DATA_DIR, corpus_warnings, validate_corpus
from .embed import DEFAULT_MODEL, make_embed_fn
from .pipeline import AxisModel, embed_valid, evaluate, fit_from_corpus, web_export
from .text import BLOCKING_FLAGS, CASINGS, input_flags

DEFAULT_REPO = "tasuku-kobayashi/emoscript"


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def _finite(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


# --------------------------------------------------------------------------- commands
NEGATIVE_KINDS = ("hard", "generic")


def cmd_fit(args) -> int:
    negatives = tuple(x.strip() for x in args.negatives.split(",") if x.strip())
    unknown = [x for x in negatives if x not in NEGATIVE_KINDS]
    if unknown or not negatives:
        raise SystemExit(f"--negatives: unknown kind(s) {unknown}; choose from {', '.join(NEGATIVE_KINDS)}")
    embed_fn = make_embed_fn(args.model)
    model = fit_from_corpus(
        embed_fn,
        args.data_dir,
        k=args.k,
        negatives=negatives,
        calibrate=not args.no_calibrate,
        n_folds=args.folds,
        seed=args.seed,
        model_name=args.model,
        casing=args.casing,
        balance_negatives=not args.no_balance_negatives,
        sense=not args.no_sense,
    )
    saved = model.save(args.out)
    oof = model.meta.get("oof", {})
    print(
        f"axis saved to {saved} (k={model.k}, dim={model.axes.shape[1]}, "
        f"model={args.model}, casing={model.casing})"
    )
    if oof:
        thr = oof.get("threshold@0.5")
        thr_s = f"{thr:+.4f}" if thr is not None else "n/a (two-feature calibrator)"
        print(
            f"  out-of-fold: AUC {oof['auc']:.3f}  log-loss {oof['log_loss']:.3f}  "
            f"acc@0.5 {oof['accuracy@0.5']:.3f}  threshold(p=0.5) {thr_s}"
        )
        g = model.meta.get("sense_gate_oof")
        if g:
            print(
                f"  sense gate (out-of-fold): famous pass {g['famous_pass_rate']:.2f}  "
                f"nonsense pass {g['nonsense_pass_rate']:.2f}  "
                f"other negatives pass {g['other_negative_pass_rate']:.2f}"
            )
    for w in model.meta.get("warnings", []):
        print(f"  WARNING: {w}")
    return 0


def _read_texts(args) -> list[str]:
    texts: list[str] = list(args.taglines or [])
    if args.file == "-":
        texts += [line.rstrip("\n") for line in sys.stdin if line.strip()]
    elif args.file:
        with open(args.file, encoding="utf-8-sig") as fh:
            texts += [line.rstrip("\n") for line in fh if line.strip()]
    if not texts:
        raise SystemExit("score: give taglines as arguments, --file PATH, or --file - for stdin")
    return texts


def cmd_score(args) -> int:
    if not 0 < args.threshold < 1:
        raise SystemExit("--threshold must be strictly between 0 and 1")
    if not 0 < args.prior < 1:
        raise SystemExit("--prior must be strictly between 0 and 1")
    if args.min_rel is not None and not args.brief:
        raise SystemExit("--min-rel needs --brief")
    if args.top_rel is not None and (not args.brief or not 0 < args.top_rel <= 1):
        raise SystemExit("--top-rel needs --brief and a fraction in (0, 1]")
    if args.brief is not None and not any(ch.isalpha() for ch in args.brief):
        raise SystemExit("--brief must contain words")
    model = AxisModel.load(args.axis)
    texts = _read_texts(args)
    embed_fn = make_embed_fn(model.model_name, model.prefix)

    ok, E = embed_valid(model, texts, embed_fn)  # embedded once, reused for score / explain / brief
    scores = np.full(len(texts), np.nan)
    probs = np.full(len(texts), np.nan) if model.calibrator is not None else None
    if ok.any():
        scores[ok] = model.score(E)
        if probs is not None:
            probs[ok] = model.proba_from_embeddings(E, prior=args.prior)
    relevance = None
    if args.brief:
        relevance = np.full(len(texts), np.nan)
        if ok.any():
            relevance[ok] = E @ model.embed([args.brief], embed_fn)[0]
    sense_p = np.full(len(texts), np.nan)
    gate_ok = np.ones(len(texts), dtype=bool)
    if ok.any() and model.sense_gate is not None:
        sense_p[ok] = model.sense_proba(E)
        if not args.no_sense_gate:
            gate_ok[ok] = model.passes_sense_gate(E)
    rel_cut = None
    if relevance is not None and args.top_rel is not None:
        valid_rel = relevance[ok]
        if len(valid_rel):
            rel_cut = float(np.quantile(valid_rel, 1 - args.top_rel))
    explanations = model.explain(E, n=args.neighbors) if args.explain and ok.any() else None
    exp_index = np.cumsum(ok) - 1
    if args.no_famous_check:
        near = [None] * len(texts)
    else:
        near = model.near_famous(texts)
        if not model.famous_texts:
            print(
                "note: this axis file carries no famous-line catalogue; the copy check is off",
                file=sys.stderr,
            )
    flags = [input_flags(t) for t in texts]

    rows = []
    for i, t in enumerate(texts):
        skipped = not ok[i]
        row = {"text": t, "score": _finite(scores[i])}
        if skipped:
            row["skipped"] = "no letters"
        if explanations is not None and not skipped:
            ex = explanations[exp_index[i]]
            row["axis"] = ex["axis"]
            row["nearest"] = [{"text": a, "cos": b} for a, b in ex["nearest"]]
        if relevance is not None:
            row["brief_relevance"] = _finite(relevance[i])
        if model.sense_gate is not None:
            row["sense"] = _finite(sense_p[i])
        if flags[i]:
            row["flags"] = flags[i]
        if near[i] is not None:
            row["near_famous"] = near[i]
        # A row passes only if the score says so AND nothing disqualifies it: a blocking
        # input flag, a verbatim copy of a famous line, or (with --brief) low relevance.
        blocked = sorted(BLOCKING_FLAGS & set(flags[i]))
        if near[i] is not None and near[i]["severity"] == "copy":
            blocked.append("near_famous:copy")
        if relevance is not None and args.min_rel is not None and not skipped and relevance[i] < args.min_rel:
            blocked.append("off_brief")
        if rel_cut is not None and not skipped and relevance[i] < rel_cut:
            blocked.append("off_brief")
        if not gate_ok[i]:
            blocked.append("sense_gate")
        if probs is not None:
            row["prob"] = _finite(probs[i])
            row["pass"] = (not skipped) and bool(probs[i] >= args.threshold) and not blocked
        if blocked:
            row["blocked_by"] = blocked
        rows.append(row)
    if args.sort:
        # with a brief, rank by style AND relevance, z-scored over the unblocked rows of this batch
        if relevance is not None:
            clean = [r for r in rows if r["score"] is not None and "blocked_by" not in r]
            s = np.array([r["score"] for r in clean])
            rel = np.array([r["brief_relevance"] for r in clean])
            mu_s, sd_s = (s.mean(), s.std()) if len(s) > 1 else (0.0, 1.0)
            mu_r, sd_r = (rel.mean(), rel.std()) if len(rel) > 1 else (0.0, 1.0)
            for r in rows:
                if r["score"] is not None and r.get("brief_relevance") is not None:
                    r["rank_score"] = float(
                        (r["score"] - mu_s) / (sd_s + 1e-12) + (r["brief_relevance"] - mu_r) / (sd_r + 1e-12)
                    )
        key_name = "rank_score" if relevance is not None else "score"
        rows.sort(
            key=lambda r: (
                "blocked_by" in r or r["score"] is None,
                -(r.get(key_name) if r.get(key_name) is not None else float("-inf")),
                -(r["score"] if r["score"] is not None else float("-inf")),
            )
        )
    if args.json:
        _print_json(rows)
        return 0
    for r in rows:
        rel = f"  rel={r['brief_relevance']:.2f}" if r.get("brief_relevance") is not None else ""
        if r.get("sense") is not None:
            rel += f"  sense={r['sense']:.2f}"
        if r["score"] is None:
            print(f"   n/a   skipped ({r['skipped']})  {r['text']}")
        elif "prob" in r:
            print(
                f"{r['score']:+.4f}  p={r['prob']:.2f}  {'PASS' if r['pass'] else 'fail'}{rel}  {r['text']}"
            )
        else:
            print(f"{r['score']:+.4f}{rel}  {r['text']}")
        notes = []
        if "near_famous" in r:
            nf = r["near_famous"]
            who = f" ({nf['brand']})" if nf.get("brand") else ""
            label = {
                "copy": "copies famous line",
                "parody": "parodies famous line",
                "possible": "possibly riffs on",
            }
            notes.append(f'{label[nf["severity"]]}: "{nf["text"]}"{who}')
        if r.get("flags"):
            notes.append("flags: " + ", ".join(r["flags"]))
        if r.get("blocked_by"):
            notes.append("not passing: " + ", ".join(r["blocked_by"]))
        if notes:
            print("          ! " + "; ".join(notes))
        if "nearest" in r:
            axis_note = f" (axis {r['axis']})" if model.k > 1 else ""
            near_s = "; ".join(f"{n['text']} ({n['cos']:.2f})" for n in r["nearest"])
            print(f"          nearest famous{axis_note}: {near_s}")
    return 0


def cmd_eval(args) -> int:
    model = AxisModel.load(args.axis)
    embed_fn = make_embed_fn(model.model_name, model.prefix)
    report = evaluate(model, embed_fn, args.data_dir)
    report["axis"] = str(args.axis)
    report["k"] = model.k
    report["model"] = model.model_name
    if args.json:
        _print_json(report)
        return 0
    print(f"[{args.axis}] k={model.k} model={model.model_name}")
    for g, v in report["auc"].items():
        lo, hi = report["auc_ci95"][g]
        print(f"  AUC held-out taglines vs {g:<12s} {v:.3f}  [{lo:.3f}, {hi:.3f}]  (n={report['n'][g]})")
    if "excluding_sister_brands" in report:
        ex = report["excluding_sister_brands"]
        aucs = "  ".join(f"{g} {v:.3f}" for g, v in ex["auc"].items())
        tb = ex.get("auc_topic_bland_paired")
        tb_s = f"  topic_bland {tb:.3f}" if tb is not None else ""
        print(f"  without {ex['n_sister']} sister-brand lines (n={ex['n_pos']}): {aucs}{tb_s}")
    if "good_copy" in report:
        gc = report["good_copy"]
        extra = f"  by probability {gc['auc_proba_vs_junk']:.3f}" if "auc_proba_vs_junk" in gc else ""
        print(
            f"  good (unknown) copy vs junk      {gc['auc_vs_junk']:.3f}{extra}   "
            f"vs hard {gc['auc_vs_hard']:.3f}   famous vs good {gc['auc_famous_vs_good']:.3f}"
        )
        if "by_register" in gc:
            reg = "  ".join(f"{k} {v['pass_rate']:.2f} (n={v['n']})" for k, v in gc["by_register"].items())
            print(f"  good copy PASS by register:      {reg}")
    if "contrast_pair_win_rate" in report:
        lo, hi = report["contrast_pair_win_ci95"]
        print(
            f"  topic-matched pair win rate      {report['contrast_pair_win_rate']:.3f}  [{lo:.3f}, {hi:.3f}]"
        )
    if "pass_rate" in report:
        rates = "  ".join(f"{g} {v:.2f}" for g, v in report["pass_rate"].items())
        print(f"  PASS rate at p>=0.5:             {rates}")
        if "sense_gate_pass_rate" in report:
            rates = "  ".join(f"{g} {v:.2f}" for g, v in report["sense_gate_pass_rate"].items())
            print(f"  sense gate alone passes:         {rates}")
        rates = "  ".join(f"{g} {v:.2f}" for g, v in report["pass_rate_after_guards"].items())
        print(f"  ... after input flags / famous:  {rates}")
    for name, v in report.get("auc_vs_hard_by_length", {}).items():
        print(f"  AUC vs hard, taglines {name:<10s} {v['auc']:.3f}  (n={v['n']})")
    if "calibration" in report:
        c = report["calibration"]
        thr = c["threshold@0.5"]
        thr_s = f"{thr:+.4f}" if thr is not None else "n/a"
        print(
            f"  calibration (vs {'+'.join(c['negatives'])}): log-loss {c['log_loss']:.3f}  "
            f"brier {c['brier']:.3f}  acc@0.5 {c['accuracy@0.5']:.3f}  threshold {thr_s}"
        )
    return 0


def cmd_corpus_check(args) -> int:
    problems = validate_corpus(args.data_dir)
    if problems:
        print(f"{len(problems)} problem(s) in {args.data_dir}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"corpus OK: {args.data_dir}")
    warnings = corpus_warnings(args.data_dir)
    if warnings:
        print(f"{len(warnings)} warning(s) (not failures):")
        for w in warnings:
            print(f"  ~ {w}")
    return 0


def cmd_info(args) -> int:
    model = AxisModel.load(args.axis)
    info = {
        "axis": str(args.axis),
        "k": model.k,
        "dim": int(model.axes.shape[1]),
        "model": model.model_name,
        "prefix": model.prefix,
        "casing": model.casing,
        "calibrated": model.calibrator is not None,
        "threshold@0.5": model.threshold_for(0.5),
        "exemplars": len(model.exemplar_texts),
        "sense_axis": model.sense_axis is not None,
        "meta": model.meta,
    }
    _print_json(info)
    return 0


def cmd_export_web(args) -> int:
    model = AxisModel.load(args.axis)
    blob = web_export(model, web_model=args.web_model, pooling=args.pooling, revision=args.web_revision)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(blob, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    emb = blob["embedding"]
    print(
        f"web model written to {out} (k={model.k}, dim={emb['dim']}, {emb['web_model']} / "
        f"{emb['pooling']} pooling, {len(blob['famous'])} famous lines)"
    )
    return 0


def _http_get_json(url: str) -> dict:
    import urllib.request

    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - https GitHub API
        return json.load(r)


def _http_download(url: str, out: Path) -> None:
    import shutil
    import urllib.request

    with urllib.request.urlopen(url, timeout=600) as r, open(out, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)


def cmd_fetch_axis(args) -> int:
    """Download a prebuilt axis attached to a GitHub Release."""
    ref = "latest" if args.tag == "latest" else f"tags/{args.tag}"
    release = _http_get_json(f"https://api.github.com/repos/{args.repo}/releases/{ref}")
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    name = args.name or f"emoscript-axis-k{args.k}-canonical-multilingual-e5-large.npz"
    if name not in assets:
        print(f"no asset named {name!r} in release {release.get('tag_name', args.tag)}; available:")
        for a in sorted(assets):
            print(f"  - {a}")
        return 1
    out = Path(args.out)
    _http_download(assets[name], out)
    model = AxisModel.load(out)
    print(f"saved {out} (release {release.get('tag_name')}, k={model.k}, model={model.model_name})")
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="emoscript", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", action="version", version=f"emoscript {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fit", help="fit the axis (and calibrator) from the corpus")
    p.add_argument("--k", type=int, default=1, help="number of axes (k-means clusters of positives)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="fastembed model name")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="axis.npz")
    p.add_argument("--negatives", default="hard,generic", help="comma list of: hard, generic")
    p.add_argument("--no-calibrate", action="store_true", help="skip the cross-fitted calibrator")
    p.add_argument("--folds", type=int, default=5, help="folds for out-of-fold calibration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-sense",
        action="store_true",
        help="drop the second 'is this a message' axis; calibrate on the style score alone",
    )
    p.add_argument(
        "--no-balance-negatives",
        action="store_true",
        help="weight the negative centroid per line instead of equally per negative family",
    )
    p.add_argument(
        "--casing",
        choices=CASINGS,
        default="canonical",
        help="text normalization before embedding; 'canonical' (default) removes case, terminal "
        "punctuation and LLM/markup wrappers",
    )
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("score", help="score one or more taglines")
    p.add_argument("taglines", nargs="*")
    p.add_argument("--axis", default="axis.npz")
    p.add_argument("--file", help="read one tagline per line from PATH (or '-' for stdin)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--sort", action="store_true", help="highest score first")
    p.add_argument("--threshold", type=float, default=0.5, help="probability for PASS")
    p.add_argument("--explain", action="store_true", help="show the nearest famous taglines and matched axis")
    p.add_argument("--neighbors", type=int, default=3, help="how many nearest taglines with --explain")
    p.add_argument(
        "--prior",
        type=float,
        default=0.5,
        help="base rate of great copy among your candidates (default 0.5; try 0.05 for raw LLM output)",
    )
    p.add_argument(
        "--brief",
        help="product / campaign brief; adds rel= cosine(line, brief) and, with --sort, "
        "ranks by style + relevance",
    )
    p.add_argument(
        "--min-rel",
        type=float,
        default=None,
        help="with --brief: lines whose relevance is below this do not PASS (e5 cosines sit around "
        "0.75-0.90; read the rel= column of a few known-good lines to pick a value)",
    )
    p.add_argument("--no-famous-check", action="store_true", help="skip the near-famous-line warning")
    p.add_argument(
        "--no-sense-gate",
        action="store_true",
        help="do not require P(message) >= 0.5 on the sense axis for PASS (the gate is what stops nonsense)",
    )
    p.add_argument(
        "--top-rel",
        type=float,
        default=None,
        help="with --brief: only the top FRACTION of this batch by relevance can PASS "
        "(relative to the batch, unlike --min-rel)",
    )
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("eval", help="evaluate a fitted axis on the held-out splits")
    p.add_argument("--axis", default="axis.npz")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("corpus-check", help="validate corpus integrity (no model needed)")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(fn=cmd_corpus_check)

    p = sub.add_parser("info", help="describe an axis file")
    p.add_argument("--axis", default="axis.npz")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("export-web", help="write an axis as JSON for the browser app in web/")
    p.add_argument("--axis", default="axis.npz")
    p.add_argument("--out", default="web/model.json")
    p.add_argument(
        "--web-model", help="transformers.js model id (default: known mapping for the axis's model)"
    )
    p.add_argument(
        "--pooling", choices=("cls", "mean"), help="pooling used by that model (default: known mapping)"
    )
    p.add_argument("--web-revision", help="hub revision to pin (default: known mapping, else 'main')")
    p.set_defaults(fn=cmd_export_web)

    p = sub.add_parser("fetch-axis", help="download a prebuilt axis from GitHub Releases")
    p.add_argument("--repo", default=DEFAULT_REPO, help="owner/name")
    p.add_argument("--tag", default="latest", help="release tag (default: latest)")
    p.add_argument("--k", type=int, default=1, help="which prebuilt variant (k=1 default, k=2)")
    p.add_argument("--name", help="exact asset file name (overrides --k)")
    p.add_argument("--out", default="axis.npz")
    p.set_defaults(fn=cmd_fetch_axis)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if hasattr(args, "data_dir"):
        args.data_dir = Path(args.data_dir)
    return int(args.fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

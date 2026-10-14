"""Command-line interface.

emoscript fit --out axis.npz              # fit axes (+ calibrator) from the bundled corpus (k=2, lower-cased)
emoscript score "Just ship it." --axis axis.npz
emoscript score "Just ship it." --explain      # + nearest famous taglines, matched axis
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

from . import __version__
from .dataio import DATA_DIR, validate_corpus
from .embed import DEFAULT_MODEL, make_embed_fn
from .pipeline import AxisModel, evaluate, fit_from_corpus, score_texts
from .text import CASINGS

DEFAULT_REPO = "tasuku-kobayashi/emoscript"


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- commands
def cmd_fit(args) -> int:
    negatives = tuple(x.strip() for x in args.negatives.split(",") if x.strip())
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
    )
    model.save(args.out)
    oof = model.meta.get("oof", {})
    print(
        f"axis saved to {args.out} (k={model.k}, dim={model.axes.shape[1]}, "
        f"model={args.model}, casing={model.casing})"
    )
    if oof:
        print(
            f"  out-of-fold: AUC {oof['auc']:.3f}  log-loss {oof['log_loss']:.3f}  "
            f"acc@0.5 {oof['accuracy@0.5']:.3f}  threshold(p=0.5) {oof['threshold@0.5']:+.4f}"
        )
    return 0


def _read_texts(args) -> list[str]:
    texts: list[str] = list(args.taglines or [])
    if args.file == "-":
        texts += [line.rstrip("\n") for line in sys.stdin if line.strip()]
    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            texts += [line.rstrip("\n") for line in fh if line.strip()]
    if not texts:
        raise SystemExit("score: give taglines as arguments, --file PATH, or --file - for stdin")
    return texts


def cmd_score(args) -> int:
    model = AxisModel.load(args.axis)
    texts = _read_texts(args)
    embed_fn = make_embed_fn(model.model_name, model.prefix)
    scores, probs = score_texts(model, texts, embed_fn)
    explanations = model.explain(model.embed(texts, embed_fn), n=args.neighbors) if args.explain else None
    rows = []
    for i, t in enumerate(texts):
        row = {"text": t, "score": float(scores[i])}
        if probs is not None:
            row["prob"] = float(probs[i])
            row["pass"] = bool(probs[i] >= args.threshold)
        if explanations is not None:
            row["axis"] = explanations[i]["axis"]
            row["nearest"] = [{"text": a, "cos": b} for a, b in explanations[i]["nearest"]]
        rows.append(row)
    if args.sort:
        rows.sort(key=lambda r: -r["score"])
    if args.json:
        _print_json(rows)
        return 0
    for r in rows:
        if "prob" in r:
            print(f"{r['score']:+.4f}  p={r['prob']:.2f}  {'PASS' if r['pass'] else 'fail'}  {r['text']}")
        else:
            print(f"{r['score']:+.4f}  {r['text']}")
        if "nearest" in r:
            axis_note = f" (axis {r['axis']})" if model.k > 1 else ""
            near = "; ".join(f"{n['text']} ({n['cos']:.2f})" for n in r["nearest"])
            print(f"          nearest famous{axis_note}: {near}")
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
    if "contrast_pair_win_rate" in report:
        print(f"  topic-matched pair win rate      {report['contrast_pair_win_rate']:.3f}")
    for name, v in report.get("auc_vs_hard_by_length", {}).items():
        print(f"  AUC vs hard, taglines {name:<10s} {v['auc']:.3f}  (n={v['n']})")
    if "calibration" in report:
        c = report["calibration"]
        print(
            f"  calibration: log-loss {c['log_loss']:.3f}  brier {c['brier']:.3f}  "
            f"acc@0.5 {c['accuracy@0.5']:.3f}  threshold {c['threshold@0.5']:+.4f}"
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
        "meta": model.meta,
    }
    _print_json(info)
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
    name = args.name or f"emoscript-axis-k{args.k}-lower-multilingual-e5-large.npz"
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
    p.add_argument("--k", type=int, default=2, help="number of axes (k-means clusters of positives)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="fastembed model name")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="axis.npz")
    p.add_argument("--negatives", default="hard,generic", help="comma list of: hard, generic")
    p.add_argument("--no-calibrate", action="store_true", help="skip the cross-fitted calibrator")
    p.add_argument("--folds", type=int, default=5, help="folds for out-of-fold calibration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--casing",
        choices=CASINGS,
        default="lower",
        help="normalize letter case before embedding; 'lower' (default) removes the Title-Case confound",
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

    p = sub.add_parser("fetch-axis", help="download a prebuilt axis from GitHub Releases")
    p.add_argument("--repo", default=DEFAULT_REPO, help="owner/name")
    p.add_argument("--tag", default="latest", help="release tag (default: latest)")
    p.add_argument("--k", type=int, default=2, help="which prebuilt variant (k=1 or k=2)")
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

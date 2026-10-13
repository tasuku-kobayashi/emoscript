"""Command-line interface.

    copyaxis fit                     # build the axis from the bundled corpus -> axis.npz
    copyaxis score "Just ship it."   # score taglines against a fitted axis
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .axis import fit_diff_axis, l2_normalize, score
from .dataio import DATA_DIR, load_split
from .embed import DEFAULT_MODEL, embed_texts, get_model


def _build_axis(model_name: str, data_dir) -> tuple[np.ndarray, object]:
    model = get_model(model_name)
    pos = [r["text"] for r in load_split("slogans_train", data_dir)]
    hard = [r["text"] for r in load_split("negatives_hard_train", data_dir)]
    generic = [r["text"] for r in load_split("negatives_generic_train", data_dir)]
    E_pos = l2_normalize(embed_texts(pos, model=model))
    E_neg = l2_normalize(embed_texts(hard + generic, model=model))
    return fit_diff_axis(E_pos, E_neg), model


def cmd_fit(args) -> None:
    axis, _ = _build_axis(args.model, args.data_dir)
    np.savez_compressed(args.out, axis=axis, model=args.model)
    print(f"axis saved to {args.out} (dim={axis.shape[0]}, model={args.model})")


def cmd_score(args) -> None:
    blob = np.load(args.axis)
    axis, model_name = blob["axis"], str(blob["model"])
    model = get_model(model_name)
    E = l2_normalize(embed_texts(args.taglines, model=model))
    for text, s in zip(args.taglines, score(E, axis)):
        print(f"{s:+.4f}  {text}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="copyaxis", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="fit the axis from the corpus")
    p_fit.add_argument("--model", default=DEFAULT_MODEL)
    p_fit.add_argument("--data-dir", default=str(DATA_DIR))
    p_fit.add_argument("--out", default="axis.npz")
    p_fit.set_defaults(fn=cmd_fit)

    p_score = sub.add_parser("score", help="score one or more taglines")
    p_score.add_argument("taglines", nargs="+")
    p_score.add_argument("--axis", default="axis.npz")
    p_score.set_defaults(fn=cmd_score)

    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    args.fn(args)


if __name__ == "__main__":
    main()

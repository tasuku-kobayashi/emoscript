"""Write the parity fixture the Node test compares the JavaScript port against.

    python web/test/make_fixture.py --axis axis.npz [--out web/test/fixture.json]

Records, for every line of fixture_texts.txt: the canonical form, input flags
and near-famous match (model-free, from emoscript.text / pipeline), plus the
full `emoscript score --sort --json` rows with and without a brief (needs the
embedding model). Re-run whenever text.py, near_famous or the CLI row logic
changes, or when web/model.json is re-exported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from emoscript.cli import main as cli_main
from emoscript.pipeline import AxisModel
from emoscript.text import canonicalize, input_flags

HERE = Path(__file__).parent
BRIEF = "a breakfast cereal for busy families"


def score_rows(axis: str, texts_file: Path, extra: list[str]) -> list[dict]:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_main(["score", "--axis", axis, "--file", str(texts_file), "--json", "--sort", *extra])
    return json.loads(buf.getvalue())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--out", default=str(HERE / "fixture.json"))
    ap.add_argument("--texts", default=str(HERE / "fixture_texts.txt"))
    args = ap.parse_args()
    texts_file = Path(args.texts)
    texts = texts_file.read_text(encoding="utf-8").splitlines()
    model = AxisModel.load(args.axis)
    near = model.near_famous(texts)
    fixture = {
        "axis_model": model.model_name,
        "casing": model.casing,
        "brief": BRIEF,
        "lines": [
            {"text": t, "canonical": canonicalize(t), "flags": input_flags(t), "near_famous": near[i]}
            for i, t in enumerate(texts)
        ],
        "score_sorted": score_rows(args.axis, texts_file, []),
        "score_brief_top_rel": score_rows(
            args.axis,
            texts_file,
            ["--brief", BRIEF, "--top-rel", "0.5", "--prior", "0.2", "--threshold", "0.3"],
        ),
    }
    Path(args.out).write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"fixture written to {args.out} ({len(texts)} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

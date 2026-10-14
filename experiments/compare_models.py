"""Compare embedding backends on the held-out splits.

For each model: fit the default M2 axis (k=1) and a k=3 multi-axis on the
train splits, report held-out AUCs and the topic-matched pair win rate.

    python experiments/compare_models.py --models BAAI/bge-small-en-v1.5,intfloat/multilingual-e5-large

Models must be supported by fastembed (see TextEmbedding.list_supported_models()).
Each model is downloaded on first use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emoscript.dataio import DATA_DIR  # noqa: E402
from validate_axis import run  # noqa: E402

DEFAULT_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-large",
    "mixedbread-ai/mxbai-embed-large-v1",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--json", help="write all reports to this path")
    args = ap.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    reports = {}
    header = (
        f"{'model':42s} {'variant':24s} {'hard':>6s} {'generic':>8s} "
        f"{'topic':>6s} {'pairwin':>8s} {'long':>6s}"
    )
    print(header)
    print("-" * len(header))
    for m in models:
        try:
            res = run(m, [1, 3], Path(args.data_dir))
        except Exception as e:  # noqa: BLE001 - keep going with the other models
            print(f"{m:42s} FAILED: {e}")
            continue
        reports[m] = res
        res_lower = run(m, [1], Path(args.data_dir), casing="lower")
        reports[m + " (lower)"] = res_lower
        m2 = "M2 diff (hard+generic, balanced families)"
        rows = [(name, res["variants"][name]) for name in (m2, "M2 multi-axis k=3, balanced families")] + [
            ("M2 k=1, lower-cased", res_lower["variants"][m2])
        ]
        for name, v in rows:
            name = name.replace(", balanced families", "")
            long_auc = v["by_length"].get("long>=8w", {}).get("auc_vs_hard", float("nan"))
            print(
                f"{m:42s} {name:24s} {v['auc_vs_hard']:6.3f} {v['auc_vs_generic']:8.3f} "
                f"{v['auc_vs_topic_bland']:6.3f} {v['topic_pair_win_rate']:8.3f} {long_auc:6.3f}"
            )
    if args.json:
        for r in reports.values():
            r.pop("_ranking", None)
        Path(args.json).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreports written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

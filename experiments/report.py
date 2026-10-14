"""Render one or more validate_axis.py JSON reports as a Markdown table.

python experiments/validate_axis.py --json none.json
python experiments/validate_axis.py --json lower.json --casing lower
python experiments/report.py none.json lower.json > docs/results.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt_ci(v: float | None, ci: tuple[float, float] | None) -> str:
    if v is None:
        return "–"
    if ci is None:
        return f"{v:.3f}"
    return f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def render(reports: list[dict]) -> str:
    out: list[str] = []
    for res in reports:
        n = res["n"]
        out.append(f"### {res['model']} · casing={res.get('casing', 'none')}")
        out.append("")
        out.append(
            f"train {n['pos_tr']} taglines / {n['hard_tr']} hard / {n['gen_tr']} generic · "
            f"test {n['pos_te']} taglines / {n['hard_te']} hard / {n['gen_te']} generic / "
            f"{n.get('topic_te', 0)} topic-matched / {n.get('quotes', 0)} quotes / {n.get('junk', 0)} junk"
        )
        out.append("")
        out.append(
            "| variant | AUC vs hard | vs generic | vs topic-matched | pair win | vs quotes | vs junk | "
            "AUC long ≥8w | PASS pos / bland / junk | log-loss |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, v in res["variants"].items():
            ci = v.get("auc_ci95", {})
            long_auc = v["by_length"].get("long>=8w", {}).get("auc_vs_hard")
            cal = v.get("calibration")
            pr = v.get("pass_rate@0.5")
            if pr:
                bland = [pr[g] for g in ("hard_te", "gen_te", "topic_te") if g in pr]
                junk_cell = f"{pr.get('junk', float('nan')):.2f}"
                if "junk_pass_rate_after_flags" in v:
                    junk_cell += f" ({v['junk_pass_rate_after_flags']:.2f} after flags)"
                pass_cell = f"{pr['pos_te']:.2f} / {sum(bland) / len(bland):.2f} / {junk_cell}"
            else:
                pass_cell = "–"
            win = f"{v['topic_pair_win_rate']:.3f}"
            if "topic_pair_win_ci95" in v:
                lo, hi = v["topic_pair_win_ci95"]
                win += f" [{lo:.2f}, {hi:.2f}]"
            out.append(
                "| "
                + " | ".join(
                    [
                        name,
                        fmt_ci(v["auc_vs_hard"], ci.get("hard_te")),
                        fmt_ci(v["auc_vs_generic"], ci.get("gen_te")),
                        fmt_ci(v["auc_vs_topic_bland"], ci.get("topic_te")),
                        win,
                        fmt_ci(v.get("auc_vs_quotes"), ci.get("quotes")),
                        fmt_ci(v.get("auc_vs_junk"), ci.get("junk")),
                        f"{long_auc:.3f}" if long_auc is not None else "–",
                        pass_cell,
                        f"{cal['test_log_loss']:.3f}" if cal else "–",
                    ]
                )
                + " |"
            )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    paths = (argv if argv is not None else sys.argv[1:]) or []
    if not paths:
        print(__doc__)
        return 2
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    print(render(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())

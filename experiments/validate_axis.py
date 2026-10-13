"""Held-out validation of the great-copy axis.

Fits three axis variants on the *_train splits and evaluates rank AUC on the
*_test splits (never used for fitting):
  M1  positive centroid only
  M2  diff axis: positives - (hard + generic negatives)   [recommended]
  M3  diff axis: positives - hard negatives only

Run from the repo root:
    python experiments/validate_axis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from copyaxis.axis import auc, fit_diff_axis, fit_mean_axis, l2_normalize
from copyaxis.dataio import load_split
from copyaxis.embed import embed_texts, get_model

PROBES = [
    "Think fresh.",                       # coined, punchy
    "We sell high quality furniture at low prices.",  # bland ad-speak
    "To be or not to be, that is the question.",  # great writing, not ad copy
]


def main():
    model = get_model()
    texts, groups = [], []

    def add(rows, group):
        for r in rows:
            texts.append(r["text"])
            groups.append(group)

    add(load_split("slogans_train"), "pos_tr")
    add(load_split("negatives_hard_train"), "hard_tr")
    add(load_split("negatives_generic_train"), "gen_tr")
    add(load_split("slogans_test"), "pos_te")
    add(load_split("negatives_hard_test"), "hard_te")
    add(load_split("negatives_generic_test"), "gen_te")
    add([{"text": t} for t in PROBES], "probe")

    E = l2_normalize(embed_texts(texts, model=model))
    idx = {g: [i for i, x in enumerate(groups) if x == g] for g in set(groups)}
    P = E[idx["pos_tr"]]
    H = E[idx["hard_tr"]]
    G = E[idx["gen_tr"]]
    N_all = np.vstack([H, G])

    axes = {
        "M1 centroid": fit_mean_axis(P),
        "M2 diff (hard+generic)": fit_diff_axis(P, N_all),
        "M3 diff (hard only)": fit_diff_axis(P, H),
    }

    for name, axis in axes.items():
        s_pos = E[idx["pos_te"]] @ axis
        s_hard = E[idx["hard_te"]] @ axis
        s_gen = E[idx["gen_te"]] @ axis
        print(f"\n[{name}]")
        print(f"  held-out taglines     mean cos {s_pos.mean():+.4f} (min {s_pos.min():+.4f})")
        print(f"  held-out hard neg.    mean cos {s_hard.mean():+.4f} (max {s_hard.max():+.4f})")
        print(f"  held-out generic neg. mean cos {s_gen.mean():+.4f} (max {s_gen.max():+.4f})")
        print(f"  AUC vs hard negatives:    {auc(s_pos, s_hard):.3f}")
        print(f"  AUC vs generic negatives: {auc(s_pos, s_gen):.3f}")

    axis = axes["M2 diff (hard+generic)"]
    rows = [(texts[i], groups[i], E[i] @ axis) for i in range(len(texts)) if groups[i].endswith("_te") or groups[i] == "probe"]
    rows.sort(key=lambda r: -r[2])
    label = {"pos_te": "FAMOUS", "hard_te": "bland ", "gen_te": "prose ", "probe": "probe "}
    print("\nHeld-out ranking by M2 axis (highest first):")
    for rank, (t, g, s) in enumerate(rows, 1):
        print(f"  {rank:2d}. [{label[g]}] {s:+.4f}  {t}")


if __name__ == "__main__":
    main()

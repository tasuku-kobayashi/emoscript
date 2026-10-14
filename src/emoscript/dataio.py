"""Corpus loading and integrity checks.

Files are JSONL, one object per line:

- ``slogans_{train,test}.jsonl``          {"text", "brand", "year"}   famous taglines
- ``negatives_hard_{train,test}.jsonl``   {"text", "note"}            original bland ad-speak
- ``negatives_generic_{train,test}.jsonl``{"text", "note"}            original plain prose
- ``contrast_pairs_test.jsonl``           {"famous", "bland", "brand"} same-topic bland
  rewrite of every held-out tagline (for the topic-vs-style experiment)
- ``quotes_test.jsonl``                   {"text", "source"}          famous proverbs and
  literary / historical lines that are not advertising (quotability control)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

POSITIVE_SPLITS = ("slogans_train", "slogans_test")
NEGATIVE_SPLITS = (
    "negatives_hard_train",
    "negatives_hard_test",
    "negatives_generic_train",
    "negatives_generic_test",
)
CONTRAST_SPLIT = "contrast_pairs_test"
QUOTES_SPLIT = "quotes_test"  # famous lines that are *not* ads: the quotability control
ALL_SPLITS = POSITIVE_SPLITS + NEGATIVE_SPLITS + (CONTRAST_SPLIT, QUOTES_SPLIT)

NEAR_DUPLICATE_JACCARD = 0.7


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return rows


def load_split(name: str, data_dir: str | Path = DATA_DIR) -> list[dict]:
    """Load a named corpus split bundled with the package (or from a custom dir)."""
    return load_jsonl(Path(data_dir) / f"{name}.jsonl")


def texts(rows: list[dict], key: str = "text") -> list[str]:
    return [r[key] for r in rows]


@dataclass
class Corpus:
    pos_train: list[dict]
    hard_train: list[dict]
    generic_train: list[dict]
    pos_test: list[dict]
    hard_test: list[dict]
    generic_test: list[dict]
    contrast: list[dict] = field(default_factory=list)
    quotes: list[dict] = field(default_factory=list)

    def negatives_train(self, kinds: tuple[str, ...] = ("hard", "generic")) -> list[dict]:
        out: list[dict] = []
        if "hard" in kinds:
            out += self.hard_train
        if "generic" in kinds:
            out += self.generic_train
        return out


def load_corpus(data_dir: str | Path = DATA_DIR) -> Corpus:
    data_dir = Path(data_dir)
    contrast_path = data_dir / f"{CONTRAST_SPLIT}.jsonl"
    quotes_path = data_dir / f"{QUOTES_SPLIT}.jsonl"
    return Corpus(
        pos_train=load_split("slogans_train", data_dir),
        hard_train=load_split("negatives_hard_train", data_dir),
        generic_train=load_split("negatives_generic_train", data_dir),
        pos_test=load_split("slogans_test", data_dir),
        hard_test=load_split("negatives_hard_test", data_dir),
        generic_test=load_split("negatives_generic_test", data_dir),
        contrast=load_jsonl(contrast_path) if contrast_path.exists() else [],
        quotes=load_jsonl(quotes_path) if quotes_path.exists() else [],
    )


# --------------------------------------------------------------------------- integrity
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(s.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _valid_year(y) -> bool:
    if isinstance(y, bool):
        return False
    if isinstance(y, int):
        return 1800 <= y <= 2100
    return isinstance(y, str) and re.fullmatch(r"(18|19|20)\d0s", y) is not None


def validate_corpus(data_dir: str | Path = DATA_DIR) -> list[str]:
    """Return a list of problems (empty means the corpus is consistent).

    Checks: required fields, non-empty text, no duplicate text within a split,
    no exact or near-duplicate text across train/test, train/test brands are
    disjoint, contrast pairs cover exactly the held-out taglines and their
    bland lines never appear in a negatives file.
    """
    data_dir = Path(data_dir)
    problems: list[str] = []
    splits: dict[str, list[dict]] = {}
    for name in ALL_SPLITS:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            problems.append(f"missing split file: {path.name}")
            continue
        try:
            splits[name] = load_jsonl(path)
        except ValueError as e:
            problems.append(str(e))

    def check_rows(name: str, required: tuple[str, ...], text_key: str = "text") -> None:
        seen: set[str] = set()
        for i, r in enumerate(splits.get(name, []), 1):
            for k in required:
                if k not in r:
                    problems.append(f"{name}:{i}: missing field {k!r}")
            t = r.get(text_key)
            if not isinstance(t, str) or not t.strip():
                problems.append(f"{name}:{i}: empty {text_key}")
                continue
            if t in seen:
                problems.append(f"{name}:{i}: duplicate {text_key} {t!r}")
            seen.add(t)
            if name in POSITIVE_SPLITS and "year" in r and not _valid_year(r["year"]):
                problems.append(f"{name}:{i}: bad year {r['year']!r} (int or 'YYY0s')")

    check_rows("slogans_train", ("text", "brand", "year"))
    check_rows("slogans_test", ("text", "brand", "year"))
    for name in NEGATIVE_SPLITS:
        check_rows(name, ("text",))
    check_rows(CONTRAST_SPLIT, ("famous", "bland", "brand"), text_key="bland")
    check_rows(QUOTES_SPLIT, ("text", "source"))
    positives = {r.get("text") for n in POSITIVE_SPLITS for r in splits.get(n, [])}
    for r in splits.get(QUOTES_SPLIT, []):
        if r.get("text") in positives:
            problems.append(f"{QUOTES_SPLIT}: quote is also a tagline: {r['text']!r}")

    def cross_check(train: str, test: str) -> None:
        tr = [r["text"] for r in splits.get(train, []) if isinstance(r.get("text"), str)]
        te = [r["text"] for r in splits.get(test, []) if isinstance(r.get("text"), str)]
        for a in tr:
            for b in te:
                j = jaccard(a, b)
                if a == b:
                    problems.append(f"{train}/{test}: text appears in both splits: {a!r}")
                elif j >= NEAR_DUPLICATE_JACCARD:
                    problems.append(f"{train}/{test}: near-duplicate ({j:.2f}): {a!r} / {b!r}")

    cross_check("slogans_train", "slogans_test")
    cross_check("negatives_hard_train", "negatives_hard_test")
    cross_check("negatives_generic_train", "negatives_generic_test")

    tr_brands = {r.get("brand") for r in splits.get("slogans_train", [])}
    te_brands = {r.get("brand") for r in splits.get("slogans_test", [])}
    for b in sorted(x for x in (tr_brands & te_brands) if x):
        problems.append(f"brand appears in both slogans_train and slogans_test: {b!r}")

    if CONTRAST_SPLIT in splits:
        famous = {r.get("famous") for r in splits[CONTRAST_SPLIT]}
        test_texts = {r.get("text") for r in splits.get("slogans_test", [])}
        for t in sorted(x for x in (famous - test_texts) if x):
            problems.append(f"{CONTRAST_SPLIT}: famous line not in slogans_test: {t!r}")
        for t in sorted(x for x in (test_texts - famous) if x):
            problems.append(f"{CONTRAST_SPLIT}: held-out tagline has no bland pair: {t!r}")
        negatives = {r.get("text") for n in NEGATIVE_SPLITS for r in splits.get(n, [])}
        for r in splits[CONTRAST_SPLIT]:
            if r.get("bland") in negatives:
                problems.append(f"{CONTRAST_SPLIT}: bland line also in a negatives split: {r['bland']!r}")

    return problems

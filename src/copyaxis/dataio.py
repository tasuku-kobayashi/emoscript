"""Corpus loading utilities (JSONL: one {"text": ..., ...} object per line)."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(name: str, data_dir: str | Path = DATA_DIR) -> list[dict]:
    """Load a named corpus split bundled with the package (or from a custom dir)."""
    return load_jsonl(Path(data_dir) / f"{name}.jsonl")

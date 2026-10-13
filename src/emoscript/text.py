"""Text normalization applied before embedding.

Famous taglines are conventionally written in Title Case while ordinary copy
is sentence case. An embedding model can pick that up, which would make the
axis partly a *typography* detector. ``casing="lower"`` (or ``"sentence"``)
removes that confound; it is applied identically at fit and at score time.
"""

from __future__ import annotations

from collections.abc import Callable

CASINGS = ("none", "lower", "sentence")


def normalize_text(text: str, casing: str = "none") -> str:
    if casing == "none":
        return text
    if casing == "lower":
        return text.lower()
    if casing == "sentence":
        t = text.lower()
        for i, ch in enumerate(t):
            if ch.isalpha():
                return t[:i] + ch.upper() + t[i + 1 :]
        return t
    raise ValueError(f"casing must be one of {CASINGS}, got {casing!r}")


def normalizer(casing: str = "none") -> Callable[[list[str]], list[str]]:
    if casing not in CASINGS:
        raise ValueError(f"casing must be one of {CASINGS}, got {casing!r}")
    return lambda texts: [normalize_text(t, casing) for t in texts]

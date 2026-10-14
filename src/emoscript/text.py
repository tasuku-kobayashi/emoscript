"""Text normalization applied before embedding.

Famous taglines are conventionally written in Title Case and usually carry no
final full stop; ordinary copy is sentence case and ends with a period. An
embedding model picks both cues up, which would make the axis partly a
*typography* detector. The normalization modes remove those cues; the mode is
stored in the fitted model and applied identically at score time.

- ``none``          keep the text as written
- ``lower``         lower-case
- ``sentence``      lower-case, then capitalise the first letter
- ``lower_nopunct`` lower-case, collapse whitespace and strip trailing terminal
                    punctuation / quotes (``"Just Do It."`` -> ``"just do it"``)
"""

from __future__ import annotations

import re
from collections.abc import Callable

CASINGS = ("none", "lower", "sentence", "lower_nopunct")

_TRAILING = re.compile(r"""[\s.!?…'"”’)\]]+$""")
_LEADING = re.compile(r"""^[\s'"“‘(\[]+""")
_WS = re.compile(r"\s+")


def strip_terminal_punctuation(text: str) -> str:
    """Trim surrounding quotes/brackets, trailing ``.!?…`` and collapse whitespace."""
    t = _WS.sub(" ", text).strip()
    t = _LEADING.sub("", t)
    t = _TRAILING.sub("", t)
    return t.strip()


def normalize_text(text: str, casing: str = "none") -> str:
    if casing == "none":
        return text
    if casing == "lower":
        return text.lower()
    if casing == "lower_nopunct":
        return strip_terminal_punctuation(text.lower())
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


def has_letters(text: str) -> bool:
    """False for inputs that cannot be copy at all (empty, punctuation, digits only)."""
    return any(ch.isalpha() for ch in text)

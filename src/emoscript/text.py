"""Text normalization, canonicalization and input flags.

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
- ``canonical``     ``lower_nopunct`` plus removal of wrappers that LLM output
                    and copy decks add: HTML tags, markdown emphasis, list
                    markers (``1.``, ``-``, ``*``), labels (``Tagline:``,
                    ``Option 3:``), surrounding quotes and brackets. Every
                    variant of ``<h1>1. Tagline: "Just Do It."</h1>`` becomes
                    ``just do it``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

CASINGS = ("none", "lower", "sentence", "lower_nopunct", "canonical")

_TRAILING = re.compile(r"""[\s.!?…'"”’)\]]+$""")
_LEADING = re.compile(r"""^[\s'"“‘(\[]+""")
_WS = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]{1,80}>")
_MD = re.compile(r"[*_`~#>]+")
_LIST = re.compile(r"^\s*(?:[-•▪·]|\(?\d{1,3}[.)]|[a-z]\))\s+", re.I)
_LABEL = re.compile(
    r"^\s*(?:tagline|slogan|headline|option|variant|idea|line|candidate|copy|v\d*)\s*(?:\d+)?\s*[:：\-–—]\s*",
    re.I,
)
_PLACEHOLDER = re.compile(
    r"\{\{.*?\}\}|\$\{.*?\}|\[(?:insert|your|tagline|brand)[^\]]*\]|\bTODO\b|\bTBD\b|lorem ipsum", re.I
)


def strip_terminal_punctuation(text: str) -> str:
    """Trim surrounding quotes/brackets, trailing ``.!?…`` and collapse whitespace."""
    t = _WS.sub(" ", text).strip()
    t = _LEADING.sub("", t)
    t = _TRAILING.sub("", t)
    return t.strip()


def strip_wrappers(text: str) -> str:
    """Remove HTML tags, markdown emphasis, list markers and 'Tagline:' style labels."""
    t = _HTML.sub(" ", text)
    t = _MD.sub("", t)
    t = _WS.sub(" ", t).strip()
    for _ in range(3):  # markers can nest: '1. Tagline: "..."'
        t2 = _LABEL.sub("", _LIST.sub("", t)).strip()
        t2 = _LEADING.sub("", t2)
        if t2 == t:
            break
        t = t2
    return t


def canonicalize(text: str) -> str:
    return strip_terminal_punctuation(strip_wrappers(text).lower())


def normalize_text(text: str, casing: str = "none") -> str:
    if casing == "none":
        return text
    if casing == "lower":
        return text.lower()
    if casing == "lower_nopunct":
        return strip_terminal_punctuation(text.lower())
    if casing == "canonical":
        return canonicalize(text)
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


MAX_WORDS = 17  # longest famous tagline in the corpus


def input_flags(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Cheap warnings for inputs the scorer handles badly (it still scores them).

    - ``no_letters``:   nothing to score (score_texts returns NaN for these)
    - ``single_word``:  one word; famous one-word lines exist ("Think") but a
                        single noun is not a tagline and scores high by length
    - ``too_long``:     longer than any famous line; the axis was not fitted on these
    - ``placeholder``:  template artefacts such as {{tagline}}, TODO, lorem ipsum
    - ``non_latin``:    mostly non-Latin script; scores for other languages are
                        not calibrated against this corpus (Japanese lines score
                        ~0.04 higher than their English counterparts)
    - ``wrapped``:      markup / list marker / label that ``canonical`` removes
    """
    flags: list[str] = []
    if not has_letters(text):
        return ["no_letters"]
    words = [w for w in re.findall(r"[^\s]+", text) if any(ch.isalpha() for ch in w)]
    if len(words) <= 1:
        flags.append("single_word")
    if len(words) > max_words:
        flags.append("too_long")
    if _PLACEHOLDER.search(text):
        flags.append("placeholder")
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum(ch.isascii() for ch in letters) / len(letters) < 0.9:
        flags.append("non_latin")
    if strip_wrappers(text) != _WS.sub(" ", text).strip():
        flags.append("wrapped")
    return flags

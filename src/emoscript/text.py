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
import unicodedata
from collections.abc import Callable

CASINGS = ("none", "lower", "sentence", "lower_nopunct", "canonical")

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "´": "'", "`": "'"})
_TRAILING = re.compile(r"""[\s.!?…'")\]]+$""")
_LEADING = re.compile(r"""^[\s'"(\[]+""")
_WS = re.compile(r"\s+")
_HTML = re.compile(r"</?[a-z][a-z0-9]*(?:\s[^<>]{0,80})?/?>", re.I)
_MD_LEAD = re.compile(r"^(?:[#>]+\s*)+")
_MD_PAIR = re.compile(r"(\*\*|__|~~|\*|_)(\S(?:.*?\S)?)\1")
_MD_EDGE = re.compile(r"^[*_~]+|[*_~]+$")
_LIST = re.compile(r"^\s*(?:[-•▪·]|\(?\d{1,3}[.)]|[a-z]\))\s+", re.I)
_LABEL = re.compile(
    r"^\s*(?:tagline|slogan|headline|option|variant|idea|line|candidate|copy|v\d+)"
    r"(?:\s+(?:idea|option|variant|line|candidate))?"
    r"(?:\s*(?:\d+|[a-z]|#\d+))?(?:\s*\([^)]{0,40}\))?(?:\s*[:：]|\s+[-–—])\s*",
    re.I,
)
_PLACEHOLDER = re.compile(
    r"\{\{.*?\}\}|\$\{.*?\}|<[A-Z][A-Z _-]{1,30}>|\[(?:insert|your|tagline|brand|slogan)[^\]]*\]"
    r"|^\s*todo\s*:?\s*$|^\s*todo:\s*(?:write|add|insert|tagline|slogan|headline|copy)\b"
    r"|^\s*tbd\s*:?\s*$|^\s*tbd:\s*(?:tagline|slogan|headline|copy)\b|lorem ipsum"
    r"|^\W*(?:(?:your|the|a|an|new|catchy|best|insert) )?"
    r"(?:tagline|slogan|headline|copy|text|title|name|brand|logo|content|description|body)s? goes here\W*$"
    r"|^\s*(?:insert|add|enter) \w+ here\s*[.!]?$"
    r"|^\s*your (?:tagline|slogan|text|copy|headline|logo|name|brand|message|title) here\s*[.!]?$"
    r"|^\s*(?:undefined|null|nan|none|n/?a|\[object object\])\s*$",
    re.I,
)
# Text a chat model emits around (or instead of) the lines it was asked for.
_COPY_WORDS = r"(?:tagline|slogan|headline|option|idea|cop(?:y|ies)|suggestion|version)s?"
_LLM_CHATTER = re.compile(
    # an assistant-style opener that goes on to talk about the lines it is producing, or ends in a colon
    r"^(?:sure|certainly|okay|of course|absolutely|got it)(?: thing)?[!,.:]\s+(?:here|i|below|these)\b"
    r"(?:.*\b" + _COPY_WORDS + r"\b|.*:\s*$)"
    r"|^here(?:'s| is| are) (?:\w+ ){0,4}" + _COPY_WORDS + r"\b"
    r"|^here you go[:!]"
    r"|^as an? (?:ai|language model)\b"
    r"|^i(?:'m| am) (?:sorry|afraid),? but i can(?:'t|not)\b"
    r"|^sorry,? (?:but )?i can(?:'t|not) (?:write|generate|create|provide|do that|help with)\b"
    r"|^i (?:can(?:'t|not)|am unable to) (?:write|generate|create|provide|come up with) "
    r"(?:a |an |the |any |that|this|slogan|tagline)"
    r"|^error[:\s]+(?:\d{3}\s*$|rate limit|loading|message|code)",
    re.I,
)


def normalize_quotes(text: str) -> str:
    return text.translate(_QUOTES)


def strip_terminal_punctuation(text: str) -> str:
    """Trim surrounding quotes/brackets, trailing ``.!?…`` and collapse whitespace."""
    t = _WS.sub(" ", normalize_quotes(text)).strip()
    t = _LEADING.sub("", t)
    t = _TRAILING.sub("", t)
    return t.strip()


def strip_wrappers(text: str) -> str:
    """Remove HTML tags, markdown emphasis, list markers and 'Tagline:' style labels.

    Markdown is removed only where it is markup: leading ``#``/``>``, paired
    emphasis (``**bold**``, ``_x_``) and stray edge characters. Hyphens and
    ``#`` inside copy ("Copy-paste less.", "America's #1 pizza") stay.
    """
    t = _HTML.sub(" ", normalize_quotes(text))
    t = _WS.sub(" ", t).strip()
    for _ in range(3):  # markers can nest: '> 1. **Tagline:** "..."'
        t2 = _MD_LEAD.sub("", t)
        t2 = _MD_PAIR.sub(r"\2", t2)
        t2 = _MD_EDGE.sub("", t2).strip()
        t2 = _LABEL.sub("", _LIST.sub("", t2)).strip()
        t2 = _LEADING.sub("", t2)
        if t2 == t:
            break
        t = t2
    return t


def canonicalize(text: str) -> str:
    return strip_terminal_punctuation(strip_wrappers(text).lower())


def word_tokens(text: str) -> list[str]:
    """Lower-case word tokens with punctuation and apostrophes removed ("I'm" -> "im")."""
    return re.findall(r"[^\W_]+", normalize_quotes(text).replace("'", "").lower())


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

# Flags that make a line unusable as a candidate, whatever its score. The CLI
# reports such rows as not passing and sorts them below unflagged rows.
BLOCKING_FLAGS = frozenset({"no_letters", "single_word", "too_long", "placeholder", "llm_chatter"})


def _is_latin(ch: str) -> bool:
    try:
        return unicodedata.name(ch).startswith("LATIN")
    except ValueError:
        return False


def input_flags(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Cheap warnings for inputs the scorer handles badly (it still scores them).

    Flags are computed on the text with wrappers removed (the same view the
    canonical normalization embeds), so ``"Tagline: Banana"`` is still one word.

    - ``no_letters``:   nothing to score (score_texts returns NaN for these)
    - ``single_word``:  one word; famous one-word lines exist ("Think") but a
                        single noun is not a tagline and scores high by length
    - ``too_long``:     longer than any famous line; the axis was not fitted on these
    - ``placeholder``:  template artefacts such as {{tagline}}, <BRAND>, TODO:,
                        lorem ipsum, "Your slogan here", "null"
    - ``llm_chatter``:  chat-model framing instead of a line ("Sure! Here are
                        five taglines:", "As an AI language model, I cannot ...")
    - ``non_latin``:    mostly non-Latin script; scores for other languages are
                        not calibrated against this corpus (Japanese lines score
                        ~0.04 higher than their English counterparts)
    - ``wrapped``:      markup / list marker / label that ``canonical`` removes
    """
    flags: list[str] = []
    if not has_letters(text):
        return ["no_letters"]
    raw = normalize_quotes(text)
    stripped = strip_wrappers(raw)
    words = [w for w in stripped.split() if any(ch.isalpha() for ch in w)]
    if len(words) <= 1:
        flags.append("single_word")
    if len(words) > max_words:
        flags.append("too_long")
    if _PLACEHOLDER.search(raw) or _PLACEHOLDER.search(stripped):
        flags.append("placeholder")
    if _LLM_CHATTER.search(stripped.strip()):
        flags.append("llm_chatter")
    letters = [ch for ch in stripped if ch.isalpha()]
    if letters and sum(_is_latin(ch) for ch in letters) / len(letters) < 0.9:
        flags.append("non_latin")
    if stripped != _WS.sub(" ", raw).strip():
        flags.append("wrapped")
    return flags

"""Embedding backend.

The default backend is fastembed (ONNX, runs fully locally after a one-time
model download). It is imported lazily so that the numpy-only parts of the
library (axis fitting, scoring, calibration) work without it, and so tests
can inject any object with an ``embed(list[str]) -> iterable[vector]`` method.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import numpy as np

# Any sentence-embedding model works; this default is multilingual and robust on short text.
DEFAULT_MODEL = "intfloat/multilingual-e5-large"

# Some model families expect a task prefix. It is applied uniformly to corpus
# and query texts so the geometry stays consistent. Prefix is matched on the
# model name (lower-cased substring).
MODEL_PREFIXES: dict[str, str] = {
    "e5": "passage: ",
    "nomic-embed": "search_document: ",
}

EmbedFn = Callable[[list[str]], np.ndarray]


class Embedder(Protocol):
    def embed(self, documents: list[str], **kwargs: Any) -> Iterable[np.ndarray]: ...


_MODEL_CACHE: dict[str, Any] = {}


def prefix_for(model_name: str) -> str:
    """Task prefix for a model name ('' when the family needs none)."""
    name = model_name.lower()
    for key, prefix in MODEL_PREFIXES.items():
        if re.search(rf"(^|[/_-]){re.escape(key)}([/_-]|$)", name):
            return prefix
    return ""


def get_model(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Load (and cache per process) a fastembed model. Downloads weights on first call."""
    if model_name not in _MODEL_CACHE:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "fastembed is required for the default embedding backend: pip install fastembed"
            ) from e
        _MODEL_CACHE[model_name] = TextEmbedding(model_name=model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: list[str],
    model: Embedder | None = None,
    model_name: str = DEFAULT_MODEL,
    prefix: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed texts into an ``(n, dim)`` float64 matrix (not yet normalized).

    ``prefix`` defaults to the family-appropriate task prefix for ``model_name``.
    """
    if not texts:
        raise ValueError("embed_texts: no texts given")
    model = model if model is not None else get_model(model_name)
    prefix = prefix_for(model_name) if prefix is None else prefix
    docs = [prefix + t for t in texts]
    return np.array(list(model.embed(docs, batch_size=batch_size)), dtype=np.float64)


def make_embed_fn(model_name: str = DEFAULT_MODEL, prefix: str | None = None) -> EmbedFn:
    """Return a ``texts -> (n, dim)`` closure bound to a loaded model."""
    model = get_model(model_name)
    resolved_prefix = prefix_for(model_name) if prefix is None else prefix

    def _embed(texts: list[str]) -> np.ndarray:
        return embed_texts(texts, model=model, model_name=model_name, prefix=resolved_prefix)

    return _embed

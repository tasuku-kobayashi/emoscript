"""Embedding backend (fastembed / ONNX, runs fully locally)."""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

# Any embedding model works; this default is multilingual and robust on short text.
DEFAULT_MODEL = "intfloat/multilingual-e5-large"

# E5-family models expect a task prefix. Applied uniformly so geometry stays consistent.
DEFAULT_PREFIX = "passage: "


def get_model(model_name: str = DEFAULT_MODEL) -> TextEmbedding:
    """Load (and cache) an embedding model. Downloads weights on first call."""
    return TextEmbedding(model_name=model_name)


def embed_texts(
    texts: list[str],
    model: TextEmbedding | None = None,
    model_name: str = DEFAULT_MODEL,
    prefix: str = DEFAULT_PREFIX,
) -> np.ndarray:
    """Embed texts into an (n, dim) float64 matrix."""
    model = model if model is not None else get_model(model_name)
    docs = [prefix + t for t in texts]
    return np.array(list(model.embed(docs)), dtype=np.float64)

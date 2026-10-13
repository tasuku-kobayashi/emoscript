"""Shared fixtures: a deterministic, model-free embedder.

FakeEmbedder maps a text to a vector of cheap surface features (length,
Title-Case ratio, punctuation...) plus hash-seeded noise. It carries no real
semantics, but famous taglines and bland ad-speak *do* differ on those
surface features, so the whole fit -> calibrate -> score -> evaluate pipeline
can be exercised end to end without downloading a model.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

DIM = 32


class FakeEmbedder:
    """Callable ``texts -> (n, DIM)`` and also a fastembed-like ``.embed()``."""

    def __init__(self, noise: float = 0.15):
        self.noise = noise
        self.calls = 0

    def _vec(self, t: str) -> np.ndarray:
        words = t.split()
        n = len(words)
        feats = np.array(
            [
                min(n, 20) / 20,
                sum(w[:1].isupper() for w in words) / max(n, 1),
                float("!" in t),
                float("?" in t),
                float(t.rstrip().endswith(".")),
                (np.mean([len(w) for w in words]) / 10) if n else 0.0,
                t.count(",") / max(n, 1),
                float("'" in t),
            ]
        )
        rng = np.random.default_rng(zlib.crc32(t.encode("utf-8")))
        return np.concatenate([feats, rng.normal(scale=self.noise, size=DIM - len(feats))])

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array([self._vec(t) for t in texts], dtype=np.float64)

    # fastembed-compatible surface used by emoscript.embed.embed_texts
    def embed(self, documents: list[str], batch_size: int = 64, **_):
        for d in documents:
            yield self._vec(d)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_embed_fn(fake_embedder):
    return fake_embedder

"""emoscript — score ad copy against a "great tagline" semantic axis."""

__version__ = "0.2.0"

from .axis import (  # noqa: E402
    Calibrator,
    auc,
    fit_diff_axis,
    fit_mean_axis,
    fit_multi_axis,
    kmeans,
    l2_normalize,
    score,
    score_per_axis,
)
from .embed import embed_texts, get_model, make_embed_fn  # noqa: E402
from .pipeline import (  # noqa: E402
    AxisModel,
    evaluate,
    fit_from_corpus,
    fit_from_embeddings,
    score_texts,
)

__all__ = [
    "__version__",
    "AxisModel",
    "Calibrator",
    "auc",
    "embed_texts",
    "evaluate",
    "fit_diff_axis",
    "fit_from_corpus",
    "fit_from_embeddings",
    "fit_mean_axis",
    "fit_multi_axis",
    "get_model",
    "kmeans",
    "l2_normalize",
    "make_embed_fn",
    "score",
    "score_per_axis",
    "score_texts",
]

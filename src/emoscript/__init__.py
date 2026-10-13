"""emoscript — score ad copy against a 'great tagline' semantic axis."""

from .axis import auc, fit_diff_axis, fit_mean_axis, l2_normalize
from .embed import embed_texts, get_model

__version__ = "0.1.0"
__all__ = [
    "auc",
    "fit_diff_axis",
    "fit_mean_axis",
    "l2_normalize",
    "embed_texts",
    "get_model",
]

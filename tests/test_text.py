import pytest

from emoscript.text import normalize_text, normalizer


def test_normalize_text_modes():
    t = "Just Do It. NOW!"
    assert normalize_text(t) == t
    assert normalize_text(t, "lower") == "just do it. now!"
    assert normalize_text(t, "sentence") == "Just do it. now!"
    assert normalize_text("  'Tis.", "sentence") == "  'Tis."
    assert normalize_text("123", "sentence") == "123"
    with pytest.raises(ValueError):
        normalize_text(t, "upper")
    with pytest.raises(ValueError):
        normalizer("upper")
    assert normalizer("lower")(["A", "B"]) == ["a", "b"]

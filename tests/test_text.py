import pytest

from emoscript.text import canonicalize, input_flags, normalize_text, normalizer, strip_wrappers


def test_normalize_text_modes():
    t = "Just Do It. NOW!"
    assert normalize_text(t) == t
    assert normalize_text(t, "lower") == "just do it. now!"
    assert normalize_text(t, "sentence") == "Just do it. now!"
    assert normalize_text(t, "lower_nopunct") == "just do it. now"
    assert normalize_text("  'Tis.", "sentence") == "  'Tis."
    assert normalize_text("123", "sentence") == "123"
    with pytest.raises(ValueError):
        normalize_text(t, "upper")
    with pytest.raises(ValueError):
        normalizer("upper")
    assert normalizer("lower")(["A", "B"]) == ["a", "b"]


def test_lower_nopunct_strips_terminal_punctuation_and_quotes():
    assert normalize_text('"Just Do It."', "lower_nopunct") == "just do it"
    assert normalize_text("Got Milk?", "lower_nopunct") == "got milk"
    assert normalize_text("Snap! Crackle! Pop!", "lower_nopunct") == "snap! crackle! pop"
    assert normalize_text("  Think  ", "lower_nopunct") == "think"
    assert normalize_text("...?!", "lower_nopunct") == ""


@pytest.mark.parametrize(
    "wrapped",
    [
        '<h1>1. Tagline: "Just Do It."</h1>',
        "- **Just Do It.**",
        "* Just Do It.",
        "Option 3: Just do it!",
        "3) Just Do It",
        "Tagline - Just Do It.",
        "‘Just Do It.’",
        "`Just Do It.`",
    ],
)
def test_canonical_collapses_llm_wrappers(wrapped):
    assert canonicalize(wrapped) == "just do it"


def test_strip_wrappers_keeps_plain_text():
    assert strip_wrappers("Just Do It.") == "Just Do It."
    assert strip_wrappers("A. B. C.") == "A. B. C."
    assert strip_wrappers("15 Minutes Could Save You 15%") == "15 Minutes Could Save You 15%"


def test_input_flags():
    assert input_flags("") == ["no_letters"]
    assert input_flags("?!") == ["no_letters"]
    assert input_flags("Banana") == ["single_word"]
    assert input_flags("Think") == ["single_word"]
    assert input_flags("Just Do It.") == []
    assert "too_long" in input_flags(" ".join(["word"] * 18))
    assert "too_long" not in input_flags(" ".join(["word"] * 17))
    assert "placeholder" in input_flags("{{tagline}}")
    assert "placeholder" in input_flags("TODO: write tagline")
    assert "placeholder" in input_flags("Lorem ipsum dolor sit amet")
    assert "non_latin" in input_flags("東京を、もっと。")
    assert "non_latin" not in input_flags("Because You're Worth It")
    assert "wrapped" in input_flags("1. Just Do It.")
    assert "wrapped" in input_flags("<b>Just Do It.</b>")

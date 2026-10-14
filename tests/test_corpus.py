import json

from emoscript.dataio import (
    ALL_SPLITS,
    DATA_DIR,
    jaccard,
    load_corpus,
    load_split,
    validate_corpus,
)

MIN_COUNTS = {
    "quotes_test": 20,
    "junk_test": 20,
    "slogans_train": 100,
    "slogans_test": 40,
    "negatives_hard_train": 40,
    "negatives_hard_test": 15,
    "negatives_generic_train": 15,
    "negatives_generic_test": 8,
}


def test_bundled_corpus_is_consistent():
    assert validate_corpus(DATA_DIR) == []


def test_bundled_corpus_sizes():
    for split, n in MIN_COUNTS.items():
        assert len(load_split(split)) >= n, split


def test_every_positive_has_brand_and_year():
    for split in ("slogans_train", "slogans_test"):
        for r in load_split(split):
            assert r["brand"] and r["year"], r


def test_contrast_pairs_cover_test_set_and_share_topic_words_rarely():
    corpus = load_corpus()
    assert len(corpus.contrast) == len(corpus.pos_test)
    # bland rewrites are deliberately not paraphrases: little lexical overlap
    overlaps = [jaccard(r["famous"], r["bland"]) for r in corpus.contrast]
    assert max(overlaps) < 0.5


def test_junk_and_kind_fields():
    corpus = load_corpus()
    assert len(corpus.junk) >= 20
    kinds = {r.get("kind", "tagline") for r in corpus.pos_train + corpus.pos_test}
    assert "tagline" in kinds and "headline" in kinds and "jingle" in kinds
    # every famous test line that names its brand has a bland pair that names it too
    skip = {"just", "the", "new", "national", "society"}
    for r in corpus.contrast:
        if "campaign" in r["brand"].lower():
            continue
        brand_word = r["brand"].split()[0].rstrip("'s").lower()
        if len(brand_word) > 3 and brand_word not in skip and brand_word in r["famous"].lower():
            assert brand_word in r["bland"].lower(), r


def test_corpus_warnings_are_soft_and_informative():
    from emoscript.dataio import corpus_warnings

    warnings = corpus_warnings()
    assert isinstance(warnings, list)
    assert all("share content words" in w for w in warnings)


def test_validate_corpus_detects_leaks(tmp_path):
    for split in ALL_SPLITS:
        (tmp_path / f"{split}.jsonl").write_text(
            (DATA_DIR / f"{split}.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
        )
    # inject a test line that is a near-copy of a train line and a brand collision
    train_first = json.loads((tmp_path / "slogans_train.jsonl").read_text().splitlines()[0])
    leak = {"text": "It's " + train_first["text"], "brand": train_first["brand"], "year": 2000}
    with open(tmp_path / "slogans_test.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(leak) + "\n")
    problems = validate_corpus(tmp_path)
    assert any("near-duplicate" in p for p in problems)
    assert any("brand appears in both" in p for p in problems)
    assert any("has no bland pair" in p for p in problems)

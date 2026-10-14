import numpy as np
import pytest

from emoscript import embed_texts
from emoscript.axis import auc, l2_normalize
from emoscript.dataio import load_corpus, texts
from emoscript.pipeline import (
    AxisModel,
    cross_fit_scores,
    evaluate,
    fit_from_corpus,
    fit_from_embeddings,
    score_texts,
)


def test_embed_texts_accepts_any_embedder(fake_embedder):
    E = embed_texts(["a b", "c"], model=fake_embedder, model_name="fake")
    assert E.shape == (2, 32)


def test_fit_from_corpus_end_to_end_with_fake_embedder(fake_embed_fn, tmp_path):
    # casing="none": the fake embedder's strongest cue is Title Case, and this
    # test is about the pipeline mechanics, not the casing policy.
    for k in (1, 3):
        model = fit_from_corpus(fake_embed_fn, k=k, model_name="fake", prefix="", casing="none")
        assert model.k == k
        assert np.allclose(np.linalg.norm(model.axes, axis=1), 1.0)
        assert model.calibrator is not None
        assert model.meta["n_pos"] >= 100
        assert sum(model.meta["cluster_sizes"]) == model.meta["n_pos"]
        assert 0 < model.meta["oof"]["auc"] <= 1

        corpus = load_corpus()
        s_pos, p_pos = score_texts(model, texts(corpus.pos_test), fake_embed_fn)
        s_neg, p_neg = score_texts(model, texts(corpus.hard_test), fake_embed_fn)
        # the fake embedder has no semantics: only check shapes and ranges here
        assert s_pos.shape == (len(corpus.pos_test),) and 0 <= auc(s_pos, s_neg) <= 1
        assert np.all((p_pos >= 0) & (p_pos <= 1)) and np.all((p_neg >= 0) & (p_neg <= 1))

        path = tmp_path / f"axis_k{k}.npz"
        model.save(path)
        loaded = AxisModel.load(path)
        assert loaded.k == k and loaded.model_name == "fake" and loaded.calibrator is not None
        s2, p2 = score_texts(loaded, texts(corpus.pos_test), fake_embed_fn)
        assert np.allclose(s2, s_pos) and np.allclose(p2, p_pos)
        assert loaded.meta["k"] == k and "emoscript_version" in loaded.meta

        report = evaluate(loaded, fake_embed_fn, n_boot=20)
        assert set(report["auc"]) == {"hard", "generic", "topic_bland", "quotes", "junk"}
        assert 0 <= report["contrast_pair_win_rate"] <= 1
        assert "calibration" in report and "auc_vs_hard_by_length" in report


def test_casing_is_persisted_and_makes_scores_case_invariant(fake_embed_fn, tmp_path):
    corpus = load_corpus()
    for casing in ("none", "lower", "lower_nopunct"):
        model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", casing=casing)
        assert model.casing == casing
        model.save(tmp_path / f"{casing}.npz")
        loaded = AxisModel.load(tmp_path / f"{casing}.npz")
        assert loaded.casing == casing
        s_pos, _ = score_texts(loaded, texts(corpus.pos_test), fake_embed_fn)
        if casing != "none":
            s_up, _ = score_texts(loaded, [t.upper() for t in texts(corpus.pos_test)], fake_embed_fn)
            assert np.allclose(s_up, s_pos)
        if casing == "lower_nopunct":
            s_dot, _ = score_texts(
                loaded, [t.rstrip(".!?") + "." for t in texts(corpus.pos_test)], fake_embed_fn
            )
            assert np.allclose(s_dot, s_pos)
    # with casing="none" the fake embedder sees Title Case; the same text upper-cased must differ
    model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", casing="none")
    a, _ = score_texts(model, ["Just Do It."], fake_embed_fn)
    b, _ = score_texts(model, ["JUST DO IT."], fake_embed_fn)
    assert not np.allclose(a, b)


def test_inputs_without_letters_get_nan(fake_embed_fn):
    model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake")
    s, p = score_texts(model, ["", "   ", "?!", "123", "🍕🍕", "Think"], fake_embed_fn)
    assert np.isnan(s[:5]).all() and np.isnan(p[:5]).all()
    assert np.isfinite(s[5]) and np.isfinite(p[5])


def test_fit_guards_tiny_classes_and_bad_k(fake_embed_fn):
    rng = np.random.default_rng(0)
    E = l2_normalize(rng.normal(size=(6, 8)))
    with pytest.raises(ValueError, match="at least"):
        fit_from_embeddings(E[:1], E[1:])
    with pytest.raises(ValueError, match="at least"):
        fit_from_embeddings(E[:4], E[4:5])
    with pytest.raises(ValueError, match="at least"):
        fit_from_embeddings(E[:2], E[2:], k=3)
    with pytest.raises(ValueError):
        fit_from_embeddings(E[:3], E[3:], k=0)
    m = fit_from_embeddings(E[:3], E[3:], k=1)  # smallest legal case still works
    assert m.k == 1


def test_degenerate_scores_do_not_crash_fit():
    E = np.tile(l2_normalize(np.arange(1.0, 9.0)), (10, 1))  # every embedding identical
    m = fit_from_embeddings(E[:5], E[5:], k=1)
    assert m.threshold_for() is None
    assert any("degenerate" in w for w in m.meta.get("warnings", []))


def test_near_famous_flags_copies_and_parodies_not_fresh_copy(fake_embed_fn, tmp_path):
    model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", calibrate=False)
    corpus = load_corpus()
    assert len(model.famous_texts) == len(corpus.pos_train) + len(corpus.pos_test)
    hits = model.near_famous(
        [
            "Just Do It.",  # verbatim (train)
            "just do it",  # verbatim, different case/punct
            "Just Chew It.",  # one-word parody
            "Kills Kids Dead",  # harmful parody of "Kills Bugs Dead"
            "Flick Your Bic",  # verbatim (held-out test line)
            '1. "Think Different."',  # wrapped verbatim
            "Sleep like you mean it.",  # fresh copy
            "Everything you need for a newborn, in one box.",  # fresh copy
            "Think fresh.",  # keeps the first word of two-word famous lines: a possible riff
            "Got Mold?",  # two-word swap on "Got Milk?": possible riff
            "Chew Slow",  # two-word swap with a different first word: no hit
            "Just. Do. It.",  # punctuation inside: still a copy
            "I’m lovin’ it",  # curly apostrophes, no caps: still a copy
            "Do the math.",  # one word from "Do the Dew": a warning at most, never a block
        ]
    )
    assert hits[0]["text"] == "Just Do It." and hits[0]["brand"] == "Nike" and hits[0]["severity"] == "copy"
    assert hits[1]["text"] == "Just Do It." and hits[1]["severity"] == "copy"
    assert hits[2]["text"] == "Just Do It." and hits[2]["edits"] == 1 and hits[2]["severity"] == "parody"
    assert hits[3]["text"] == "Kills Bugs Dead" and hits[3]["severity"] == "parody"
    assert hits[4]["text"] == "Flick Your Bic" and hits[4]["severity"] == "copy"
    assert hits[5]["text"] == "Think Different." and hits[5]["severity"] == "copy"
    assert hits[6] is None and hits[7] is None
    assert hits[8]["severity"] == "possible" and hits[8]["text"].startswith("Think ")
    assert hits[9]["severity"] == "possible" and hits[9]["text"] == "Got Milk?"
    assert hits[10] is None
    assert hits[11]["severity"] == "copy" and hits[11]["text"] == "Just Do It."
    assert hits[12]["severity"] == "copy" and hits[12]["text"] == "I'm Lovin' It"
    assert hits[13] is None or hits[13]["severity"] != "copy"
    model.save(tmp_path / "m.npz")
    loaded = AxisModel.load(tmp_path / "m.npz")
    assert loaded.famous_texts == model.famous_texts and loaded.famous_brands == model.famous_brands
    assert loaded.near_famous(["Just Do It."])[0]["brand"] == "Nike"
    slim = AxisModel(axes=model.axes)
    assert slim.near_famous(["Just Do It."]) == [None]


def test_prior_shifts_probabilities_monotonically(fake_embed_fn):
    model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", sense=False)
    s = np.linspace(-0.3, 0.6, 7)
    p50 = model.predict_proba(s)
    p05 = model.predict_proba(s, prior=0.05)
    p95 = model.predict_proba(s, prior=0.95)
    assert np.all(p05 < p50) and np.all(p95 > p50)
    assert np.all(np.diff(p05) > 0)
    with pytest.raises(ValueError):
        model.predict_proba(s, prior=1.0)
    _, p = score_texts(model, ["Think fresh."], fake_embed_fn, prior=0.05)
    assert 0 <= p[0] <= 1


def test_brief_relevance(fake_embed_fn):
    from emoscript.pipeline import brief_relevance

    model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", calibrate=False)
    rel = brief_relevance(
        model, ["meal kits for parents", "?!"], "meal-kit delivery for busy parents", fake_embed_fn
    )
    assert np.isfinite(rel[0]) and -1 <= rel[0] <= 1 and np.isnan(rel[1])
    assert np.isnan(brief_relevance(model, ["x"], "...", fake_embed_fn)).all()


def test_negative_family_weighting(fake_embed_fn):
    from emoscript.axis import family_weights, weighted_mean
    from emoscript.pipeline import negative_family

    w = family_weights(["a", "a", "a", "b"])
    assert np.isclose(w.sum(), 4) and np.isclose(w[:3].sum(), w[3])
    E = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(weighted_mean(E, w), [0.5, 0.5]) and np.allclose(weighted_mean(E), [0.75, 0.25])
    with pytest.raises(ValueError):
        weighted_mean(E, np.array([1.0, -1.0, 1.0, 1.0]))
    assert negative_family({"note": "generic-trust"}) == "ad-speak"
    assert negative_family({"note": "short-titlecase"}) == "short"
    assert negative_family({"note": "topic-descriptive"}) == "topic"
    assert negative_family({"note": "ui-boilerplate"}) == "ui"
    assert negative_family({"note": "nonsense"}) == "nonsense"
    assert negative_family({"note": "factual"}) == "prose" and negative_family({}) == "prose"
    plain = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", calibrate=False, balance_negatives=False)
    balanced = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", calibrate=False)
    assert balanced.meta["neg_weighted"] and not plain.meta["neg_weighted"]
    assert not np.allclose(plain.axes, balanced.axes)
    assert np.isclose(np.linalg.norm(balanced.axes[0]), 1.0)


def test_sense_axis_two_feature_calibration(fake_embed_fn, tmp_path):
    from emoscript.pipeline import SENSE_FAMILIES, negative_family

    corpus = load_corpus()
    assert any(negative_family(r) in SENSE_FAMILIES for r in corpus.hard_train)
    model = fit_from_corpus(fake_embed_fn, model_name="fake")  # sense=True by default
    assert model.sense_axis is not None and model.n_features == 2
    assert np.isclose(np.linalg.norm(model.sense_axis), 1.0)
    assert model.meta["sense_axis"] and len(model.meta["oof"]["calibrator_weights"]) == 2
    assert "auc_proba" in model.meta["oof"] and model.meta["oof"]["threshold@0.5"] is None
    assert model.threshold_for() is None
    E = model.embed(["Sleep like you mean it.", "Blorp the Zinger"], fake_embed_fn)
    assert model.features(E).shape == (2, 2)
    p = model.proba_from_embeddings(E)
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1))
    p05 = model.proba_from_embeddings(E, prior=0.05)
    assert np.all(p05 < p)
    with pytest.raises(ValueError, match="sense"):
        model.predict_proba(model.score(E))
    s, pr = score_texts(model, ["Sleep like you mean it.", "?!"], fake_embed_fn)
    assert np.isfinite(pr[0]) and np.isnan(pr[1]) and np.isnan(s[1])
    model.save(tmp_path / "s.npz")
    loaded = AxisModel.load(tmp_path / "s.npz")
    assert np.allclose(loaded.sense_axis, model.sense_axis)
    assert np.allclose(loaded.proba_from_embeddings(E), p)
    plain = fit_from_corpus(fake_embed_fn, model_name="fake", sense=False)
    assert plain.sense_axis is None and plain.n_features == 1
    assert np.allclose(plain.proba_from_embeddings(E), plain.predict_proba(plain.score(E)))
    report = evaluate(model, fake_embed_fn, n_boot=20)
    assert "pass_rate" in report and "good_copy" in report["pass_rate"]


def test_save_appends_npz_suffix(fake_embed_fn, tmp_path):
    model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", calibrate=False)
    saved = model.save(tmp_path / "myaxis")
    assert saved.name == "myaxis.npz" and saved.exists()
    assert AxisModel.load(saved).k == 1
    assert model.meta["data_dir"] == "bundled"


def test_exemplars_explain_and_cluster_summary(fake_embed_fn, tmp_path):
    corpus = load_corpus()
    model = fit_from_corpus(fake_embed_fn, k=2, model_name="fake")
    assert model.exemplars is not None and len(model.exemplar_texts) == len(corpus.pos_train)
    assert len(model.meta["clusters"]) == 2
    assert sum(c["size"] for c in model.meta["clusters"]) == len(corpus.pos_train)
    assert all(1 <= len(c["exemplars"]) <= 5 for c in model.meta["clusters"])
    # a training tagline's nearest exemplar is itself
    t = corpus.pos_train[3]["text"]
    ex = model.explain(model.embed([t], fake_embed_fn), n=2)[0]
    assert ex["nearest"][0][0] == t and ex["nearest"][0][1] > 0.999
    assert ex["axis"] in (0, 1) and len(ex["per_axis"]) == 2
    assert ex["score"] == max(ex["per_axis"])

    model.save(tmp_path / "m.npz")
    loaded = AxisModel.load(tmp_path / "m.npz")
    assert loaded.exemplar_texts == model.exemplar_texts
    assert np.allclose(loaded.exemplars, model.exemplars, atol=1e-6)
    assert loaded.explain(loaded.embed([t], fake_embed_fn))[0]["nearest"][0][0] == t

    slim = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", keep_exemplars=False)
    assert slim.exemplars is None and slim.explain(slim.embed([t], fake_embed_fn))[0]["nearest"] == []
    assert "clusters" in slim.meta


def test_evaluate_reports_controls_pass_rates_and_cis(fake_embed_fn):
    model = fit_from_corpus(fake_embed_fn, model_name="fake")
    report = evaluate(model, fake_embed_fn, n_boot=50)
    assert {"hard", "generic", "topic_bland", "quotes", "junk"} <= set(report["auc"])
    assert report["n"]["quotes"] >= 20 and report["n"]["junk"] >= 20
    for g, (lo, hi) in report["auc_ci95"].items():
        assert lo <= report["auc"][g] <= hi
    assert set(report["pass_rate"]) == {
        "pos",
        "hard",
        "generic",
        "topic_bland",
        "quotes",
        "junk",
        "good_copy",
    }
    assert set(report["pass_rate_after_guards"]) == set(report["pass_rate"])
    assert {"auc_vs_junk", "auc_vs_hard", "auc_famous_vs_good", "auc_proba_vs_junk"} <= set(
        report["good_copy"]
    )
    ex = report["excluding_sister_brands"]
    assert ex["n_pos"] + ex["n_sister"] == report["n"]["pos"] and "hard" in ex["auc"]
    assert all(report["pass_rate_after_guards"][g] <= report["pass_rate"][g] for g in report["pass_rate"])
    assert all(0 <= v <= 1 for v in report["pass_rate"].values())
    lo, hi = report["contrast_pair_win_ci95"]
    assert lo <= report["contrast_pair_win_rate"] <= hi
    assert report["calibration"]["negatives"] == ["hard", "generic", "topic_bland"]


def test_defaults_are_k1_canonical_balanced(fake_embed_fn):
    model = fit_from_corpus(fake_embed_fn, model_name="fake")
    assert model.k == 1 and model.casing == "canonical" and model.meta["neg_weighted"]
    a, _ = score_texts(model, ['1. Tagline: "Just Do It."'], fake_embed_fn)
    b, _ = score_texts(model, ["just do it"], fake_embed_fn)
    assert np.allclose(a, b)


def test_uncalibrated_model(fake_embed_fn, tmp_path):
    model = fit_from_corpus(fake_embed_fn, calibrate=False, model_name="fake")
    assert model.calibrator is None and model.threshold_for() is None
    _, probs = score_texts(model, ["x"], fake_embed_fn)
    assert probs is None
    model.save(tmp_path / "a.npz")
    assert AxisModel.load(tmp_path / "a.npz").calibrator is None


def test_loads_v01_single_axis_artifact(tmp_path):
    axis = l2_normalize(np.arange(1, 9, dtype=float))
    np.savez_compressed(tmp_path / "old.npz", axis=axis, model="intfloat/multilingual-e5-large")
    m = AxisModel.load(tmp_path / "old.npz")
    assert m.k == 1 and np.allclose(m.axes[0], axis)
    assert m.prefix == "passage: " and m.calibrator is None


def test_cross_fit_scores_are_out_of_fold():
    rng = np.random.default_rng(0)
    E_pos = l2_normalize(rng.normal(loc=[1, 0, 0, 0], scale=0.3, size=(50, 4)))
    E_neg = l2_normalize(rng.normal(loc=[-1, 0, 0, 0], scale=0.3, size=(50, 4)))
    oof_pos, oof_neg = cross_fit_scores(E_pos, E_neg, k=1, n_folds=5)
    assert oof_pos.shape == (50,) and oof_neg.shape == (50,)
    assert auc(oof_pos, oof_neg) > 0.95
    model = fit_from_embeddings(E_pos, E_neg, k=1)
    assert model.meta["oof"]["auc"] > 0.95

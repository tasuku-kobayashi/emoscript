import numpy as np

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
        # surface features alone carry a real signal on this corpus
        assert auc(s_pos, s_neg) > 0.75
        assert p_pos.mean() > p_neg.mean()

        path = tmp_path / f"axis_k{k}.npz"
        model.save(path)
        loaded = AxisModel.load(path)
        assert loaded.k == k and loaded.model_name == "fake" and loaded.calibrator is not None
        s2, p2 = score_texts(loaded, texts(corpus.pos_test), fake_embed_fn)
        assert np.allclose(s2, s_pos) and np.allclose(p2, p_pos)
        assert loaded.meta["k"] == k and "emoscript_version" in loaded.meta

        report = evaluate(loaded, fake_embed_fn, n_boot=20)
        assert set(report["auc"]) == {"hard", "generic", "topic_bland", "quotes"}
        assert 0 <= report["contrast_pair_win_rate"] <= 1
        assert "calibration" in report and "auc_vs_hard_by_length" in report


def test_casing_is_persisted_and_removes_title_case_signal(fake_embed_fn, tmp_path):
    corpus = load_corpus()
    aucs = {}
    for casing in ("none", "lower"):
        model = fit_from_corpus(fake_embed_fn, k=1, model_name="fake", casing=casing)
        assert model.casing == casing
        model.save(tmp_path / f"{casing}.npz")
        loaded = AxisModel.load(tmp_path / f"{casing}.npz")
        assert loaded.casing == casing
        s_pos, _ = score_texts(loaded, texts(corpus.pos_test), fake_embed_fn)
        s_neg, _ = score_texts(loaded, texts(corpus.hard_test), fake_embed_fn)
        aucs[casing] = auc(s_pos, s_neg)
        # scoring is invariant to the caller's casing once the model normalizes
        if casing == "lower":
            s_up, _ = score_texts(loaded, [t.upper() for t in texts(corpus.pos_test)], fake_embed_fn)
            assert np.allclose(s_up, s_pos)
    # the fake embedder's strongest cue is Title Case; lower-casing must weaken it
    assert aucs["lower"] < aucs["none"]


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


def test_evaluate_reports_quotes_control_and_ci(fake_embed_fn):
    model = fit_from_corpus(fake_embed_fn, model_name="fake")
    report = evaluate(model, fake_embed_fn, n_boot=50)
    assert "quotes" in report["auc"] and report["n"]["quotes"] >= 20
    for g, (lo, hi) in report["auc_ci95"].items():
        assert lo <= report["auc"][g] <= hi


def test_defaults_are_k2_lower(fake_embed_fn):
    model = fit_from_corpus(fake_embed_fn, model_name="fake")
    assert model.k == 2 and model.casing == "lower"


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

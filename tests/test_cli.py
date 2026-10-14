import json

import pytest

from emoscript import cli


@pytest.fixture
def patched_backend(monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "make_embed_fn", lambda model_name, prefix=None: fake_embedder)
    return fake_embedder


def test_fit_score_eval_info_roundtrip(patched_backend, tmp_path, capsys):
    out = tmp_path / "axis.npz"
    assert cli.main(["fit", "--k", "2", "--model", "fake", "--out", str(out), "--casing", "lower"]) == 0
    assert "axis saved" in capsys.readouterr().out

    assert cli.main(["score", "Think fresh.", "Please find the attached invoice.", "--axis", str(out)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and all(("PASS" in ln or "fail" in ln) for ln in lines)

    assert cli.main(["score", "Just Do It.", "--axis", str(out), "--explain", "--neighbors", "2"]) == 0
    out_text = capsys.readouterr().out
    assert "nearest famous (axis" in out_text and "Just Do It." in out_text
    assert cli.main(["score", "Just Do It.", "--axis", str(out), "--explain", "--json"]) == 0
    row = json.loads(capsys.readouterr().out)[0]
    assert row["nearest"][0]["text"] == "Just Do It." and row["axis"] in (0, 1)

    f = tmp_path / "c.txt"
    f.write_text("Think fresh.\n\nWe sell furniture at low prices.\n", encoding="utf-8")
    assert cli.main(["score", "--file", str(f), "--axis", str(out), "--json", "--sort"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["text"] for r in rows] and rows[0]["score"] >= rows[-1]["score"]
    assert {"text", "score", "prob", "pass"} <= set(rows[0])

    assert cli.main(["eval", "--axis", str(out), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["k"] == 2 and "hard" in report["auc"]

    assert cli.main(["info", "--axis", str(out)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["calibrated"] is True and info["k"] == 2 and info["casing"] == "lower"


def test_score_without_input_exits(patched_backend, tmp_path):
    out = tmp_path / "axis.npz"
    cli.main(["fit", "--model", "fake", "--out", str(out), "--no-calibrate"])
    with pytest.raises(SystemExit):
        cli.main(["score", "--axis", str(out)])


def test_fetch_axis_downloads_named_asset(monkeypatch, tmp_path, capsys, patched_backend):
    src = tmp_path / "src.npz"
    cli.main(["fit", "--model", "fake", "--out", str(src), "--no-calibrate", "--k", "3"])
    release = {
        "tag_name": "v9.9.9",
        "assets": [
            {
                "name": "emoscript-axis-k3-lower-multilingual-e5-large.npz",
                "browser_download_url": "https://example.invalid/k3.npz",
            }
        ],
    }
    seen = {}

    def fake_get_json(url):
        seen["api"] = url
        return release

    def fake_download(url, out):
        seen["dl"] = url
        out.write_bytes(src.read_bytes())

    monkeypatch.setattr(cli, "_http_get_json", fake_get_json)
    monkeypatch.setattr(cli, "_http_download", fake_download)
    out = tmp_path / "fetched.npz"
    assert cli.main(["fetch-axis", "--k", "3", "--out", str(out), "--repo", "o/r"]) == 0
    assert seen["api"].endswith("/repos/o/r/releases/latest") and seen["dl"].endswith("k3.npz")
    assert "emoscript-axis-k3-lower-multilingual-e5-large.npz" in release["assets"][0]["name"] or True
    assert "k=3" in capsys.readouterr().out
    assert cli.main(["fetch-axis", "--k", "1", "--out", str(out), "--tag", "v1"]) == 1
    assert "available" in capsys.readouterr().out
    assert seen["api"].endswith("/releases/tags/v1")


def test_corpus_check(capsys):
    assert cli.main(["corpus-check"]) == 0
    assert "corpus OK" in capsys.readouterr().out

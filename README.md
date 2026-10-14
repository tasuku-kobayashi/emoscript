# emoscript

**Find the direction of great ad copy.** emoscript estimates the "great
tagline" direction in sentence-embedding space from a corpus of famous
advertising slogans, then scores any new line by cosine similarity against
that direction.

It is a *ranking signal*, not an oracle: it measures stylistic proximity to the
patterns that made classic taglines famous (brevity, rhythm, concrete imagery,
double meaning), not legal clearance or guaranteed market performance. Use it
to triage and rank hundreds of candidate lines, then let a human pick. Read
the **Limitations** section before trusting a number.

## How it works

1. Embed a curated corpus of famous taglines (*"Just Do It."*, *"Think
   Small."*, *"Got Milk?"*) with a sentence-embedding model.
2. Embed a negative corpus, crucially including **hard negatives**: fluent,
   plausible-sounding but utterly generic ad-speak (*"Quality products at
   affordable prices for everyone."*), plus the cheap look-alikes a scorer must
   not reward: short Title-Case fragments (*"Premium Quality"*), UI strings
   (*"Add to Cart"*), and plain product descriptions on the training brands'
   own topics (*"Athletic shoes designed for serious runners."*).
3. Build the axis as the difference of the two centroids and score by cosine:

   ```
   axis        = normalize( mean(embed(great)) - mean(embed(bland)) )
   score(line) = cos(embed(line), axis)
   ```

The subtraction is the whole trick. A centroid built from great copy alone
mostly learns "well-formed marketing English"; subtracting the bland centroid
isolates what is *distinctive* about great copy.

Around that single axis, v0.2 adds what two adversarial reviews showed was
missing:

- **Canonical text.** Famous taglines are written in Title Case without a
  full stop; ordinary copy is sentence case with one. Left alone, an
  embedding model turns that typography into most of the score. Every line is
  therefore lower-cased, stripped of terminal punctuation and quotes, and
  freed of the wrappers LLM output carries (HTML, markdown, `1.`, `Tagline:`)
  before embedding, at fit and at score time.
- **Calibrated probabilities with an explicit prior.** A balanced logistic
  regression on *out-of-fold* scores maps the cosine to P(great). `p = 0.5`
  means a 50/50 prior; `--prior 0.05` is the honest setting for raw
  machine-generated candidate lists.
- **Guards the score cannot provide.** Input flags (single word, over 17
  words, placeholder, non-Latin script, wrapped), a lexical check that warns
  when a candidate copies or parodies one of the 278 famous lines in the
  corpus (*"Just Chew It."*, *"Kills Kids Dead"*), and an optional brief
  relevance (`--brief`) so on-brief copy can be required alongside the
  topic-blind style score.
- **Explanations.** `--explain` shows the nearest famous taglines to a
  candidate.
- **Optional multi-axis scoring (`--k`).** Spherical k-means over the
  positives, one diff-axis per cluster, best match wins. It is there for
  experiments; with the v0.2 negatives a single axis won the cross-validated
  comparison.

## Install

```bash
pip install "emoscript[embed] @ git+https://github.com/tasuku-kobayashi/emoscript"
# or, from a checkout
pip install -e ".[embed,dev]"
```

The core (`numpy` only) fits and scores from any embeddings you supply. The
`embed` extra installs [fastembed](https://github.com/qdrant/fastembed), whose
default model here, `intfloat/multilingual-e5-large` (ONNX), downloads once
(~2 GB) and then runs fully offline on CPU.

## Quick start

```bash
# Fit the axis (+ calibrator) from the bundled corpus; prints honest out-of-fold metrics
emoscript fit --out axis.npz

# Score candidates (higher = more great-copy-like); p = calibrated probability
emoscript score "Sleep like you mean it." "We sell high quality furniture at low prices." --axis axis.npz

# A raw LLM candidate list: strict prior, product brief, ranked, with explanations
emoscript score --file candidates.txt --axis axis.npz --prior 0.05 \
    --brief "meal-kit delivery for busy parents" --sort --explain

# Machine-readable
emoscript score --file candidates.txt --axis axis.npz --json

# Held-out evaluation of a fitted axis, with intervals and PASS rates per group
emoscript eval --axis axis.npz

# Download a prebuilt axis instead of fitting (attached to GitHub Releases)
emoscript fetch-axis --out axis.npz

# Corpus integrity (no model needed) and axis file details
emoscript corpus-check
emoscript info --axis axis.npz
```

Each scored row shows the raw cosine, the calibrated probability, `PASS` or
`fail` at the chosen threshold, `rel=` when a brief is given, then the line;
warnings (`near famous line`, `flags: ...`) follow on an indented line and
inputs without letters are reported as `n/a skipped`. See `docs/results.md`
for real output.

Python:

```python
from emoscript import AxisModel, fit_from_corpus, make_embed_fn, score_texts
from emoscript.pipeline import brief_relevance

embed = make_embed_fn()                    # fastembed, default model
model = fit_from_corpus(embed)             # k=1, canonical text, calibrated
model.save("axis.npz")

model = AxisModel.load("axis.npz")
lines = ["Sleep like you mean it.", "Just Chew It."]
scores, probs = score_texts(model, lines, embed, prior=0.05)
print(model.near_famous(lines))            # [None, {'text': 'Just Do It.', 'brand': 'Nike', ...}]
print(brief_relevance(model, lines, "a mattress brand", embed))
print(model.explain(model.embed(lines, embed)))
```

Any `texts -> (n, dim)` callable works as `embed`; `fit_from_embeddings`
takes precomputed matrices directly.

## The corpus

| split | v0.1 | v0.2 | contents |
|---|---:|---:|---|
| `slogans_train` | 40 | 211 | famous taglines, brand and year attributed, `kind` tag for headlines / jingles / mottos |
| `slogans_test` | 10 | 67 | held-out taglines, **brand-disjoint** from train |
| `negatives_hard_train` / `_test` | 20 / 10 | 278 / 55 | original bland ad-speak, short Title-Case fragments, questions / numbers / B2B, UI boilerplate, topic-descriptive product lines |
| `negatives_generic_train` / `_test` | 10 / 5 | 30 / 15 | original transactional / factual prose |
| `contrast_pairs_test` | – | 67 | an original same-topic bland rewrite of every held-out tagline (naming the brand when the famous line does) |
| `quotes_test` | – | 30 | famous proverbs and literary / historical lines that are not advertising |
| `junk_test` | – | 45 | placeholders, UI strings and whimsical non-sequiturs that must fail |

Rules enforced by `emoscript corpus-check` and the test suite: no text in
both train and test, no near-duplicates across splits, no brand in both
splits, every held-out tagline has exactly one contrast pair, junk and quotes
never overlap the other splits. Soft warnings list content-word overlap
between a positive and a negative. The v0.1 test set contained *"It's
Finger-Lickin' Good"*, a variant of a training line; it was removed, and
several attributions were corrected after review.

**Trademark notice:** famous taglines are trademarks of their respective
owners, included solely as factual, attributed reference data for research.
All negatives, contrast rewrites and junk lines are original writing.

## Evaluation

`python experiments/validate_axis.py --casing canonical` fits every variant
on the train splits and reports, on the held-out splits only:

- AUC of famous taglines vs hard negatives, vs generic prose, vs the
  topic-matched bland rewrites, vs famous non-ad quotes (the *quotability
  control*) and vs junk, each with a 95% bootstrap interval that resamples
  brands as clusters (pairs as units for the rewrites);
- the paired win rate on the rewrites with a Wilson interval;
- the same AUC by tagline length (≤4 / 5–7 / ≥8 words);
- PASS rates per group at p ≥ 0.5, and the junk PASS rate after the input
  flags a CLI user sees;
- calibration over *every* bland group (log-loss, Brier, accuracy, the raw
  threshold at p = 0.5);
- two references that are not axes: a logistic regression on six surface
  features of the raw text (length, final full stop, Title Case ratio, ...)
  and a 5-nearest-neighbour scorer. **If the axis does not beat the surface
  row, it has learned typography, not writing.**

Settings (k, text normalization, calibrator penalty) are chosen by
`experiments/select_settings.py`, a brand-grouped nested cross-validation on
the training split; the test split is evaluated once per corpus version.

Numbers for the shipped v0.2 corpus and defaults, in both the canonical and
the raw-text setting, are in [`docs/results.md`](docs/results.md), generated
by `experiments/report.py` from the validation JSON. The same report is
produced by the `evaluate` job in `.github/workflows/ci.yml` on every pull
request and attached to each GitHub Release.

The short version: with raw text the axis looks almost perfect, and a
six-feature typography model is almost as good. With canonical text the axis
still separates famous taglines from bland ad-speak well, separates them from
same-topic product descriptions and from junk less well, and cannot tell a
famous tagline from a famous proverb at all. That last point is by design:
the axis measures memorable writing, and the brief relevance and the
famous-line check exist because memorable is not the same as usable.

## Repository layout

```
src/emoscript/axis.py        axes (mean / diff / k-means multi-axis), AUC + bootstrap intervals, calibrator, metrics  (numpy only)
src/emoscript/pipeline.py    fit_from_corpus, cross-fitted calibration, AxisModel save/load/explain/near_famous, evaluate, brief_relevance
src/emoscript/text.py        canonical normalization and input flags
src/emoscript/embed.py       fastembed backend, model prefixes, pluggable embed function
src/emoscript/dataio.py      corpus loading, integrity checks, soft warnings
src/emoscript/cli.py         fit / score / eval / info / corpus-check / fetch-axis
src/emoscript/data/          corpora as JSONL
experiments/validate_axis.py held-out validation protocol
experiments/select_settings.py train-only nested CV for k / normalization / L2
experiments/report.py        validation JSON -> Markdown
experiments/compare_models.py the protocol across embedding models
docs/results.md              current numbers
tests/                       model-free test suite (a surface-feature fake embedder drives the pipeline)
.github/workflows/           ci.yml (lint, tests, real evaluation), release.yml (prebuilt axes on tags)
```

Axis files are `.npz` and backward compatible: a v0.1 file with a single
`axis` vector loads as a k=1 model without calibrator.

## Limitations

- **Short beats long.** Famous taglines are short; the axis still rewards
  brevity more than it should, and long benefit-led copy scores low. Compare
  candidates of similar length, or read `--explain` before cutting a line.
- **Memorable is not usable.** Proverbs, lyrics, whimsical nonsense and
  parodies of famous slogans score like great copy. Use `--brief`, read the
  near-famous warnings, and treat the junk PASS rate in `docs/results.md` as
  the honest floor.
- **Topic still leaks a little**, and the corpus is Anglophone with a
  20th-century US/UK canon; the axis inherits that taste. Other languages
  score, but are not calibrated against this corpus (Japanese lines came out
  about 0.04 higher than their English counterparts in testing).
- **Formatting matters less than it did, not nothing.** Canonical text
  removes casing, punctuation and wrappers; paraphrases still move scores by
  roughly a third of the spread between different lines.
- The held-out set is a few hundred lines; read the intervals, not the point
  estimates, and never compare two settings on the test split.
- The tool cannot tell you whether a line is *available*, *appropriate* or
  *effective*, only whether it sits stylistically near famous taglines.

## Contributing

The corpus is the product. See `CONTRIBUTING.md`.

## License

MIT (code and original corpus contributions). Third-party taglines remain the
property of their respective owners; see the note in `LICENSE`.

---

## 日本語の概要

emoscript は、英語の広告キャッチコピーについて「名コピーが埋め込み空間で占める方向
（意味軸）」を推定し、任意の文案にコサイン類似度で「名コピーらしさ」を与える
ライブラリです。文才の絶対評価ではなく、大量の文案の一次スクリーニング・
ランキング用のシグナルです。

- 軸 = normalize(正例重心 − 負例重心)。負例に「それっぽい駄文」「短いTitle Caseの断片」
  「UI定型文」「商品説明の凡庸文」を含めるのが核心。
- 埋め込み前に小文字化・末尾句読点除去・LLM出力のラッパー除去（canonical）を行い、
  書式だけで点が動かないようにしている。
- 交差適合した out-of-fold スコアにロジスティック回帰を載せ、確率と閾値を出力。
  `--prior` で事前確率を指定できる。
- 入力フラグ、名作コピーの模倣検出、`--brief` によるブリーフ適合度を併用する。
- 持ち出しテストはブランド単位で分離。全テスト正例に同一話題の凡庸文を対にして
  文体純度を測定。名言・ことわざ、ゴミ文字列の対照群もある。
- 設定（軸数・正規化・正則化）は train 内の入れ子CVで選び、test は版ごとに1回だけ評価。
  実測値は `docs/results.md` を参照。

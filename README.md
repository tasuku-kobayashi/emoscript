# emoscript

**Find the direction of great ad copy.** emoscript estimates the "great
tagline" direction in sentence-embedding space from a corpus of famous
advertising slogans, then scores any new line by cosine similarity against
that direction.

It is a *ranking signal*, not an oracle: it measures stylistic proximity to the
patterns that made classic taglines famous (brevity, rhythm, concrete imagery,
double meaning), not legal clearance or guaranteed market performance. Use it
to triage and rank hundreds of candidate lines, then let a human pick.

## How it works

1. Embed a curated corpus of famous taglines (*"Just Do It."*, *"Think
   Small."*, *"Got Milk?"*) with a sentence-embedding model.
2. Embed a negative corpus, crucially including **hard negatives**: fluent,
   plausible-sounding but utterly generic ad-speak (*"Quality products at
   affordable prices for everyone."*).
3. Build the axis as the difference of the two centroids and score by cosine:

   ```
   axis        = normalize( mean(embed(great)) - mean(embed(bland)) )
   score(line) = cos(embed(line), axis)
   ```

The subtraction is the whole trick. A centroid built from great copy alone
mostly learns "well-formed marketing English"; subtracting the bland centroid
isolates what is *distinctive* about great copy.

On top of that single axis, v0.2 adds:

- **Multiple axes (`--k`).** Spherical k-means splits the positives into
  k flavours (terse / informational / humorous ...) and fits one diff-axis per
  cluster. A line's score is its best match over the axes, so a famous but
  long, information-dense slogan (GEICO's *"15 Minutes Could Save You 15% or
  More on Car Insurance"*) is no longer penalised for not being terse.
- **Calibrated probabilities.** A one-feature logistic regression maps the raw
  cosine to P(great copy). It is fitted on *out-of-fold* scores (5-fold
  cross-fitting on the training corpus), so the probabilities, the
  `PASS`/`fail` threshold and the reported training AUC are honest.
- **Style-vs-topic control.** Every held-out tagline has an original,
  same-topic bland rewrite (*"Belong Anywhere"* vs *"Find unique places to
  stay around the world."*). Separating those pairs proves the axis is
  reading style, not product category.
- **Case normalisation (`--casing lower`).** Famous taglines are written in
  Title Case and ordinary copy in sentence case. Embedding everything
  lower-cased removes that typographic cue so you can see how much of the
  separation is genuinely about the writing.

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
# Fit axes (+ calibrator) from the bundled corpus
emoscript fit --k 3 --out axis.npz

# Score candidate lines (higher = more great-copy-like); p = calibrated probability
emoscript score "Think fresh." "We sell high quality furniture at low prices." --axis axis.npz
#  +0.2731  p=0.91  PASS  Think fresh.
#  -0.0412  p=0.07  fail  We sell high quality furniture at low prices.

# Rank a file of candidates, one per line, as JSON
emoscript score --file candidates.txt --sort --json --axis axis.npz

# Held-out evaluation of a fitted axis
emoscript eval --axis axis.npz

# Download a prebuilt axis instead of fitting (attached to GitHub Releases)
emoscript fetch-axis --k 3 --out axis.npz

# Corpus integrity check (no model needed) and axis file details
emoscript corpus-check
emoscript info --axis axis.npz
```

Python:

```python
from emoscript import AxisModel, fit_from_corpus, make_embed_fn, score_texts

embed = make_embed_fn()                           # fastembed, default model
model = fit_from_corpus(embed, k=3, casing="lower")
model.save("axis.npz")

model = AxisModel.load("axis.npz")
scores, probs = score_texts(model, ["Melts in your mouth, not in your funnel."], embed)
print(scores[0], probs[0], model.threshold_for(0.5))
```

Any `texts -> (n, dim)` callable works as `embed`, so you can plug in another
provider; `fit_from_embeddings` takes precomputed matrices directly.

## The corpus

| split | v0.1 | v0.2 | contents |
|---|---:|---:|---|
| `slogans_train` | 40 | 211 | famous taglines, brand and year attributed |
| `slogans_test` | 10 | 67 | held-out taglines, **brand-disjoint** from train |
| `negatives_hard_train` / `_test` | 20 / 10 | 52 / 22 | original bland-but-plausible ad lines |
| `negatives_generic_train` / `_test` | 10 / 5 | 22 / 11 | original transactional / factual prose |
| `contrast_pairs_test` | – | 67 | an original same-topic bland rewrite of every held-out tagline |

Rules enforced by `emoscript corpus-check` (and the test suite): no text in
both train and test, no near-duplicates across splits (token Jaccard ≥ 0.7),
no brand in both splits, every held-out tagline has exactly one contrast pair.
The v0.1 test set contained *"It's Finger-Lickin' Good"*, a variant of a
training line; it was removed.

The positives span 235 brands, US and UK, 1872–2016, and include deliberately
long or grammar-breaking classics (Rolls-Royce's electric clock, the Big Mac
ingredient list, *"Lipsmackin' Thirstquenchin' ..."*) so the multi-axis
variant has a real second mode to find.

**Trademark notice:** famous taglines are trademarks of their respective
owners, included solely as factual, attributed reference data for research.
All negatives and contrast rewrites are original writing.

## Evaluation

`python experiments/validate_axis.py` fits every variant on the train splits
and reports, on the held-out splits only:

- AUC of famous taglines vs hard negatives, vs generic prose, and vs the
  topic-matched bland rewrites, plus the paired win rate on those rewrites;
- the same AUC broken down by tagline length (≤4 / 5–7 / ≥8 words), which is
  where the single axis used to fail;
- calibration quality of the probabilities on the held-out set (log-loss,
  Brier, accuracy at p=0.5, and the raw-score threshold that corresponds to
  p=0.5);
- the lowest-scoring famous lines and highest-scoring bland lines, and a full
  ranking. Add `--casing lower` for the typography-free run and `--json` for a
  machine-readable report. `experiments/compare_models.py` runs the same
  protocol across several fastembed models.

**v0.1 result (40/30 train, 10/10/5 test):** the diff-axis against hard
negatives alone separated held-out taglines from held-out bland copy
perfectly (AUC 1.00); hard+generic reached 0.99 vs bland and 1.00 vs prose,
its one miss being the GEICO line. A positive-only centroid managed 0.92.

**v0.2 result:** the corpus is roughly five times larger and the test set is
brand-disjoint, so the v0.1 numbers no longer apply. The `evaluate` job in
`.github/workflows/ci.yml` runs the full protocol with the default model on
every pull request and publishes `validation.txt` / `validation.json` as a
workflow artifact (and in the job summary); the release workflow attaches the
same report next to the prebuilt axes. Please read the numbers from there
rather than from this file.

## Repository layout

```
src/emoscript/axis.py        axes (mean / diff / k-means multi-axis), AUC, calibrator, metrics  (numpy only)
src/emoscript/pipeline.py    fit_from_corpus, cross-fitted calibration, AxisModel save/load, evaluate
src/emoscript/embed.py       fastembed backend, model prefixes, pluggable embed function
src/emoscript/text.py        case normalisation
src/emoscript/dataio.py      corpus loading + integrity checks
src/emoscript/cli.py         fit / score / eval / info / corpus-check / fetch-axis
src/emoscript/data/          corpora as JSONL
experiments/validate_axis.py held-out validation protocol
experiments/compare_models.py same protocol across embedding models
tests/                       model-free test suite (a surface-feature fake embedder drives the pipeline)
.github/workflows/           ci.yml (lint, tests, real evaluation), release.yml (prebuilt axes on tags)
```

Axis files are `.npz` and backward compatible: a v0.1 file with a single
`axis` vector loads as a k=1 model without calibrator.

## Limitations

- A short line's embedding is influenced by topic as well as style; the
  topic-matched contrast pairs measure that, they do not remove it.
- The corpus is Anglophone and skews to the 20th-century US/UK canon; the axis
  inherits that taste.
- Casing is a confound: compare `--casing none` and `--casing lower` before
  trusting a number.
- Even at a few hundred lines the evaluation set is small; treat scores as one
  signal among many and sanity-check rankings.
- The tool cannot tell you whether a line is *available*, *appropriate* or
  *effective*, only whether it sits stylistically near famous taglines.

## Roadmap

- Read the v0.2 numbers off CI and record them here; adopt `--casing lower`
  as the default if the gap is small.
- Keep growing the hard negatives, in particular Title-Case bland lines and
  same-topic rewrites for the training set.
- Per-cluster labels for the multi-axis model (which flavour did a line match).
- Evaluate stronger / English-only embedding models with
  `experiments/compare_models.py` and switch the default if one wins clearly.

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

- 軸 = normalize(正例重心 − 負例重心)。負例に「それっぽい駄文」を含めるのが核心。
- `--k` で k-means による複数軸化（簡潔系・情報系など）、最大値で判定。
- 交差適合した out-of-fold スコアにロジスティック回帰を載せ、確率と閾値を出力。
- 持ち出しテストはブランド単位で分離。全テスト正例に同一話題の凡庸文を対にして
  文体純度を測定。`--casing lower` で大文字小文字の交絡を除去。
- 実モデルでの評価値は CI（`evaluate` ジョブ）の成果物を参照。

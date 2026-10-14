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

   The bland mean weights each negative *family* equally (ad-speak, short
   fragments, UI strings, product descriptions, nonsense, prose), so the
   largest family cannot decide the direction on its own.

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
- **Guards the score cannot provide, and they decide.** Input flags (single
  word, over 17 words, placeholder, chat-model chatter such as *"Sure, here
  is a tagline:"*, non-Latin script, wrapped) and a lexical check against the
  278 famous lines in the corpus, compared word by word so *"Just. Do. It."*
  and *"I’m lovin’ it"* count as copies. A verbatim copy or a blocking flag
  makes a row not pass and sorts it last, however high its score; a parody
  (*"Just Chew It."*, *"Kills Kids Dead"*) or a possible riff (*"Got Mold?"*)
  is a warning, because ordinary short copy sits one word away from some
  famous line too often for a lexical rule to be allowed a veto. With
  `--brief`, relevance to the brief is shown, `--sort` ranks by style and
  relevance together (z-scored over the unblocked rows), and `--top-rel`
  (relative to the batch) or `--min-rel` (absolute) turns relevance into a
  gate, because the style axis alone prefers *"Chase the Gravy"* to
  on-brief copy.
- **Explanations.** `--explain` shows the nearest famous taglines to a
  candidate.
- **A second "sense" axis, used as a gate.** Famous lines minus the nonsense
  and UI-string families, with its own out-of-fold calibrator: a line must
  score P(message) ≥ 0.5 on it to PASS, whatever its style score. It takes
  the junk pass rate from 42% to 4% while 97% of usable copy clears it
  (`--no-sense-gate` to disable; `sense=` is printed). Famous proverbs and
  lyrics clear the gate too; `--brief` with `--top-rel` is the tool for them.
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
emoscript fit --out axis.npz            # negative families weighted equally; --no-balance-negatives for per-line

# Score candidates (higher = more great-copy-like); p = calibrated probability
emoscript score "Sleep like you mean it." "We sell high quality furniture at low prices." --axis axis.npz

# A raw LLM candidate list: strict prior, product brief with a relevance gate, ranked, explained
emoscript score --file candidates.txt --axis axis.npz --prior 0.05 \
    --brief "meal-kit delivery for busy parents" --top-rel 0.5 --sort --explain

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

Real output of `emoscript score --file lines.txt --prior 0.3 --brief "a mattress brand" --sort`
with the default model (details and the full evaluation in
[`docs/results.md`](docs/results.md)):

```
+0.0823  p=0.93  PASS  rel=0.85  sense=1.00  Sleep like you mean it.
+0.0818  p=0.88  PASS  rel=0.80  sense=0.79  Just Chew It.
          ! parodies famous line: "Just Do It." (Nike)
-0.0925  p=0.03  fail  rel=0.84  sense=0.99  Comfort, redefined for modern living.
+0.0868  p=0.85  fail  rel=0.81  sense=0.16  Ride the Spoon.
          ! not passing: sense_gate
+0.0923  p=0.89  fail  rel=0.80  sense=0.47  Chase the Gravy
          ! not passing: sense_gate
+0.0434  p=0.60  fail  rel=0.79  sense=0.39  1. "Just. Do. It."
          ! copies famous line: "Just Do It." (Nike); flags: wrapped; not passing: near_famous:copy, sense_gate
+0.0152  p=0.46  fail  rel=0.79  sense=0.93  Sure, here is a tagline:
          ! flags: llm_chatter; not passing: llm_chatter
-0.0483  p=0.03  fail  rel=0.79  sense=0.00  Lorem Ipsum
          ! flags: placeholder; not passing: placeholder, sense_gate
   n/a   skipped (no letters)  ?!
          ! flags: no_letters; not passing: no_letters
```

Each row shows the raw cosine, the calibrated probability, `PASS` or `fail`,
`rel=` when a brief is given, then the line; the indented line below carries
the famous-line match, the input flags and the reasons a row does not pass.
Inputs without letters are reported as `n/a skipped`.

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
| `slogans_test` | 10 | 173 | held-out taglines, **brand-disjoint** from train (sister brands of training brands flagged and reported separately) |
| `negatives_hard_train` / `_test` | 20 / 10 | 329 / 85 | original bland ad-speak, short Title-Case fragments, questions / numbers / B2B, UI boilerplate, topic-descriptive product lines |
| `negatives_generic_train` / `_test` | 10 / 5 | 30 / 22 | original transactional / factual prose |
| `contrast_pairs_test` | – | 173 | an original same-topic bland rewrite of every held-out tagline (naming the brand when the famous line does) |
| `quotes_test` | – | 50 | famous proverbs and literary / historical lines that are not advertising |
| `junk_test` | – | 80 | placeholders, UI strings and whimsical non-sequiturs that must fail |
| `good_copy_test` | – | 88 | original, usable, unknown lines that should pass: the cost side of every junk measure |

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

The short version (canonical text, the shipped default; intervals and the
full tables in `docs/results.md`): 173 held-out famous taglines vs bland
ad-speak **AUC 0.95**, vs plain prose 0.97, vs same-topic product
descriptions 0.94 with a paired win rate of 0.97, vs junk 0.79 by style
score, and vs famous non-advertising quotes 0.47. With raw text a
six-feature typography model matches the axis, which is why raw text is
not the default. The sense gate takes the junk PASS rate from 42% to 4% at
no cost to usable copy and a six-point cost on famous lines.

The honest ceiling comes from the good-copy control, 88 original usable
lines that are not famous: 43% of those at tagline length pass, 12% of the
longer benefit-led and B2B lines do. The axis learned the shape of a famous
tagline. Use it for tagline-length candidates and read the near-famous,
sense and brief signals next to the score; it is not a scorer for body copy.

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
- **Memorable is not usable.** Proverbs, lyrics and whimsical nonsense score
  like great copy (held-out famous taglines vs famous non-ad quotes: AUC
  about 0.5). Copies and parodies of corpus slogans are caught lexically
  (copies block, parodies warn); nonsense is not. Use `--brief` with
  `--min-rel`, and treat the junk PASS rate in `docs/results.md` as the
  honest floor. The input-flag patterns were written with the junk control
  in view, so "junk after flags" is partly in-sample; the junk AUC is not.
- **One-word lines are blocked.** *"Think"*, *"Lemon."* and *"Zoom-Zoom"*
  are famous, but a single word is also the easiest way to game a
  brevity-loving axis, so `single_word` is a blocking flag; the score is
  still printed.
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

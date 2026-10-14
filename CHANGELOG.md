# Changelog

## 0.2.0

### Evaluation set and controls (second review round)
- Held-out taglines 67 → 173 (brand-disjoint; 53 lines from sister brands of
  training brands are flagged `sister_brand` and reported separately), with
  brandless contrast rewrites for every one; hard / generic test negatives
  55 / 15 → 85 / 22; junk 45 → 80; quotes 30 → 50.
- `good_copy_test.jsonl`: 88 original, usable, unknown lines across registers,
  reported as PASS rate overall and by length. It shows that the axis scores
  tagline-shaped writing: 43% of tagline-length lines pass, 12% of longer
  benefit-led lines.
- Sense axis (famous minus nonsense + UI families) feeding a two-feature
  calibrator; junk PASS 0.42 → 0.29 at no cost to famous lines.

Renamed from `copyaxis` to `emoscript` (import path, CLI, docs).

### Corpus
- Positives 50 → 278 (211 train / 67 test) across 235 brands, 1872–2016,
  with brand and year, a `kind` tag for headlines / jingles / mottos / posters,
  and attribution fixes from review.
- Hard negatives 30 → 353 (298 train / 55 test) in seven families: bland
  ad-speak, short / Title Case / no full stop, questions-numbers-B2B, UI
  boilerplate, topic-descriptive product descriptions on the training brands,
  whimsical non-sequiturs. Generic prose negatives 15 → 45. All original
  writing.
- New evaluation-only controls: `contrast_pairs_test.jsonl` (an original
  same-topic bland rewrite of every held-out tagline, naming the brand when
  the famous line does), `quotes_test.jsonl` (30 famous non-advertising
  lines), `junk_test.jsonl` (45 placeholders, UI strings and whimsical
  non-sequiturs that must fail).
- Train/test are brand-disjoint and near-duplicate free; `emoscript
  corpus-check` enforces the rules and prints soft warnings for content-word
  overlap across classes.

### Model
- `fit_multi_axis` uses raw cluster means (the unit-length spherical centre
  minus a raw negative mean leaned every axis along the common direction);
  k-means re-seeds empty clusters with distinct points.
- Calibrated probabilities: balanced one-feature logistic regression fitted
  on 5-fold out-of-fold scores, `prior=` to shift the base rate, degenerate
  and NaN guards.
- Text normalization stored in the model: `canonical` (default) lower-cases,
  strips terminal punctuation and quotes, HTML / markdown, list markers and
  "Tagline:" labels; also `lower`, `lower_nopunct`, `sentence`, `none`.
- `AxisModel` with `save` / `load` (`.npz`, reads v0.1 files), exemplars for
  `explain`, the famous-line catalogue for `near_famous`, cluster summaries.
- Negative families (ad-speak, short, questions/numbers/B2B, UI,
  topic-descriptive, nonsense, prose) weighted equally in the negative
  centroid (`--no-balance-negatives` for one vote per line).
- Defaults k=1, canonical, L2=1, balanced families chosen by brand-grouped
  nested CV on the training split (`experiments/select_settings.py`); the
  test split was evaluated once per corpus version.
- Guards decide: blocking input flags (no letters, single word, over 17
  words, placeholder, LLM chatter) and verbatim famous-line copies make a
  row not pass and sort it last; parodies and possible riffs are warnings.
  `--brief` ranks by style + relevance (z-scored over unblocked rows) and
  `--min-rel` gates on relevance; each line is embedded once.
- Pluggable embedding function; fastembed is an optional extra.

### Tooling
- CLI: `fit`, `score` (args / file / stdin, `--json`, `--sort`,
  `--threshold`, `--prior`, `--brief`, `--explain`, input flags, near-famous
  warnings, `n/a` for inputs without letters), `eval`, `info`,
  `corpus-check`, `fetch-axis`.
- `experiments/validate_axis.py`: AUC with brand-cluster / paired bootstrap
  intervals vs hard, generic, topic-matched, quotes and junk; pair win rate
  with Wilson interval; length breakdown; calibration over every bland group;
  PASS rates per group and junk PASS rate after input flags; surface-feature
  and kNN references; `--casing`, `--json`. `experiments/report.py` renders
  Markdown; `experiments/compare_models.py` runs the protocol across models.
- Model-free test suite (surface-feature fake embedder), ruff, GitHub Actions
  CI with a real-model evaluation job, release workflow that attaches prebuilt
  axes.

## 0.1.0

Initial release as `copyaxis`: diff-axis scorer, 40/10 famous taglines,
synthetic negatives, held-out validation script.

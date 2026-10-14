# Changelog

## 0.2.0

Renamed from `copyaxis` to `emoscript` (import path, CLI, docs).

### Corpus
- Positives 50 → 278 (211 train / 67 test) across 235 brands, 1872–2016,
  with brand and year, a `kind` tag for headlines / jingles / mottos / posters,
  and attribution fixes from review.
- Hard negatives 30 → 333 (278 train / 55 test) in six families: bland
  ad-speak, short / Title Case / no full stop, questions-numbers-B2B, UI
  boilerplate, topic-descriptive product descriptions on the training brands.
  Generic prose negatives 15 → 45. All original writing.
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
- Defaults k=1, canonical, L2=1 chosen by brand-grouped nested CV on the
  training split (`experiments/select_settings.py`); the test split was not
  consulted.
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

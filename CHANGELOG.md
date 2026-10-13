# Changelog

## 0.2.0

Renamed from `copyaxis` to `emoscript` (import path, CLI, docs).

### Corpus
- Positives 50 → 278 (211 train / 67 test) across 235 brands, 1872–2016,
  including long and grammar-breaking classics.
- Hard negatives 30 → 74, generic negatives 15 → 33 (all original writing).
- New `contrast_pairs_test.jsonl`: an original same-topic bland rewrite of
  every held-out tagline, for the style-vs-topic control.
- Train/test are now brand-disjoint; the KFC variant that leaked into the
  v0.1 test set was removed. `emoscript corpus-check` enforces the rules.

### Model
- Multi-axis scoring (`--k`): spherical k-means over the positives, one
  diff-axis per cluster, max over axes.
- Calibrated probabilities: one-feature logistic regression fitted on 5-fold
  out-of-fold scores; `PASS`/`fail` at a probability threshold; honest
  training AUC reported at fit time.
- Case normalisation (`--casing none|lower|sentence`) stored in the axis file
  and applied identically at scoring time.
- `AxisModel` with `save`/`load` (`.npz`, backward compatible with v0.1
  single-axis files), metadata and version stamp.
- Pluggable embedding function; fastembed is now an optional extra
  (`pip install "emoscript[embed]"`); model-family task prefixes.

### Tooling
- CLI: `fit`, `score` (args / file / stdin, `--json`, `--sort`,
  `--threshold`), `eval`, `info`, `corpus-check`, `fetch-axis`.
- `experiments/validate_axis.py`: AUC vs hard / generic / topic-matched
  negatives, paired win rate, length breakdown, calibration metrics, JSON
  output, `--casing`. `experiments/compare_models.py` runs it across models.
- Model-free test suite (surface-feature fake embedder), ruff, GitHub Actions
  CI with a real-model evaluation job, release workflow that attaches prebuilt
  axes.

## 0.1.0

Initial release as `copyaxis`: diff-axis scorer, 40/10 famous taglines,
synthetic negatives, held-out validation script.

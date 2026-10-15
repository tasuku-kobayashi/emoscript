# Changelog

## Unreleased

### Browser edition
- `web/`: a static GitHub Pages app that embeds in the browser with
  transformers.js (`Xenova/bge-small-en-v1.5`, fp32 by default) and ports the
  scorer to JavaScript: canonical normalization, input flags, the near-famous
  check, calibrated p with prior shift, the sense gate, brief relevance with
  `--min-rel` / `--top-rel` equivalents, the same sort order, CSV / JSON export.
- `emoscript export-web` writes the axis, calibrators, sense axis and famous
  catalogue as JSON; `WEB_MODELS` maps fastembed names to transformers.js ids
  and pooling.
- Parity tests (`web/test`): model-free checks against a Python-generated
  fixture, and an end-to-end run through transformers.js that must agree with
  fastembed within 0.01 in score (observed 0.0004 at fp32; int8 drifts to
  0.03–0.05, so it is labelled approximate in the UI). CI runs both; a Pages
  workflow deploys `web/`.
- Held-out numbers for the browser model are in docs/results.md: hard 0.942,
  topic-matched 0.789 (e5-large: 0.942), topic_bland PASS 0.43 (0.17).
- After an adversarial review of the port: Unicode-aware digit / word
  classes in the JavaScript regexes (full-width "Option ２:" strips as in
  Python, so a famous copy behind it is still blocked); the hub revision of
  the browser model is pinned in model.json; the parity test matches rows by
  occurrence, compares flags, near-famous, sense, p and relevance at 0.002,
  and the fixture carries 102 lines including the divergence probes; the app
  rejects pending embeddings when the worker is replaced, ignores superseded
  runs, falls back to WASM when WebGPU fails mid-run, caps input (5000 lines,
  2000 characters), neutralises formula-like cells in the CSV and marks
  results stale when the input changes.

- UI redesign: one text box, one button. The result leads with a hero score
  (a large ring and number for the selected line, colored by state: green
  passes, amber close, red ordinary, gray disqualified, always with the word
  beside it), an information list (sense, relevance, style score, famous-line
  match, why it is out) and a ranking of all lines with their scores; tapping
  a line puts it in the hero. Options and explanations sit behind the ? and ⚙
  buttons. Dark mode. The headline is a seven-step grade on the style score
  (Excellent ≥ +0.15, Great ≥ +0.10, Good ≥ +0.05, Fair ≥ 0, Average ≥ −0.05,
  Weak ≥ −0.12, Flat), cut on the browser model's held-out distribution; the
  calibrated p and the pass verdict sit under it, and each information row
  says what high and low mean.

### Fixed
- `placeholder` flag: the `<BRAND>` pattern was case-insensitive, so
  `<strong>Fresh every day</strong>` and `<em>…</em>` were blocked as
  placeholders. Upper-case tags only now (Python and JS).

### Known divergences of the browser port (display flags only)
- `non_latin` uses `\p{Script=Latin}`, Python the Unicode name prefix: full-width
  Latin letters and modifier letters are flagged by Python only.
- Python's whitespace set (`\x1c`–`\x1f`, `\x85`) and case-insensitive
  matches (ſ, K, İ) differ from JavaScript's in corner cases.

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
- Sense axis (famous minus nonsense + UI families) with its own
  out-of-fold calibrator used as a PASS gate: junk PASS 0.42 → 0.04, good
  copy unchanged (97% clear the gate), famous lines 0.81 → 0.75.
  `--no-sense-gate`, `sense=` in output, `--top-rel` for a batch-relative
  brief gate; chatter and placeholder patterns relaxed after seven false
  blocks on real copy.

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

# Contributing to emoscript

Thanks for your interest! The corpus is the heart of this project: better data
beats cleverer code.

## Adding famous taglines (positives)

- Edit `src/emoscript/data/slogans_train.jsonl`, or `slogans_test.jsonl` if you
  are adding evaluation data. One JSON object per line:
  `{"text": "...", "brand": "...", "year": 1959}` (`year` may be a decade
  string such as `"1950s"` when the exact year is not documented).
- Only taglines that are genuinely famous and documented in published sources
  (books on advertising history, reputable press). Include brand and year.
- **Train and test are brand-disjoint.** A brand that already appears in one
  split must stay in that split. Never let a line, or a close variant of a
  line, appear in both.
- Every line added to `slogans_test.jsonl` needs a matching row in
  `contrast_pairs_test.jsonl`: `{"famous": "<the tagline>", "bland": "<your
  original, same-topic, deliberately forgettable rewrite>", "brand": "..."}`.
- Do not copy large corpora of slogans from other datasets or compilations;
  add lines individually with attribution.

## Adding negatives

- **Hard negatives** (`negatives_hard_*.jsonl`): fluent, professional, ad-like
  lines that say nothing, the kind of copy the axis should learn to *reject*.
  These must be your own original writing. Title-Case hard negatives are
  especially welcome (they remove a typographic shortcut).
- **Generic negatives** (`negatives_generic_*.jsonl`): plain transactional or
  informational prose. Also original writing.
- Never mine real ads for negatives: reproducing them wholesale raises the
  same copyright issues we avoid by writing our own.

## Checks to run

```bash
pip install -e ".[dev]"
emoscript corpus-check      # integrity: fields, leaks, brand overlap, contrast coverage
ruff check src tests experiments && ruff format --check src tests experiments
pytest                      # model-free; a fake embedder drives the full pipeline
```

With the embedding extra installed (`pip install -e ".[dev,embed]"`, ~2 GB
model download on first use) also run the real evaluation and paste the
summary into your PR:

```bash
python experiments/validate_axis.py --no-ranking
python experiments/validate_axis.py --no-ranking --casing lower
```

CI runs the same evaluation on every pull request and publishes the report as
a workflow artifact. A good corpus change keeps the held-out AUCs and the
topic-matched pair win rate flat or better, in both casing modes. If your
change makes things worse, say so in the PR; negative results are valuable
here too.

## Code

- Python 3.10+, `numpy` only in the core; `fastembed` stays an optional extra
  behind `emoscript.embed`.
- Anything that needs embeddings must accept an injected `embed_fn` so it can
  be tested with `tests/conftest.py::FakeEmbedder`.
- Keep `ruff` clean and add tests for new behaviour.

## Releasing

Tag `vX.Y.Z` on `main`. The release workflow runs the tests, builds the
package, fits the prebuilt axes (k=1 and k=3) with the default model, and
attaches them with the evaluation report to the GitHub Release;
`emoscript fetch-axis` downloads them.

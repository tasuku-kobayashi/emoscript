# Contributing to emoscript

Thanks for your interest! The corpus is the heart of this project: better data
beats cleverer code, and the two adversarial reviews that shaped v0.2 both
came back to the same point: every cheap cue the negatives fail to cover
becomes a shortcut the axis will take.

## Adding famous taglines (positives)

- Edit `src/emoscript/data/slogans_train.jsonl`, or `slogans_test.jsonl` if you
  are adding evaluation data. One JSON object per line:
  `{"text": "...", "brand": "...", "year": 1959}` (`year` may be a decade
  string such as `"1950s"` when the exact year is not documented). Add
  `"kind": "headline" | "jingle" | "motto" | "poster" | "billing"` when the
  line is a famous piece of advertising but not strictly a tagline.
- Only lines that are genuinely famous and documented in published sources
  (books on advertising history, reputable press). Include brand and year.
- **Train and test are brand-disjoint.** A brand that already appears in one
  split must stay in that split. Never let a line, or a close variant of a
  line, appear in both. (Sister brands of one company are allowed on both
  sides; note it in the PR if you add one.)
- Every line added to `slogans_test.jsonl` needs a matching row in
  `contrast_pairs_test.jsonl`: `{"famous": "<the tagline>", "bland": "<your
  original, same-topic, deliberately forgettable rewrite>", "brand": "..."}`.
  If the famous line names the brand, the bland rewrite must name it too.
- Do not copy large corpora of slogans from other datasets or compilations;
  add lines individually with attribution.

## Adding negatives

All negatives must be your own original writing. Never mine real ads:
reproducing them wholesale raises the copyright issues we avoid by writing
our own. The families, with the `note` value to use:

| family | note | what it teaches the axis |
|---|---|---|
| bland ad-speak | `generic-*` | fluent, professional lines that say nothing |
| short / Title Case / no full stop | `short-titlecase`, `short-lower`, `mid-titlecase` | that brevity and typography alone are not greatness |
| questions, numbers, B2B | `question-number-b2b` | registers the famous lines rarely use |
| UI and boilerplate | `ui-boilerplate` | "Add to Cart", "Page Not Found" are not copy |
| topic-descriptive | `topic-descriptive` | a plain product description on the topic of a training brand, so topic words stop counting |
| plain prose | `negatives_generic_*` | transactional and factual sentences |

Vary the format within every family (sentence case with a period, Title Case
without one, lower case). `note` values are validated against
`emoscript.dataio.NOTE_PREFIXES` and decide the family weight, so use the
exact ones above. Lines that could pass as real taglines belong in the
positives or nowhere; "The difference is in the details." was flagged by
review as too good for a negative.

`junk_test.jsonl` (placeholders, UI strings, whimsical nonsense) and
`quotes_test.jsonl` (famous non-advertising lines) are evaluation-only
controls; add to them freely, never to the training negatives.
`corpus-check` rejects training negatives that duplicate or nearly duplicate
a control line, so a new nonsense negative must use different nouns and
frames than the junk control ("Bounce the Ladder", not "Bounce the Gravy").

## Checks to run

```bash
pip install -e ".[dev]"
emoscript corpus-check      # integrity: fields, leaks, brand overlap, contrast coverage (+ soft warnings)
ruff check src tests experiments && ruff format --check src tests experiments
pytest                      # model-free; a fake embedder drives the full pipeline
```

With the embedding extra installed (`pip install -e ".[dev,embed]"`, ~2 GB
model download on first use) also run the real evaluation and paste the
summary into your PR:

```bash
python experiments/validate_axis.py --no-ranking --casing canonical   # the shipped setting
python experiments/validate_axis.py --no-ranking                      # raw text, for reference
```

CI runs the same evaluation on every pull request and publishes the report as
a workflow artifact. A good corpus change keeps the held-out AUCs, the
topic-matched pair win rate and the junk PASS rate flat or better **in the
canonical setting**. If your change makes things worse, say so in the PR;
negative results are valuable here too.

Settings (number of axes, text normalization, calibrator penalty) are chosen
with `experiments/select_settings.py`, which cross-validates on the training
split only. Do not tune them by looking at the test split.

## Code

- Python 3.10+, `numpy` only in the core; `fastembed` stays an optional extra
  behind `emoscript.embed`.
- Anything that needs embeddings must accept an injected `embed_fn` so it can
  be tested with `tests/conftest.py::FakeEmbedder`.
- Keep `ruff` clean and add tests for new behaviour.

## Releasing

Tag `vX.Y.Z` on `main`. The release workflow runs the tests, builds the
package, fits the prebuilt axes (k=1 and k=2, canonical text) with the
default model, and attaches them with the evaluation report to the GitHub
Release; `emoscript fetch-axis` downloads them.

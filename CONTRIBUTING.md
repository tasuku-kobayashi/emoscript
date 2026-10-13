# Contributing to copyaxis

Thanks for your interest! The corpus is the heart of this project — better
data beats cleverer code.

## Adding famous taglines (positives)

- Edit `src/copyaxis/data/slogans_train.jsonl` (or `slogans_test.jsonl` if you
  are adding evaluation data — never let a line appear in both).
- One JSON object per line: `{"text": "...", "brand": "...", "year": 1959}`.
- Only taglines that are genuinely famous and documented in published sources
  (books on advertising history, reputable press). Include brand and year.
- Do not copy large corpora of slogans from other datasets or compilations;
  add lines individually with attribution.

## Adding negatives

- **Hard negatives** (`negatives_hard_train.jsonl`): fluent, professional,
  ad-like lines that say nothing — the kind of copy the axis should learn to
  *reject*. These must be your own original writing.
- **Generic negatives** (`negatives_generic_train.jsonl`): plain transactional
  or informational prose. Also original writing.
- Never mine real ads for negatives: reproducing them wholesale raises the
  same copyright issues we avoid by writing our own.

## Validation

Before opening a PR, reproduce the evaluation from the repo root:

```bash
pip install -e ".[dev]"
python experiments/validate_axis.py
pytest
```

A good corpus change keeps held-out AUC at 1.00 (or improves the margin
between positive and negative score distributions). If your change makes
things worse, say so in the PR — negative results are valuable here too.

## Code

- Python 3.10+, `numpy` + `fastembed` only; keep dependencies minimal.
- Run `pytest` before submitting.

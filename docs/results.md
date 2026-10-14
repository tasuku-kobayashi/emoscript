# Results (v0.2 corpus, `intfloat/multilingual-e5-large`)

Generated from `experiments/validate_axis.py --k 1,2 --json ...` (raw text and
`--casing canonical`) with `experiments/report.py`, on the held-out splits
only. Settings (k=1, canonical text, calibrator L2=1, negative families
weighted equally) were chosen beforehand by brand-grouped nested
cross-validation on the training split (`experiments/select_settings.py`:
out-of-fold AUC 0.921, balanced log-loss 0.353; L2 = 0.1 and 1 tie within
0.001 and 1 was kept as the smoother prior); the test split was evaluated
once for this corpus version.

Columns: AUC of held-out famous taglines against each negative group with a
95% bootstrap interval (brands resampled as clusters; pairs as units for the
topic-matched rewrites), the paired win rate with its Wilson interval, the
quotability control (famous taglines vs famous non-advertising lines), junk,
AUC for the 11 held-out taglines of 8+ words, PASS rates at p ≥ 0.5 for
famous / bland (mean of hard, generic, topic-matched) / junk (and junk after
the blocking input flags a CLI user sees), and the calibration log-loss over
every bland group. Rows marked "not an axis" are references.

## How to read it

- **The shipped setting is the canonical table, row "M2 diff (hard+generic,
  balanced families)".** The raw-text table is there to show how much of the
  separation typography provides: the six-feature surface model on raw text
  reaches 0.99 against the topic-matched rewrites and 0.93 against quotes
  without reading a word, which is why raw text is not the default.
- With canonical text the single axis separates famous taglines from bland
  ad-speak (0.95), from plain prose (0.99) and from same-topic product
  descriptions (0.92, pair win 0.94). It separates them from junk at 0.78:
  40% of the junk strings still pass on score alone, the blocking input
  flags take that to 38%, and `--brief` with `--min-rel` is the tool for
  whimsical nonsense. It cannot tell a famous tagline from a famous proverb
  (0.54): the axis measures memorable writing, not advertising.
- The input-flag patterns were written with the junk control in view, so the
  "after flags" figure is partly in-sample; the junk AUC is not.
- Long taglines are the weak spot (0.88 for 8+ words); the multi-axis variant
  no longer helps once the negatives cover short fragments.
- The kNN reference beats the axis on junk and ties it elsewhere; it is kept
  as a ceiling for what the corpus supports, not as the product, because a
  single direction is what makes the scorer explainable and cheap.

### intfloat/multilingual-e5-large · casing=none · negatives weighted per family

train 211 taglines / 298 hard / 30 generic · test 67 taglines / 55 hard / 15 generic / 67 topic-matched / 30 quotes / 45 junk

| variant | AUC vs hard | vs generic | vs topic-matched | pair win | vs quotes | vs junk | AUC long ≥8w | PASS pos / bland / junk | log-loss |
|---|---|---|---|---|---|---|---|---|---|
| surface features on raw text (not an axis) | 0.840 [0.77, 0.90] | 0.976 [0.94, 1.00] | 0.991 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.926 [0.86, 0.98] | 0.824 [0.74, 0.89] | 0.985 | – | – |
| M1 centroid | 0.648 [0.54, 0.75] | 0.931 [0.87, 0.98] | 0.997 [0.99, 1.00] | 1.000 [0.95, 1.00] | 0.938 [0.89, 0.98] | 0.448 [0.33, 0.56] | 0.426 | – | – |
| kNN-5 reference (not an axis) | 0.977 [0.95, 1.00] | 0.983 [0.96, 1.00] | 0.986 [0.97, 1.00] | 1.000 [0.95, 1.00] | 0.592 [0.48, 0.71] | 0.925 [0.88, 0.97] | 0.985 | – | – |
| M3 diff (hard only) | 0.884 [0.82, 0.94] | 0.929 [0.84, 1.00] | 0.999 [1.00, 1.00] | 1.000 [0.95, 1.00] | 0.813 [0.71, 0.90] | 0.569 [0.46, 0.68] | 0.754 | – | – |
| M2 diff (hard+generic, balanced families) | 0.935 [0.88, 0.97] | 0.974 [0.94, 1.00] | 0.994 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.819 [0.73, 0.90] | 0.722 [0.62, 0.82] | 0.841 | 0.85 / 0.06 / 0.51 (0.49 after flags) | 0.231 |
| M2 multi-axis k=2, balanced families | 0.923 [0.87, 0.97] | 0.964 [0.90, 1.00] | 0.993 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.900 [0.83, 0.95] | 0.530 [0.42, 0.64] | 0.836 | 0.93 / 0.09 / 0.84 (0.64 after flags) | 0.245 |

### intfloat/multilingual-e5-large · casing=canonical · negatives weighted per family

train 211 taglines / 298 hard / 30 generic · test 67 taglines / 55 hard / 15 generic / 67 topic-matched / 30 quotes / 45 junk

| variant | AUC vs hard | vs generic | vs topic-matched | pair win | vs quotes | vs junk | AUC long ≥8w | PASS pos / bland / junk | log-loss |
|---|---|---|---|---|---|---|---|---|---|
| surface features on raw text (not an axis) | 0.840 [0.77, 0.90] | 0.976 [0.94, 1.00] | 0.991 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.926 [0.86, 0.98] | 0.824 [0.74, 0.89] | 0.985 | – | – |
| M1 centroid | 0.437 [0.33, 0.55] | 0.853 [0.74, 0.95] | 0.864 [0.79, 0.92] | 0.851 [0.75, 0.92] | 0.530 [0.40, 0.66] | 0.357 [0.25, 0.46] | 0.084 | – | – |
| kNN-5 reference (not an axis) | 0.972 [0.94, 0.99] | 0.946 [0.89, 0.98] | 0.922 [0.88, 0.96] | 0.925 [0.84, 0.97] | 0.445 [0.34, 0.57] | 0.859 [0.78, 0.93] | 0.936 | – | – |
| M3 diff (hard only) | 0.892 [0.83, 0.94] | 0.924 [0.85, 0.98] | 0.979 [0.96, 0.99] | 1.000 [0.95, 1.00] | 0.455 [0.34, 0.57] | 0.545 [0.43, 0.65] | 0.759 | – | – |
| M2 diff (hard+generic, balanced families) | 0.954 [0.91, 0.98] | 0.988 [0.96, 1.00] | 0.918 [0.88, 0.95] | 0.940 [0.86, 0.98] | 0.536 [0.42, 0.65] | 0.776 [0.68, 0.86] | 0.883 | 0.82 / 0.08 / 0.40 (0.38 after flags) | 0.297 |
| M2 multi-axis k=2, balanced families | 0.825 [0.74, 0.89] | 0.932 [0.85, 0.99] | 0.880 [0.83, 0.93] | 0.881 [0.78, 0.94] | 0.541 [0.42, 0.66] | 0.541 [0.44, 0.65] | 0.734 | 0.73 / 0.18 / 0.73 (0.60 after flags) | 0.444 |


## Example output

`emoscript score --file lines.txt --axis axis.npz --prior 0.3 --brief "a mattress brand" --sort`
with the default model fitted by `emoscript fit` on this corpus:

```
+0.0783  p=0.92  PASS  rel=0.85  Sleep like you mean it.
+0.0024  p=0.40  fail  rel=0.82  Got Mold?
          ! possibly riffs on: "Got Milk?" (California Milk Processor Board)
+0.0765  p=0.92  PASS  rel=0.80  Just Chew It.
          ! parodies famous line: "Just Do It." (Nike)
-0.0904  p=0.02  fail  rel=0.84  Comfort, redefined for modern living.
+0.0339  p=0.69  fail  rel=0.79  1. "Just. Do. It."
          ! copies famous line: "Just Do It." (Nike); flags: wrapped; not passing: near_famous:copy
+0.0120  p=0.49  fail  rel=0.79  Sure, here is a tagline:
          ! flags: llm_chatter; not passing: llm_chatter
-0.0500  p=0.08  fail  rel=0.79  Lorem Ipsum
          ! flags: placeholder; not passing: placeholder
   n/a   skipped (no letters)  ?!
          ! flags: no_letters; not passing: no_letters
```

`emoscript eval --axis axis.npz` on the same model:

```
[final6.npz] k=1 model=intfloat/multilingual-e5-large
  AUC held-out taglines vs hard         0.954  [0.914, 0.982]  (n=55)
  AUC held-out taglines vs generic      0.988  [0.965, 1.000]  (n=15)
  AUC held-out taglines vs topic_bland  0.918  [0.876, 0.953]  (n=67)
  AUC held-out taglines vs quotes       0.536  [0.416, 0.650]  (n=30)
  AUC held-out taglines vs junk         0.776  [0.684, 0.860]  (n=45)
  topic-matched pair win rate      0.940  [0.856, 0.977]
  PASS rate at p>=0.5:             pos 0.82  hard 0.07  generic 0.00  topic_bland 0.16  quotes 0.90  junk 0.40
  ... after input flags / famous:  pos 0.82  hard 0.07  generic 0.00  topic_bland 0.16  quotes 0.87  junk 0.38
  AUC vs hard, taglines short<=4w  0.966  (n=28)
  AUC vs hard, taglines mid5-7w    0.969  (n=28)
  AUC vs hard, taglines long>=8w   0.883  (n=11)
  calibration (vs hard+generic+topic_bland): log-loss 0.297  brier 0.094  acc@0.5 0.868  threshold -0.0087
```

`emoscript fit` reported, before any test data was touched:

```
axis saved to final6.npz (k=1, dim=1024, model=intfloat/multilingual-e5-large, casing=canonical)
  out-of-fold: AUC 0.928  log-loss 0.344  acc@0.5 0.852  threshold(p=0.5) -0.0087
```

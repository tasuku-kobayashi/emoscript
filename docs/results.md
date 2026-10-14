# Results (v0.2 corpus, `intfloat/multilingual-e5-large`)

Generated from `experiments/validate_axis.py --k 1,2 --json ...` (raw text and
`--casing canonical`) with `experiments/report.py`, on the held-out splits
only. Settings (k=1, canonical text, calibrator L2=1, negative families
weighted equally) were chosen beforehand by brand-grouped nested
cross-validation on the training split (`experiments/select_settings.py`:
out-of-fold AUC 0.921, balanced log-loss 0.353); the test split was evaluated
once for this corpus version.

Columns: AUC of held-out famous taglines against each negative group with a
95% bootstrap interval (brands resampled as clusters; pairs as units for the
topic-matched rewrites), the paired win rate with its Wilson interval, the
quotability control (famous taglines vs famous non-advertising lines), junk,
AUC for the 11 held-out taglines of 8+ words, PASS rates at p ≥ 0.5 for
famous / bland (mean of hard, generic, topic-matched) / junk (and junk after
the input flags a CLI user sees), and the calibration log-loss over every
bland group. Rows marked "not an axis" are references.

## How to read it

- **The shipped setting is the canonical table, row "M2 diff (hard+generic,
  balanced families)".** The raw-text table is there to show how much of the
  separation typography provides: the six-feature surface model on raw text
  reaches 0.99 against the topic-matched rewrites and 0.93 against quotes
  without reading a word, which is why raw text is not the default.
- With canonical text the single axis separates famous taglines from bland
  ad-speak (0.94), from plain prose (0.99) and from same-topic product
  descriptions (0.93, pair win 0.97). It separates them from junk at 0.76
  (about half of the junk strings still pass on score alone; the input flags
  and the famous-line check remove a further share, and `--brief` with
  `--min-rel` is the tool for whimsical nonsense). It cannot tell a famous
  tagline from a famous proverb (0.53): the axis measures memorable writing,
  not advertising.
- Long taglines remain the weak spot (0.83 for 8+ words); the multi-axis
  variant no longer helps once the negatives cover short fragments.
- The kNN reference beats the axis on junk and ties it elsewhere; it is kept
  as a ceiling for what the corpus supports, not as the product, because a
  single direction is what makes the scorer explainable and cheap.

### intfloat/multilingual-e5-large · casing=none · negatives weighted per family

train 211 taglines / 298 hard / 30 generic · test 67 taglines / 55 hard / 15 generic / 67 topic-matched / 30 quotes / 45 junk

| variant | AUC vs hard | vs generic | vs topic-matched | pair win | vs quotes | vs junk | AUC long ≥8w | PASS pos / bland / junk | log-loss |
|---|---|---|---|---|---|---|---|---|---|
| surface features on raw text (not an axis) | 0.841 [0.77, 0.90] | 0.976 [0.94, 1.00] | 0.991 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.927 [0.87, 0.98] | 0.825 [0.74, 0.89] | 0.988 | – | – |
| M1 centroid | 0.648 [0.54, 0.75] | 0.931 [0.87, 0.98] | 0.997 [0.99, 1.00] | 1.000 [0.95, 1.00] | 0.938 [0.89, 0.98] | 0.448 [0.33, 0.56] | 0.426 | – | – |
| kNN-5 reference (not an axis) | 0.978 [0.95, 1.00] | 0.984 [0.96, 1.00] | 0.987 [0.97, 1.00] | 1.000 [0.95, 1.00] | 0.613 [0.51, 0.72] | 0.937 [0.90, 0.97] | 0.985 | – | – |
| M3 diff (hard only) | 0.884 [0.82, 0.94] | 0.929 [0.84, 1.00] | 0.999 [1.00, 1.00] | 1.000 [0.95, 1.00] | 0.815 [0.72, 0.90] | 0.572 [0.46, 0.69] | 0.755 | – | – |
| M2 diff (hard+generic, balanced families) | 0.923 [0.86, 0.97] | 0.967 [0.93, 1.00] | 0.994 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.823 [0.73, 0.90] | 0.709 [0.60, 0.81] | 0.813 | 0.85 / 0.08 / 0.56 (0.53 after flags) | 0.247 |
| M2 multi-axis k=2, balanced families | 0.911 [0.85, 0.96] | 0.961 [0.89, 1.00] | 0.994 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.900 [0.83, 0.96] | 0.511 [0.40, 0.62] | 0.803 | 0.91 / 0.09 / 0.87 (0.67 after flags) | 0.263 |

### intfloat/multilingual-e5-large · casing=canonical · negatives weighted per family

train 211 taglines / 298 hard / 30 generic · test 67 taglines / 55 hard / 15 generic / 67 topic-matched / 30 quotes / 45 junk

| variant | AUC vs hard | vs generic | vs topic-matched | pair win | vs quotes | vs junk | AUC long ≥8w | PASS pos / bland / junk | log-loss |
|---|---|---|---|---|---|---|---|---|---|
| surface features on raw text (not an axis) | 0.841 [0.77, 0.90] | 0.976 [0.94, 1.00] | 0.991 [0.98, 1.00] | 1.000 [0.95, 1.00] | 0.927 [0.87, 0.98] | 0.825 [0.74, 0.89] | 0.988 | – | – |
| M1 centroid | 0.437 [0.33, 0.55] | 0.853 [0.74, 0.95] | 0.864 [0.79, 0.92] | 0.851 [0.75, 0.92] | 0.530 [0.40, 0.66] | 0.357 [0.25, 0.46] | 0.084 | – | – |
| kNN-5 reference (not an axis) | 0.972 [0.94, 0.99] | 0.950 [0.90, 0.98] | 0.922 [0.88, 0.96] | 0.925 [0.84, 0.97] | 0.448 [0.34, 0.57] | 0.878 [0.81, 0.94] | 0.937 | – | – |
| M3 diff (hard only) | 0.891 [0.83, 0.94] | 0.925 [0.85, 0.98] | 0.978 [0.96, 0.99] | 1.000 [0.95, 1.00] | 0.458 [0.34, 0.57] | 0.552 [0.44, 0.66] | 0.760 | – | – |
| M2 diff (hard+generic, balanced families) | 0.939 [0.89, 0.97] | 0.985 [0.96, 1.00] | 0.933 [0.89, 0.97] | 0.970 [0.90, 0.99] | 0.531 [0.40, 0.65] | 0.757 [0.67, 0.84] | 0.831 | 0.85 / 0.08 / 0.47 (0.42 after flags) | 0.288 |
| M2 multi-axis k=2, balanced families | 0.807 [0.72, 0.88] | 0.931 [0.84, 0.99] | 0.893 [0.84, 0.94] | 0.925 [0.84, 0.97] | 0.546 [0.42, 0.66] | 0.531 [0.43, 0.64] | 0.676 | 0.73 / 0.18 / 0.78 (0.64 after flags) | 0.444 |


## Example output

`emoscript score --file lines.txt --axis axis.npz --prior 0.3 --brief "a mattress brand" --sort`
with the default model fitted by `emoscript fit` on this corpus:

```
+0.0892  p=0.91  PASS  rel=0.85  Sleep like you mean it.
+0.0202  p=0.45  fail  rel=0.82  Got Mold?
          ! possibly riffs on: "Got Milk?" (California Milk Processor Board)
-0.0790  p=0.02  fail  rel=0.84  Comfort, redefined for modern living.
+0.0882  p=0.91  fail  rel=0.80  Just Chew It.
          ! parodies famous line: "Just Do It." (Nike); not passing: near_famous:parody
+0.0192  p=0.44  fail  rel=0.79  Sure, here is a tagline:
          ! flags: llm_chatter; not passing: llm_chatter
-0.0332  p=0.10  fail  rel=0.79  Lorem Ipsum
          ! flags: placeholder; not passing: placeholder
   n/a   skipped (no letters)  ?!
          ! flags: no_letters; not passing: no_letters
```

`emoscript eval --axis axis.npz` on the same model:

```
[final5.npz] k=1 model=intfloat/multilingual-e5-large
  AUC held-out taglines vs hard         0.939  [0.889, 0.975]  (n=55)
  AUC held-out taglines vs generic      0.985  [0.957, 1.000]  (n=15)
  AUC held-out taglines vs topic_bland  0.933  [0.894, 0.965]  (n=67)
  AUC held-out taglines vs quotes       0.531  [0.403, 0.651]  (n=30)
  AUC held-out taglines vs junk         0.757  [0.665, 0.841]  (n=45)
  topic-matched pair win rate      0.970  [0.898, 0.992]
  PASS rate at p>=0.5:             pos 0.85  hard 0.13  generic 0.00  topic_bland 0.12  quotes 0.87  junk 0.47
  ... after input flags / famous:  pos 0.85  hard 0.13  generic 0.00  topic_bland 0.12  quotes 0.87  junk 0.42
  AUC vs hard, taglines short<=4w  0.962  (n=28)
  AUC vs hard, taglines mid5-7w    0.959  (n=28)
  AUC vs hard, taglines long>=8w   0.831  (n=11)
  calibration (vs hard+generic+topic_bland): log-loss 0.288  brier 0.089  acc@0.5 0.877  threshold +0.0031
```

`emoscript fit` reported, before any test data was touched:

```
axis saved to final5.npz (k=1, dim=1024, model=intfloat/multilingual-e5-large, casing=canonical)
  out-of-fold: AUC 0.924  log-loss 0.352  acc@0.5 0.861  threshold(p=0.5) +0.0031
```

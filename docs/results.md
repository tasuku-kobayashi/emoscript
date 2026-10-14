# Results (v0.2 corpus, `intfloat/multilingual-e5-large`)

Generated from `experiments/validate_axis.py --k 1 --json ...` (raw text and
`--casing canonical`) with `experiments/report.py`, on the held-out splits
only. Settings (k=1, canonical text, calibrator L2=1, negative families
weighted equally, sense axis on) were chosen beforehand by brand-grouped
nested cross-validation on the training split (`experiments/select_settings.py`);
the test split was evaluated once for this corpus version.

Held-out set: 173 famous taglines across brands absent from training (53 of
them from a *sister brand* of a training brand, flagged `sister_brand` and
reported separately), 85 hard + 22 generic negatives, 173 topic-matched
rewrites, 50 famous non-ad quotes, 80 junk strings, and 88 original usable
lines (`good_copy_test`) that a scorer should pass.

Columns: AUC of famous taglines against each negative group with a 95%
bootstrap interval (brands resampled as clusters; pairs as units for the
rewrites), the paired win rate with its Wilson interval, the quotability
control, junk, good copy vs junk by score (and by calibrated probability),
AUC for the 23 famous lines of 8+ words, PASS rates at p ≥ 0.5 for famous /
good / bland / junk, and the calibration log-loss over every bland group.
Rows marked "not an axis" are references.

## How to read it

- **Shipped setting: canonical table, row "M2 diff + sense axis, balanced
  families".** The raw-text table shows how much typography provides: the
  six-feature surface model on raw text reaches 0.99 against the rewrites
  without reading a word.
- **Famous vs bland:** 0.95 vs hard ad-speak (0.945 without sister brands),
  0.97 vs prose, 0.94 vs same-topic product descriptions (0.95 without sister
  brands, paired win rate 0.97). Famous 8+-word lines: 0.89.
- **Junk:** 0.79 by score. The sense axis changes no AUC by style score (it
  only enters the calibrator) but cuts the junk PASS rate from 0.42 to 0.29
  and raises good-copy-vs-junk by probability from 0.42 to 0.54. Whimsical
  nonsense ("Chase the Gravy") still tops the junk list; `--brief` with
  `--min-rel` remains the tool for it.
- **Quotes:** 0.47. The axis cannot tell a famous tagline from a famous
  proverb; it measures memorable tagline-shaped writing, not advertising.
- **The good-copy control is the honest ceiling.** Only 31% of the 88 original
  usable lines pass: 43% of those at tagline length (≤ 6 words), 12% of the
  longer benefit-led and B2B lines. Famous vs good copy is 0.83: the axis
  learned the shape of a famous tagline, and benefit-led copy sits in the
  register of the "topic-descriptive" training negatives. **Use emoscript for
  tagline-length candidates; it is not a scorer for body copy.**
- The input-flag patterns were written with the junk control in view, so the
  "after flags" figure is partly in-sample; the junk AUC is not.

### intfloat/multilingual-e5-large · casing=none · negatives weighted per family

train 211 taglines / 329 hard / 30 generic · test 173 taglines / 85 hard / 22 generic / 173 topic-matched / 50 quotes / 80 junk

| variant | AUC vs hard | vs generic | vs topic-matched | pair win | vs quotes | vs junk | good copy vs junk (by p) | AUC long ≥8w | PASS pos / good / bland / junk | log-loss |
|---|---|---|---|---|---|---|---|---|---|---|
| surface features on raw text (not an axis) | 0.829 [0.78, 0.88] | 0.939 [0.88, 0.99] | 0.990 [0.98, 1.00] | 1.000 [0.98, 1.00] | 0.914 [0.86, 0.96] | 0.796 [0.74, 0.84] | 0.263 | 0.985 | – | – |
| M1 centroid | 0.626 [0.55, 0.70] | 0.878 [0.78, 0.96] | 0.990 [0.98, 1.00] | 0.994 [0.97, 1.00] | 0.915 [0.87, 0.95] | 0.459 [0.39, 0.53] | 0.074 | 0.350 | – | – |
| kNN-5 reference (not an axis) | 0.965 [0.94, 0.98] | 0.965 [0.93, 0.99] | 0.988 [0.98, 1.00] | 0.983 [0.95, 0.99] | 0.560 [0.48, 0.64] | 0.928 [0.89, 0.96] | 0.734 | 0.984 | – | – |
| M3 diff (hard only) | 0.883 [0.84, 0.92] | 0.919 [0.84, 0.98] | 0.999 [1.00, 1.00] | 1.000 [0.98, 1.00] | 0.780 [0.71, 0.84] | 0.645 [0.57, 0.71] | 0.082 | 0.697 | – | – |
| M2 diff (hard+generic, balanced families) | 0.929 [0.90, 0.96] | 0.956 [0.91, 0.99] | 0.994 [0.99, 1.00] | 1.000 [0.98, 1.00] | 0.794 [0.72, 0.85] | 0.734 [0.66, 0.79] | 0.183 (0.183) | 0.798 | 0.85 / 0.18 / 0.10 / 0.55 (0.55 after flags) | 0.242 |
| M2 diff + sense axis, balanced families | 0.929 [0.90, 0.96] | 0.956 [0.91, 0.99] | 0.994 [0.99, 1.00] | 1.000 [0.98, 1.00] | 0.794 [0.72, 0.85] | 0.734 [0.66, 0.79] | 0.183 (0.370) | 0.798 | 0.86 / 0.22 / 0.07 / 0.38 (0.38 after flags) | 0.243 |

### intfloat/multilingual-e5-large · casing=canonical · negatives weighted per family

train 211 taglines / 329 hard / 30 generic · test 173 taglines / 85 hard / 22 generic / 173 topic-matched / 50 quotes / 80 junk

| variant | AUC vs hard | vs generic | vs topic-matched | pair win | vs quotes | vs junk | good copy vs junk (by p) | AUC long ≥8w | PASS pos / good / bland / junk | log-loss |
|---|---|---|---|---|---|---|---|---|---|---|
| surface features on raw text (not an axis) | 0.829 [0.78, 0.88] | 0.939 [0.88, 0.99] | 0.990 [0.98, 1.00] | 1.000 [0.98, 1.00] | 0.914 [0.86, 0.96] | 0.796 [0.74, 0.84] | 0.263 | 0.985 | – | – |
| M1 centroid | 0.472 [0.40, 0.54] | 0.853 [0.76, 0.93] | 0.887 [0.85, 0.92] | 0.902 [0.85, 0.94] | 0.526 [0.44, 0.62] | 0.414 [0.34, 0.49] | 0.139 | 0.076 | – | – |
| kNN-5 reference (not an axis) | 0.961 [0.94, 0.98] | 0.920 [0.87, 0.96] | 0.929 [0.90, 0.95] | 0.948 [0.90, 0.97] | 0.425 [0.34, 0.51] | 0.881 [0.84, 0.92] | 0.780 | 0.945 | – | – |
| M3 diff (hard only) | 0.895 [0.85, 0.93] | 0.922 [0.85, 0.98] | 0.986 [0.98, 0.99] | 0.988 [0.96, 1.00] | 0.378 [0.30, 0.46] | 0.659 [0.59, 0.72] | 0.231 | 0.761 | – | – |
| M2 diff (hard+generic, balanced families) | 0.952 [0.93, 0.97] | 0.971 [0.94, 1.00] | 0.942 [0.92, 0.96] | 0.965 [0.93, 0.98] | 0.469 [0.38, 0.56] | 0.786 [0.72, 0.84] | 0.418 (0.418) | 0.891 | 0.80 / 0.31 / 0.10 / 0.42 (0.41 after flags) | 0.292 |
| M2 diff + sense axis, balanced families | 0.952 [0.93, 0.97] | 0.971 [0.94, 1.00] | 0.942 [0.92, 0.96] | 0.965 [0.93, 0.98] | 0.469 [0.38, 0.56] | 0.786 [0.72, 0.84] | 0.418 (0.543) | 0.891 | 0.81 / 0.31 / 0.09 / 0.29 (0.29 after flags) | 0.311 |


## Example output

`emoscript score --file lines.txt --axis axis.npz --prior 0.3 --brief "a mattress brand" --sort`
with the default model fitted by `emoscript fit` on this corpus:

```
+0.0823  p=0.93  PASS  rel=0.85  Sleep like you mean it.
+0.0055  p=0.23  fail  rel=0.82  Got Mold?
          ! possibly riffs on: "Got Milk?" (California Milk Processor Board)
+0.0818  p=0.88  PASS  rel=0.80  Just Chew It.
          ! parodies famous line: "Just Do It." (Nike)
-0.0925  p=0.03  fail  rel=0.84  Comfort, redefined for modern living.
+0.0434  p=0.60  fail  rel=0.79  1. "Just. Do. It."
          ! copies famous line: "Just Do It." (Nike); flags: wrapped; not passing: near_famous:copy
+0.0152  p=0.46  fail  rel=0.79  Sure, here is a tagline:
          ! flags: llm_chatter; not passing: llm_chatter
-0.0483  p=0.03  fail  rel=0.79  Lorem Ipsum
          ! flags: placeholder; not passing: placeholder
   n/a   skipped (no letters)  ?!
          ! flags: no_letters; not passing: no_letters
```

`emoscript eval --axis axis.npz` on the same model:

```
[final8.npz] k=1 model=intfloat/multilingual-e5-large
  AUC held-out taglines vs hard         0.952  [0.926, 0.973]  (n=85)
  AUC held-out taglines vs generic      0.971  [0.936, 0.996]  (n=22)
  AUC held-out taglines vs topic_bland  0.942  [0.923, 0.961]  (n=173)
  AUC held-out taglines vs quotes       0.469  [0.382, 0.558]  (n=50)
  AUC held-out taglines vs junk         0.786  [0.721, 0.842]  (n=80)
  without 53 sister-brand lines (n=120): hard 0.945  generic 0.967  quotes 0.434  junk 0.762  topic_bland 0.951
  good (unknown) copy vs junk      0.418  by probability 0.543   vs hard 0.750   famous vs good 0.833
  good copy PASS by register:      tagline_length<=6w 0.43 (n=54)  longer>6w 0.12 (n=34)
  topic-matched pair win rate      0.965  [0.926, 0.984]
  PASS rate at p>=0.5:             pos 0.81  hard 0.09  generic 0.00  topic_bland 0.17  quotes 0.90  junk 0.29  good_copy 0.31
  ... after input flags / famous:  pos 0.81  hard 0.09  generic 0.00  topic_bland 0.17  quotes 0.88  junk 0.29  good_copy 0.31
  AUC vs hard, taglines short<=4w  0.958  (n=89)
  AUC vs hard, taglines mid5-7w    0.966  (n=61)
  AUC vs hard, taglines long>=8w   0.891  (n=23)
  calibration (vs hard+generic+topic_bland): log-loss 0.311  brier 0.098  acc@0.5 0.845  threshold n/a
```

`emoscript fit` reported, before any test data was touched:

```
axis saved to final8.npz (k=1, dim=1024, model=intfloat/multilingual-e5-large, casing=canonical)
  out-of-fold: AUC 0.909  log-loss 0.371  acc@0.5 0.816  threshold(p=0.5) n/a
```

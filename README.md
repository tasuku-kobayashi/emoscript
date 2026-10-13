# emoscript

**Find the direction of great ad copy.** emoscript estimates the "great tagline"
direction in sentence-embedding space from a corpus of famous advertising
slogans, then scores any new line by cosine similarity against that direction.

It is a *ranking signal*, not an oracle: it measures stylistic proximity to the
patterns that made classic taglines famous (brevity, rhythm, concrete imagery,
double meaning), not legal clearance or guaranteed market performance.

## How it works

1. Embed a curated corpus of famous taglines (e.g. *"Just Do It."*, *"Think
   Small."*, *"Got Milk?"*) with a sentence-embedding model.
2. Embed a negative corpus — crucially including **hard negatives**: fluent,
   plausible-sounding but utterly generic ad-speak (*"Quality products at
   affordable prices for everyone."*).
3. Build the axis as the difference of the two centroids:

   ```
   axis = normalize( mean(embed(great)) - mean(embed(bland)) )
   score(line) = cos(embed(line), axis)
   ```

The subtraction is the whole trick. A centroid built from great copy alone
mostly learns "well-formed marketing English" (AUC 0.92 vs held-out bland
copy in our validation). Subtracting the bland centroid isolates what is
*distinctive* about great copy: the diff-axis built against **hard negatives
alone** separates held-out famous taglines from held-out bland ad-speak
**perfectly** (AUC 1.00). Fitting against hard+generic negatives lands at
0.99 vs bland copy and 1.00 vs plain prose — its one miss is GEICO's
"15 Minutes Could Save You 15%...", a deliberately long, information-dense
slogan that breaks every brevity convention. See `experiments/validate_axis.py`
to reproduce.

## Install

```bash
pip install -e .
```

The default embedding model (`intfloat/multilingual-e5-large` via fastembed/ONNX)
downloads on first use and then runs fully offline.

## Quick start

```bash
# Fit the axis from the bundled corpus
emoscript fit --out axis.npz

# Score your own taglines (higher = more great-copy-like)
emoscript score "Think fresh." "We sell high quality furniture at low prices."
```

```python
from emoscript import embed_texts, fit_diff_axis, l2_normalize
from emoscript.dataio import load_split

pos = [r["text"] for r in load_split("slogans_train")]
neg = [r["text"] for r in load_split("negatives_hard_train")] + \
      [r["text"] for r in load_split("negatives_generic_train")]

E_pos = l2_normalize(embed_texts(pos))
E_neg = l2_normalize(embed_texts(neg))
axis = fit_diff_axis(E_pos, E_neg)

E_new = l2_normalize(embed_texts(["Melts in your mouth, not in your funnel."]))
print(float(E_new @ axis))
```

## Repository layout

```
src/emoscript/            the package (embed, axis, dataio, cli)
src/emoscript/data/       corpora as JSONL, one {"text": ...} per line
experiments/validate_axis.py   held-out evaluation (AUC + ranking)
tests/                   unit tests
```

## The corpus

| split | contents |
|---|---|
| `slogans_train` | 40 famous taglines (brand & year attributed) |
| `negatives_hard_train` | 20 original bland-but-plausible ad lines |
| `negatives_generic_train` | 10 original transactional/prose sentences |
| `*_test` | held-out evaluation splits, never used for fitting |

**Trademark notice:** famous taglines are trademarks of their respective
owners, included solely as factual, attributed reference data for research.
Contributions to the negative corpora must be your own original writing.

## Contributing

The corpus is the product. The most valuable contributions are, in order:

1. **More hard negatives** — original, fluent, forgettable ad-speak.
2. **More famous taglines** — with brand and year, from published sources.
3. **Axis variants** — clustering (multiple "flavors" of great copy), calibrated
   classifiers on top of the axis score, better embedding backends.

See `CONTRIBUTING.md` for details.

## Limitations

- A short line's embedding is influenced by topic as well as style; the axis
  inherits the vintage (mostly US, mostly 20th century) of the corpus.
- Perfect held-out AUC on ~25 lines is encouraging, not conclusive — the
  evaluation set is small and the negatives are synthetic; treat scores as one
  signal among many, and always sanity-check rankings.
- Long, informational taglines (e.g. GEICO's) score low even when famous —
  the axis currently rewards brevity and rhythm above all.
- This tool cannot tell you whether a line is *available*, *appropriate*, or
  *effective* — only whether it is stylistically close to famous taglines.

## License

MIT (code and original corpus contributions). Third-party taglines remain the
property of their respective owners; see the note in `LICENSE`.

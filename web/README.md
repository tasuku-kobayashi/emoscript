# emoscript, browser edition

Static page that scores tagline candidates entirely in the browser.

- `index.html`, `styles.css`, `app.js`: the UI; `worker.js` runs the embedding
  model (transformers.js, loaded from jsDelivr; weights from the Hugging Face hub).
- `emoscript.js`: the scorer, ported from the Python package. Keep it in step with
  `src/emoscript/text.py`, `pipeline.py` and `cli.py`.
- `model.json`: output of `emoscript export-web` (axis, calibrators, sense gate,
  famous catalogue, pinned model revision).
- `test/`: `npm test` (model-free parity with the Python fixture) and
  `npm run parity` (end to end through transformers.js).

Deployed by `.github/workflows/pages.yml` on pushes to `main` that touch this folder.

// End-to-end parity: embed the fixture lines with transformers.js, score them with
// the JavaScript port and compare against the rows `emoscript score` produced with
// fastembed (web/test/fixture.json). Needs the model from the Hugging Face hub.
//   cd web && npm install && npm run parity
// Env: EMOSCRIPT_DTYPE (fp32 | q8, default fp32), EMOSCRIPT_HF_CACHE (transformers.js cache dir),
//      EMOSCRIPT_TOL (max allowed |score difference|, default 0.01),
//      EMOSCRIPT_MODEL_JSON / EMOSCRIPT_FIXTURE (paths; default web/model.json and web/test/fixture.json).
import { readFileSync } from "node:fs";

import { env, pipeline } from "@huggingface/transformers";

import { scoreTexts, WebModel } from "../emoscript.js";

const here = new URL(".", import.meta.url);
const fixture = JSON.parse(readFileSync(process.env.EMOSCRIPT_FIXTURE || new URL("fixture.json", here), "utf8"));
const model = new WebModel(JSON.parse(readFileSync(process.env.EMOSCRIPT_MODEL_JSON || new URL("../model.json", here), "utf8")));
const dtype = process.env.EMOSCRIPT_DTYPE || "fp32";
const tol = Number(process.env.EMOSCRIPT_TOL || "0.01");
if (process.env.EMOSCRIPT_HF_CACHE) env.cacheDir = process.env.EMOSCRIPT_HF_CACHE;

if (fixture.axis_model !== model.embedding.fastembed_model) {
  throw new Error(`fixture was made with ${fixture.axis_model}, model.json with ${model.embedding.fastembed_model}`);
}
const extractor = await pipeline("feature-extraction", model.embedding.web_model, { dtype });
const embed = async (inputs) => (await extractor(inputs, { pooling: model.embedding.pooling, normalize: true })).tolist();
// the CLI drops blank lines (and strips the trailing newline only)
const texts = fixture.lines.map((l) => l.text).filter((t) => t.trim());

function compare(name, got, want) {
  const byText = new Map(want.map((r) => [r.text, r]));
  let maxScore = 0;
  let maxProb = 0;
  let maxRel = 0;
  let passMismatch = 0;
  let blockMismatch = 0;
  let orderMismatch = 0;
  for (let i = 0; i < got.length; i++) {
    const g = got[i];
    const w = byText.get(g.text);
    if (!w) throw new Error(`row missing in fixture: ${g.text}`);
    if (want[i].text !== g.text) orderMismatch++;
    if (g.score !== null) {
      maxScore = Math.max(maxScore, Math.abs(g.score - w.score));
      maxProb = Math.max(maxProb, Math.abs(g.prob - w.prob));
      if (w.brief_relevance != null) maxRel = Math.max(maxRel, Math.abs(g.brief_relevance - w.brief_relevance));
    } else if (w.score !== null) throw new Error(`skipped in JS but scored in Python: ${g.text}`);
    if (g.pass !== w.pass) {
      passMismatch++;
      console.log(`  pass differs: JS=${g.pass} py=${w.pass} p=${g.prob?.toFixed(3)}/${w.prob?.toFixed(3)} ${g.text}`);
    }
    if (JSON.stringify(g.blocked_by || null) !== JSON.stringify(w.blocked_by || null)) {
      blockMismatch++;
      console.log(`  blocked_by differs: JS=${JSON.stringify(g.blocked_by)} py=${JSON.stringify(w.blocked_by)} ${g.text}`);
    }
  }
  console.log(
    `${name}: max|Δscore|=${maxScore.toFixed(4)} max|Δprob|=${maxProb.toFixed(4)} max|Δrel|=${maxRel.toFixed(4)} ` +
      `pass mismatches=${passMismatch} blocked_by mismatches=${blockMismatch} order changes=${orderMismatch}/${got.length}`,
  );
  return { maxScore, passMismatch, blockMismatch };
}

const a = compare(`plain (${dtype})`, await scoreTexts(model, texts, embed), fixture.score_sorted);
const b = compare(
  `brief/top-rel (${dtype})`,
  await scoreTexts(model, texts, embed, { brief: fixture.brief, topRel: 0.5, prior: 0.2, threshold: 0.3 }),
  fixture.score_brief_top_rel,
);
const bad = [a, b].some((r) => r.maxScore > tol || r.blockMismatch > 0);
if (bad) {
  console.log(`PARITY FAILED (tolerance ${tol})`);
  process.exit(1);
}
console.log("parity OK");

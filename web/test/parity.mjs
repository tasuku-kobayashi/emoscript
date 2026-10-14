// End-to-end parity: embed the fixture lines with transformers.js, score them with
// the JavaScript port and compare against the rows `emoscript score` produced with
// fastembed (web/test/fixture.json). Needs the model from the Hugging Face hub.
//   cd web && npm install && npm run parity
// Env: EMOSCRIPT_DTYPE (fp32 | q8, default fp32), EMOSCRIPT_HF_CACHE (transformers.js cache dir),
//      EMOSCRIPT_TOL (max allowed |score|, |p| and |relevance| difference, default 0.002),
//      EMOSCRIPT_MODEL_JSON / EMOSCRIPT_FIXTURE (paths; default web/model.json and web/test/fixture.json).
import { readFileSync } from "node:fs";

import { env, pipeline } from "@huggingface/transformers";

import { scoreTexts, WebModel } from "../emoscript.js";

const here = new URL(".", import.meta.url);
const fixture = JSON.parse(readFileSync(process.env.EMOSCRIPT_FIXTURE || new URL("fixture.json", here), "utf8"));
const model = new WebModel(JSON.parse(readFileSync(process.env.EMOSCRIPT_MODEL_JSON || new URL("../model.json", here), "utf8")));
const dtype = process.env.EMOSCRIPT_DTYPE || "fp32";
const tol = Number(process.env.EMOSCRIPT_TOL || "0.002");
if (process.env.EMOSCRIPT_HF_CACHE) env.cacheDir = process.env.EMOSCRIPT_HF_CACHE;

if (fixture.axis_model !== model.embedding.fastembed_model) {
  throw new Error(`fixture was made with ${fixture.axis_model}, model.json with ${model.embedding.fastembed_model}`);
}
const extractor = await pipeline("feature-extraction", model.embedding.web_model, { dtype, revision: model.embedding.revision || "main" });
const embed = async (inputs) => (await extractor(inputs, { pooling: model.embedding.pooling, normalize: true })).tolist();
const texts = fixture.lines.map((l) => l.text);

function same(a, b) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

function compare(name, got, want, threshold) {
  if (got.length !== want.length) throw new Error(`${name}: JS returned ${got.length} rows, Python ${want.length}`);
  // match by (text, occurrence) so duplicate lines cannot collapse onto one Python row
  const byKey = new Map();
  const seen = new Map();
  for (const r of want) {
    const n = seen.get(r.text) || 0;
    seen.set(r.text, n + 1);
    byKey.set(`${n}\u0000${r.text}`, r);
  }
  seen.clear();
  let maxScore = 0;
  let maxProb = 0;
  let maxRel = 0;
  let maxRank = 0;
  let failures = 0;
  let orderMismatch = 0;
  const near = (g, w) => Math.abs(g - w) <= tol;
  for (let i = 0; i < got.length; i++) {
    const g = got[i];
    const n = seen.get(g.text) || 0;
    seen.set(g.text, n + 1);
    const w = byKey.get(`${n}\u0000${g.text}`);
    if (!w) throw new Error(`row missing in fixture: ${g.text}`);
    if (want[i].text !== g.text) orderMismatch++;
    const problems = [];
    if ((g.score === null) !== (w.score === null)) problems.push("skipped differs");
    if (g.score !== null && w.score !== null) {
      maxScore = Math.max(maxScore, Math.abs(g.score - w.score));
      maxProb = Math.max(maxProb, Math.abs(g.prob - w.prob));
      if (!near(g.score, w.score)) problems.push(`score ${g.score.toFixed(4)} vs ${w.score.toFixed(4)}`);
      if (!near(g.prob, w.prob)) problems.push(`prob ${g.prob.toFixed(4)} vs ${w.prob.toFixed(4)}`);
      if (!near(g.sense, w.sense)) problems.push(`sense ${g.sense.toFixed(4)} vs ${w.sense.toFixed(4)}`);
      if (w.brief_relevance != null) {
        maxRel = Math.max(maxRel, Math.abs(g.brief_relevance - w.brief_relevance));
        if (!near(g.brief_relevance, w.brief_relevance)) problems.push("brief_relevance");
      }
      if (w.rank_score != null) {
        maxRank = Math.max(maxRank, Math.abs(g.rank_score - w.rank_score));
        if (Math.abs(g.rank_score - w.rank_score) > 20 * tol) problems.push("rank_score");
      }
    }
    // a PASS verdict may legitimately flip only when p sits within tolerance of the threshold
    if (g.pass !== w.pass && !(w.prob != null && Math.abs(w.prob - threshold) <= tol)) problems.push(`pass ${g.pass} vs ${w.pass}`);
    if (!same(g.flags, w.flags)) problems.push(`flags ${JSON.stringify(g.flags)} vs ${JSON.stringify(w.flags)}`);
    if (!same(g.near_famous, w.near_famous)) problems.push("near_famous");
    // blocked_by may differ only in off_brief / sense_gate when the underlying value sits at the cut
    const gb = g.blocked_by || [];
    const wb = w.blocked_by || [];
    if (!same(gb, wb)) {
      const soft = new Set(["off_brief", "sense_gate"]);
      const diff = [...gb.filter((x) => !wb.includes(x)), ...wb.filter((x) => !gb.includes(x))];
      if (!diff.every((x) => soft.has(x))) problems.push(`blocked_by ${JSON.stringify(gb)} vs ${JSON.stringify(wb)}`);
      else console.log(`  note: ${JSON.stringify(gb)} vs ${JSON.stringify(wb)} at the cut: ${g.text}`);
    }
    if (problems.length) {
      failures++;
      console.log(`  MISMATCH ${JSON.stringify(g.text)}: ${problems.join("; ")}`);
    }
  }
  console.log(
    `${name}: max|Δscore|=${maxScore.toFixed(4)} max|Δprob|=${maxProb.toFixed(4)} max|Δrel|=${maxRel.toFixed(4)} ` +
      `max|Δrank|=${maxRank.toFixed(4)} mismatching rows=${failures} order changes=${orderMismatch}/${got.length}`,
  );
  return failures;
}

const a = compare(`plain (${dtype})`, await scoreTexts(model, texts, embed), fixture.score_sorted, 0.5);
const b = compare(
  `brief/top-rel (${dtype})`,
  await scoreTexts(model, texts, embed, { brief: fixture.brief, topRel: 0.5, prior: 0.2, threshold: 0.3 }),
  fixture.score_brief_top_rel,
  0.3,
);
if (a + b > 0) {
  console.log(`PARITY FAILED (tolerance ${tol})`);
  process.exit(1);
}
console.log("parity OK");

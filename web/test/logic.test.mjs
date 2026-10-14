// Model-free checks of the JavaScript port against the Python fixture.
//   cd web && npm test
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { canonicalize, inputFlags, nearFamous, prepareFamous, WebModel } from "../emoscript.js";

const here = new URL(".", import.meta.url);
const fixture = JSON.parse(readFileSync(new URL("fixture.json", here), "utf8"));
const model = new WebModel(JSON.parse(readFileSync(new URL("../model.json", here), "utf8")));

test("canonical form matches Python for every fixture line", () => {
  for (const line of fixture.lines) assert.equal(canonicalize(line.text), line.canonical, JSON.stringify(line.text));
});

test("input flags match Python for every fixture line", () => {
  for (const line of fixture.lines) assert.deepEqual(inputFlags(line.text), line.flags, JSON.stringify(line.text));
});

test("near-famous match agrees with Python for every fixture line", () => {
  const near = nearFamous(
    fixture.lines.map((l) => l.text),
    model.famous,
  );
  fixture.lines.forEach((line, i) => {
    const got = near[i];
    const want = line.near_famous;
    if (want === null) {
      assert.equal(got, null, JSON.stringify(line.text));
      return;
    }
    assert.ok(got, JSON.stringify(line.text));
    assert.equal(got.text, want.text, JSON.stringify(line.text));
    assert.equal(got.brand, want.brand);
    assert.equal(got.edits, want.edits);
    assert.equal(got.severity, want.severity);
    assert.ok(Math.abs(got.trigram - want.trigram) < 1e-9);
  });
});

test("famous catalogue prepared from model.json is non-empty", () => {
  assert.ok(prepareFamous(model.blob.famous).length > 100);
  assert.equal(model.casing, fixture.casing);
});

// ---- scoreTexts with a deterministic fake embedder (no model): row logic the fixture
// cannot reach through the real model (all-blocked batches, single rows, NaN, duplicates).
import { scoreTexts } from "../emoscript.js";

const dim = model.embedding.dim;
function fakeEmbed(inputs) {
  return inputs.map((t) => {
    // hash-seeded direction plus a push along the style axis for lines containing "great"
    let h = 2166136261;
    for (const ch of t) h = Math.imul(h ^ ch.codePointAt(0), 16777619) >>> 0;
    const v = new Array(dim).fill(0).map((_, i) => Math.sin(h * 0.001 + i) * 0.1);
    if (t.includes("great")) for (let i = 0; i < dim; i++) v[i] += model.axes[0][i] * 0.6 + model.senseAxis[i] * 0.6;
    return v;
  });
}

test("scoreTexts: skipped rows carry null fields, go last, and never pass", async () => {
  const rows = await scoreTexts(model, ["great line", "!!!", "great again"], fakeEmbed);
  const skipped = rows[rows.length - 1];
  assert.equal(skipped.text, "!!!");
  assert.deepEqual([skipped.score, skipped.prob, skipped.sense], [null, null, null]);
  assert.equal(skipped.pass, false);
  assert.deepEqual(skipped.blocked_by, ["no_letters"]);
  assert.deepEqual(skipped.flags, ["no_letters"]);
});

test("scoreTexts: blocking flags and famous copies block in sorted order, parodies do not", async () => {
  const rows = await scoreTexts(model, ["Just Do It.", "Sure! Here are five taglines:", "{{tagline}}", "Just chew it"], fakeEmbed);
  const by = Object.fromEntries(rows.map((r) => [r.text, r]));
  assert.deepEqual(by["Just Do It."].blocked_by.filter((x) => x !== "sense_gate"), ["near_famous:copy"]);
  assert.equal(by["Just Do It."].pass, false);
  assert.deepEqual(by["Sure! Here are five taglines:"].blocked_by.filter((x) => x !== "sense_gate"), ["llm_chatter"]);
  assert.deepEqual(by["{{tagline}}"].blocked_by.filter((x) => x !== "sense_gate"), ["placeholder", "single_word"]);
  assert.equal(by["Just chew it"].near_famous.severity, "parody");
  assert.ok(!(by["Just chew it"].blocked_by || []).includes("near_famous:copy"));
  // blocked rows sort after every unblocked row
  const firstBlocked = rows.findIndex((r) => r.blocked_by);
  assert.ok(rows.slice(firstBlocked).every((r) => r.blocked_by));
});

test("scoreTexts: single row and all-blocked batches with a brief do not divide by zero", async () => {
  const one = await scoreTexts(model, ["great single line"], fakeEmbed, { brief: "a brief", topRel: 0.5 });
  assert.equal(one.length, 1);
  assert.ok(Number.isFinite(one[0].rank_score));
  const blocked = await scoreTexts(model, ["{{a}}", "TODO"], fakeEmbed, { brief: "a brief", minRel: 0.9 });
  assert.ok(blocked.every((r) => r.blocked_by && r.blocked_by.includes("off_brief")));
  assert.ok(blocked.every((r) => Number.isFinite(r.rank_score)));
});

test("scoreTexts: topRel keeps the given share by relevance and prior shifts p monotonically", async () => {
  const texts = Array.from({ length: 10 }, (_, i) => `great candidate number ${i}`);
  const rows = await scoreTexts(model, texts, fakeEmbed, { brief: "a brief", topRel: 0.5, senseGate: false });
  const off = rows.filter((r) => (r.blocked_by || []).includes("off_brief")).length;
  assert.equal(off, 5);
  const p5 = (await scoreTexts(model, texts, fakeEmbed, { senseGate: false })).map((r) => r.prob);
  const p1 = (await scoreTexts(model, texts, fakeEmbed, { senseGate: false, prior: 0.1 })).map((r) => r.prob);
  p5.forEach((p, i) => assert.ok(p1[i] < p));
});

test("scoreTexts: duplicates are scored independently and non-finite embeddings are rejected", async () => {
  const rows = await scoreTexts(model, ["great twice", "great twice"], fakeEmbed);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].score, rows[1].score);
  await assert.rejects(() => scoreTexts(model, ["x y"], (inputs) => inputs.map(() => new Array(dim).fill(NaN))), /not finite/);
  await assert.rejects(() => scoreTexts(model, ["x y"], fakeEmbed, { threshold: 1 }), /threshold/);
  await assert.rejects(() => scoreTexts(model, ["x y"], fakeEmbed, { minRel: 0.1 }), /brief/);
});

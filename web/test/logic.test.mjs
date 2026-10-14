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

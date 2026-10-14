// UI glue: model.json + web worker (transformers.js) + the ported scorer.
import { scoreTexts, WebModel } from "./emoscript.js";

const $ = (id) => document.getElementById(id);
const status = $("status");
const progress = $("progress");
const scoreBtn = $("score");
const jsonBtn = $("download-json");
const csvBtn = $("download-csv");
const BATCH = 32;

const EXAMPLE = `Just chew it.
Sleep like you mean it.
Chase the gravy.
Quality products at affordable prices for everyone.
Premium Quality
Add to Cart
Athletic shoes designed for serious runners.
Breakfast that gets out of bed before you do.
Sure! Here are five taglines for your brand:
{{tagline}}
Because mornings don't wait.
Crunch now, conquer later.`;

let model = null;
let worker = null;
let ready = false;
let lastRows = null;
let pending = new Map();
let nextId = 1;

function setStatus(text, error = false) {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function fmtMB(n) {
  return (n / 1048576).toFixed(0);
}

function startWorker() {
  if (worker) worker.terminate();
  pending = new Map();
  ready = false;
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "progress") {
      progress.hidden = false;
      if (m.total) {
        progress.value = (100 * m.loaded) / m.total;
        setStatus(`Downloading model… ${fmtMB(m.loaded)} / ${fmtMB(m.total)} MB (cached by your browser after the first time)`);
      }
    } else if (m.type === "stage") {
      if (m.status === "fallback") setStatus(m.file);
    } else if (m.type === "ready") {
      ready = true;
      progress.hidden = true;
      scoreBtn.disabled = false;
      setStatus(`Ready: ${model.embedding.web_model} (${m.dtype}, ${m.device}). Everything runs locally.`);
    } else if (m.type === "embedded") {
      const p = pending.get(m.id);
      if (p) {
        pending.delete(m.id);
        p.resolve(m.vectors);
      }
    } else if (m.type === "error") {
      if (m.id && pending.has(m.id)) {
        const p = pending.get(m.id);
        pending.delete(m.id);
        p.reject(new Error(m.message));
      } else {
        progress.hidden = true;
        scoreBtn.disabled = true;
        setStatus(`Could not load the model: ${m.message}`, true);
      }
    }
  };
  worker.onerror = (e) => setStatus(`Worker error: ${e.message}`, true);
  setStatus("Loading model…");
  worker.postMessage({ type: "load", model: model.embedding.web_model, dtype: $("dtype").value });
}

function embedBatch(inputs) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "embed", id, inputs, pooling: model.embedding.pooling });
  });
}

async function embed(inputs) {
  const out = [];
  for (let i = 0; i < inputs.length; i += BATCH) {
    setStatus(`Embedding ${Math.min(i + BATCH, inputs.length)} / ${inputs.length}…`);
    out.push(...(await embedBatch(inputs.slice(i, i + BATCH))));
  }
  return out;
}

function readOptions() {
  const brief = $("brief").value.trim();
  const topRel = Number($("top-rel").value);
  const minRelRaw = $("min-rel").value.trim();
  return {
    threshold: Number($("threshold").value),
    prior: Number($("prior").value),
    brief: brief ? brief : null,
    topRel: brief && topRel > 0 && topRel < 1 ? topRel : null,
    minRel: brief && minRelRaw !== "" ? Number(minRelRaw) : null,
    senseGate: $("sense-gate").checked,
    famousCheck: $("famous-check").checked,
  };
}

const NOTE_LABEL = { copy: "copies famous line", parody: "parodies famous line", possible: "possibly riffs on" };

function notesFor(r) {
  const notes = [];
  if (r.near_famous) {
    const nf = r.near_famous;
    notes.push(`${NOTE_LABEL[nf.severity]}: “${nf.text}”${nf.brand ? ` (${nf.brand})` : ""}`);
  }
  if (r.flags) notes.push(`flags: ${r.flags.join(", ")}`);
  if (r.blocked_by) notes.push(`not passing: ${r.blocked_by.join(", ")}`);
  if (r.skipped) notes.push(`skipped (${r.skipped})`);
  return notes.join("; ");
}

function render(rows, withBrief) {
  const tbody = $("results").querySelector("tbody");
  tbody.innerHTML = "";
  document.querySelectorAll("#results .rel").forEach((el) => (el.hidden = !withBrief));
  const fmt = (x, d = 2) => (x == null ? "" : x.toFixed(d));
  rows.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.className = r.pass ? "pass" : r.blocked_by ? "fail blocked" : "fail";
    const cells = [
      ["num", String(i + 1)],
      ["verdict", r.score == null ? "n/a" : r.pass ? "PASS" : "fail"],
      ["num", fmt(r.prob)],
      ["num", r.score == null ? "" : (r.score >= 0 ? "+" : "") + r.score.toFixed(4)],
      ["num", fmt(r.sense)],
      ["num rel", fmt(r.brief_relevance)],
      ["line", r.text],
      ["notes", notesFor(r)],
    ];
    for (const [cls, text] of cells) {
      const td = document.createElement("td");
      td.className = cls;
      td.textContent = text;
      if (cls.includes("rel")) td.hidden = !withBrief;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  });
  const passed = rows.filter((r) => r.pass).length;
  $("summary").textContent = `${passed} of ${rows.length} pass`;
  $("results-panel").hidden = false;
}

async function runScore() {
  if (!ready) return;
  const texts = $("lines")
    .value.split(/\r?\n/)
    .filter((t) => t.trim());
  if (!texts.length) {
    setStatus("Paste at least one line.", true);
    return;
  }
  let opts;
  try {
    opts = readOptions();
  } catch (err) {
    setStatus(err.message, true);
    return;
  }
  scoreBtn.disabled = true;
  try {
    const t0 = performance.now();
    const rows = await scoreTexts(model, texts, embed, opts);
    lastRows = rows;
    render(rows, opts.brief !== null);
    jsonBtn.disabled = csvBtn.disabled = false;
    setStatus(`Scored ${rows.length} lines in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
  } catch (err) {
    setStatus(`Scoring failed: ${err.message}`, true);
  } finally {
    scoreBtn.disabled = false;
  }
}

function download(name, type, content) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function toCsv(rows) {
  const cols = ["rank", "pass", "prob", "score", "sense", "brief_relevance", "rank_score", "text", "flags", "blocked_by", "near_famous"];
  const esc = (v) => {
    if (v == null) return "";
    const s = Array.isArray(v) ? v.join("|") : typeof v === "object" ? `${v.severity}: ${v.text}` : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.join(",")];
  rows.forEach((r, i) => lines.push(cols.map((c) => esc(c === "rank" ? i + 1 : r[c])).join(",")));
  return lines.join("\n") + "\n";
}

async function main() {
  try {
    const res = await fetch("model.json");
    if (!res.ok) throw new Error(`model.json: HTTP ${res.status}`);
    model = new WebModel(await res.json());
  } catch (err) {
    setStatus(`Could not load model.json: ${err.message}`, true);
    return;
  }
  $("model-name").textContent = model.embedding.fastembed_model;
  $("model-version").textContent = `emoscript ${model.blob.emoscript_version}, k=${model.k}, dim=${model.embedding.dim}`;
  startWorker();

  scoreBtn.addEventListener("click", runScore);
  $("lines").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runScore();
  });
  $("example").addEventListener("click", () => {
    $("lines").value = EXAMPLE;
    $("brief").value = "a breakfast cereal for busy families";
  });
  $("dtype").addEventListener("change", () => {
    scoreBtn.disabled = true;
    startWorker();
  });
  jsonBtn.addEventListener("click", () => {
    if (lastRows) download("emoscript-scores.json", "application/json", JSON.stringify(lastRows, null, 2));
  });
  csvBtn.addEventListener("click", () => {
    if (lastRows) download("emoscript-scores.csv", "text/csv", toCsv(lastRows));
  });
}

main();

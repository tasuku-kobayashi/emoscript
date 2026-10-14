// UI glue: model.json + web worker (transformers.js) + the ported scorer.
import { scoreTexts, WebModel } from "./emoscript.js";

const $ = (id) => document.getElementById(id);
const status = $("status");
const progress = $("progress");
const scoreBtn = $("score");
const jsonBtn = $("download-json");
const csvBtn = $("download-csv");
const BATCH = 32;
const MAX_LINE_CHARS = 2000;
const MAX_LINES = 5000;

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
let lastResult = null; // {rows, options, dtype}
let pending = new Map();
let nextId = 1;
let run = 0; // token of the scoring run in flight (0 = none)
let lastNotice = "";

function setStatus(text, error = false) {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function fmtMB(n) {
  return (n / 1048576).toFixed(0);
}

function rejectPending(message) {
  for (const p of pending.values()) p.reject(new Error(message));
  pending = new Map();
}

function startWorker() {
  if (worker) worker.terminate();
  rejectPending("the model was reloaded");
  ready = false;
  lastNotice = "";
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "progress") {
      progress.hidden = false;
      if (m.total) {
        progress.value = (100 * m.loaded) / m.total;
        setStatus(
          `Downloading model… ${fmtMB(m.loaded)} / ${fmtMB(m.total)} MB (cached by your browser after the first time)` +
            (lastNotice ? ` — ${lastNotice}` : ""),
        );
      }
    } else if (m.type === "stage") {
      if (m.status === "fallback") {
        lastNotice = m.file;
        setStatus(m.file);
      }
    } else if (m.type === "ready") {
      ready = true;
      progress.hidden = true;
      scoreBtn.disabled = run !== 0;
      const fp16 = $("dtype").querySelector('option[value="fp16"]');
      fp16.hidden = !m.webgpu;
      fp16.disabled = !m.webgpu;
      setStatus(
        `Ready: ${model.embedding.web_model} (${m.dtype}, ${m.device}).` +
          (lastNotice ? ` ${lastNotice}.` : "") +
          " Your lines are embedded here in the browser.",
      );
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
  worker.onerror = (e) => {
    const why = e.message || "the scoring worker failed to load (is cdn.jsdelivr.net reachable?)";
    rejectPending(why);
    progress.hidden = true;
    scoreBtn.disabled = true;
    setStatus(`Worker error: ${why}`, true);
  };
  setStatus("Loading model…");
  worker.postMessage({
    type: "load",
    model: model.embedding.web_model,
    revision: model.embedding.revision || "main",
    dtype: $("dtype").value,
  });
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
  const threshold = Number($("threshold").value);
  const prior = Number($("prior").value);
  if (!(threshold > 0 && threshold < 1)) throw new Error("PASS threshold must be strictly between 0 and 1.");
  if (!(prior > 0 && prior < 1)) throw new Error("Prior must be strictly between 0 and 1.");
  if (!(topRel > 0 && topRel <= 1)) throw new Error("Keep top share must be a fraction in (0, 1]; 1 keeps every line.");
  if (minRelRaw !== "" && !brief) throw new Error("Minimum brief relevance needs a brief.");
  if (minRelRaw !== "" && !Number.isFinite(Number(minRelRaw))) throw new Error("Minimum brief relevance must be a number.");
  return {
    threshold,
    prior,
    brief: brief ? brief : null,
    topRel: brief && topRel < 1 ? topRel : null,
    minRel: brief && minRelRaw !== "" ? Number(minRelRaw) : null,
    senseGate: $("sense-gate").checked,
    famousCheck: $("famous-check").checked,
  };
}

function markStale() {
  if (!lastResult) return;
  $("summary").textContent += " (input changed: score again)";
  jsonBtn.disabled = csvBtn.disabled = true;
  lastResult = null;
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

function render(rows, withBrief, calibrated) {
  const tbody = $("results").querySelector("tbody");
  tbody.innerHTML = "";
  document.querySelectorAll("#results .rel").forEach((el) => (el.hidden = !withBrief));
  const fmt = (x, d = 2) => (x == null ? "" : x.toFixed(d));
  rows.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.className = r.pass ? "pass" : r.blocked_by ? "fail blocked" : "fail";
    const cells = [
      ["num", String(i + 1)],
      ["verdict", r.score == null ? "n/a" : !calibrated ? "" : r.pass ? "PASS" : "fail"],
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
  $("summary").textContent = calibrated ? `${passed} of ${rows.length} pass` : `${rows.length} lines (uncalibrated axis: scores only)`;
  $("results-panel").hidden = false;
}

async function runScore() {
  if (!ready || run !== 0) return;
  const texts = $("lines")
    .value.split(/\r?\n/)
    .filter((t) => t.trim());
  if (!texts.length) {
    setStatus("Paste at least one line.", true);
    return;
  }
  if (texts.length > MAX_LINES) {
    setStatus(`That is ${texts.length} lines; the page handles up to ${MAX_LINES} at a time.`, true);
    return;
  }
  const long = texts.find((t) => t.length > MAX_LINE_CHARS);
  if (long) {
    setStatus(`A line is ${long.length} characters long; the limit is ${MAX_LINE_CHARS}. This scorer is for tagline-length lines.`, true);
    return;
  }
  let opts;
  try {
    opts = readOptions();
  } catch (err) {
    setStatus(err.message, true);
    return;
  }
  const token = ++run;
  const dtype = $("dtype").value;
  scoreBtn.disabled = true;
  $("dtype").disabled = true;
  try {
    const t0 = performance.now();
    const rows = await scoreTexts(model, texts, embed, opts);
    if (token !== run) return; // superseded by a reload
    lastResult = { rows, options: opts, dtype };
    render(rows, opts.brief !== null, Boolean(model.calibrator));
    jsonBtn.disabled = csvBtn.disabled = false;
    setStatus(`Scored ${rows.length} lines in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
  } catch (err) {
    if (token === run) setStatus(`Scoring failed: ${err.message}`, true);
  } finally {
    if (token === run) {
      run = 0;
      scoreBtn.disabled = !ready;
      $("dtype").disabled = false;
    }
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
    if (typeof v === "number" || typeof v === "boolean") return String(v);
    let s = Array.isArray(v) ? v.join("|") : typeof v === "object" ? `${v.severity}: ${v.text}` : String(v);
    // pasted lines can start like a spreadsheet formula ("=HYPERLINK(...)", "@SUM"): neutralise them
    if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.join(",")];
  rows.forEach((r, i) => lines.push(cols.map((c) => esc(c === "rank" ? i + 1 : r[c])).join(",")));
  return "\ufeff" + lines.join("\r\n") + "\r\n";
}

function exportBlob() {
  const { rows, options, dtype } = lastResult;
  return {
    emoscript_version: model.blob.emoscript_version,
    embedding: { ...model.embedding, dtype },
    options,
    rows,
  };
}

async function main() {
  try {
    const res = await fetch("model.json", { cache: "no-cache" });
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
    run = 0;
    markStale();
    startWorker();
  });
  for (const id of ["lines", "brief", "threshold", "prior", "top-rel", "min-rel", "sense-gate", "famous-check"]) {
    $(id).addEventListener("input", markStale);
  }
  jsonBtn.addEventListener("click", () => {
    if (lastResult) download("emoscript-scores.json", "application/json", JSON.stringify(exportBlob(), null, 2));
  });
  csvBtn.addEventListener("click", () => {
    if (lastResult) download("emoscript-scores.csv", "text/csv;charset=utf-8", toCsv(lastResult.rows));
  });
}

main();

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
          `Downloading the model once… ${fmtMB(m.loaded)} / ${fmtMB(m.total)} MB` +
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
      setStatus(`Ready (${m.dtype}, ${m.device}).` + (lastNotice ? ` ${lastNotice}.` : ""));
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
  return {
    threshold: Number($("threshold").value),
    prior: Number($("prior").value),
    brief: brief ? brief : null,
    topRel: brief && topRel < 1 ? topRel : null,
    minRel: null,
    senseGate: $("sense-gate").checked,
    famousCheck: $("famous-check").checked,
  };
}

function markStale() {
  if (!lastResult) return;
  $("summary").textContent += " · input changed, score again";
  jsonBtn.disabled = csvBtn.disabled = true;
  lastResult = null;
}

const NEAR_LABEL = { copy: "copies", parody: "parodies", possible: "riffs on" };
const BLOCK_REASON = {
  "near_famous:copy": "copies a famous line",
  sense_gate: "does not read as a message",
  off_brief: "off brief",
  placeholder: "placeholder text",
  llm_chatter: "chat-bot text, not a line",
  single_word: "one word",
  too_long: "too long for a tagline",
  no_letters: "no words",
};

// State of a row -> (css class, word). Color never stands alone: the word says it too.
function verdictOf(r, threshold) {
  if (r.score === null) return ["off", "No words"];
  if (r.blocked_by) {
    const why = r.blocked_by.map((b) => BLOCK_REASON[b] || b);
    return ["off", "Out: " + why[0]];
  }
  if (r.pass) return ["pass", "Passes"];
  if (r.prob >= Math.min(0.3, threshold)) return ["close", "Close"];
  return ["no", "Ordinary"];
}

function ring(p) {
  const R = 30;
  const C = 2 * Math.PI * R;
  const frac = p == null ? 0 : Math.max(0, Math.min(1, p));
  const wrap = document.createElement("div");
  wrap.className = "ring";
  wrap.innerHTML =
    `<svg viewBox="0 0 72 72" aria-hidden="true"><circle class="track" cx="36" cy="36" r="${R}"/>` +
    `<circle class="fill" cx="36" cy="36" r="${R}" stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="${C.toFixed(2)}"/></svg>` +
    `<div class="pct"></div>`;
  wrap.querySelector(".pct").textContent = p == null ? "–" : `${Math.round(frac * 100)}%`;
  requestAnimationFrame(() => (wrap.querySelector(".fill").style.strokeDashoffset = (C * (1 - frac)).toFixed(2)));
  return wrap;
}

function detailsFor(r, withBrief) {
  const dl = document.createElement("dl");
  const add = (k, v) => {
    if (v == null || v === "") return;
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  };
  const f = (x, d = 2) => (x == null ? null : x.toFixed(d));
  if (r.blocked_by) add("why out", r.blocked_by.map((b) => BLOCK_REASON[b] || b).join("; "));
  if (r.near_famous) {
    const nf = r.near_famous;
    add("famous line", `${NEAR_LABEL[nf.severity]} “${nf.text}”${nf.brand ? ` (${nf.brand})` : ""}`);
  }
  add("p (great copy)", f(r.prob));
  add("score (cosine)", r.score == null ? null : (r.score >= 0 ? "+" : "") + r.score.toFixed(4));
  add("sense (is a message)", f(r.sense));
  if (withBrief) add("relevance to brief", f(r.brief_relevance));
  if (withBrief) add("rank (style + relevance)", f(r.rank_score));
  if (r.flags) add("flags", r.flags.join(", "));
  return dl;
}

function render(rows, opts, calibrated) {
  const list = $("cards");
  list.innerHTML = "";
  const withBrief = opts.brief !== null;
  for (const r of rows) {
    const li = document.createElement("li");
    const [cls, word] = calibrated ? verdictOf(r, opts.threshold) : [r.score === null ? "off" : "close", ""];
    li.className = `card ${cls}`;
    li.appendChild(ring(calibrated ? r.prob : null));
    const text = document.createElement("div");
    text.className = "text";
    const line = document.createElement("p");
    line.className = "line";
    line.textContent = r.text;
    const verdict = document.createElement("div");
    verdict.className = "verdict";
    const dot = document.createElement("span");
    dot.className = "dot";
    verdict.append(dot, document.createTextNode(word));
    if (withBrief && r.brief_relevance != null) {
      const why = document.createElement("span");
      why.className = "why";
      why.textContent = ` · relevance ${Math.round(r.brief_relevance * 100)}%`;
      verdict.appendChild(why);
    }
    text.append(line, verdict);
    const details = document.createElement("div");
    details.className = "details";
    details.appendChild(detailsFor(r, withBrief));
    li.append(text, details);
    li.addEventListener("click", () => li.classList.toggle("open"));
    list.appendChild(li);
  }
  const passed = rows.filter((r) => r.pass).length;
  const sum = $("summary");
  sum.innerHTML = "";
  const b = document.createElement("strong");
  b.textContent = calibrated ? `${passed} of ${rows.length} pass` : `${rows.length} lines`;
  sum.append(b, document.createTextNode(calibrated ? " · ranked best first · tap a line for details" : " · uncalibrated axis"));
  $("results").hidden = false;
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
    render(rows, opts, Boolean(model.calibrator));
    jsonBtn.disabled = csvBtn.disabled = false;
    setStatus(`${rows.length} lines in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
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
  $("model-version").textContent = `${model.blob.emoscript_version} · ${model.embedding.fastembed_model}`;
  startWorker();

  scoreBtn.addEventListener("click", runScore);
  $("lines").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runScore();
  });
  $("example").addEventListener("click", () => {
    $("lines").value = EXAMPLE;
    $("brief").value = "a breakfast cereal for busy families";
    $("settings").open = false;
    markStale();
    runScore();
  });
  $("dtype").addEventListener("change", () => {
    scoreBtn.disabled = true;
    run = 0;
    markStale();
    startWorker();
  });
  for (const id of ["lines", "brief", "threshold", "prior", "top-rel", "sense-gate", "famous-check"]) {
    $(id).addEventListener("input", markStale);
  }
  const pct = (v) => `${Math.round(Number(v) * 100)}%`;
  const showVals = () => {
    $("threshold-val").textContent = pct($("threshold").value);
    $("prior-val").textContent = pct($("prior").value);
    $("top-rel-val").textContent = Number($("top-rel").value) >= 1 ? "all" : `top ${pct($("top-rel").value)}`;
  };
  for (const id of ["threshold", "prior", "top-rel"]) $(id).addEventListener("input", showVals);
  showVals();
  // one popover at a time; click outside closes
  const pops = Array.from(document.querySelectorAll("details.pop"));
  for (const d of pops) d.addEventListener("toggle", () => d.open && pops.forEach((o) => o !== d && (o.open = false)));
  document.addEventListener("click", (e) => {
    if (!e.target.closest("details.pop")) pops.forEach((o) => (o.open = false));
  });
  jsonBtn.addEventListener("click", () => {
    if (lastResult) download("emoscript-scores.json", "application/json", JSON.stringify(exportBlob(), null, 2));
  });
  csvBtn.addEventListener("click", () => {
    if (lastResult) download("emoscript-scores.csv", "text/csv;charset=utf-8", toCsv(lastResult.rows));
  });
}

main();

// emoscript scoring logic, ported from the Python package (src/emoscript/text.py,
// pipeline.py, cli.py). Pure functions, no DOM, no model: the embedding step is
// injected so the same code runs in the browser app and in the Node parity test.
//
// Keep this file in step with the Python source; web/test/parity.mjs checks that
// canonical forms, flags, near-famous matches and scores agree.

const QUOTE_MAP = { "’": "'", "‘": "'", "“": '"', "”": '"', "´": "'", "`": "'" };
const RE_TRAILING = /[\s.!?…'")\]]+$/;
const RE_LEADING = /^[\s'"(\[]+/;
const RE_WS = /\s+/g;
const RE_HTML = /<\/?[a-z][a-z0-9]*(?:\s[^<>]{0,80})?\/?>/gi;
const RE_MD_LEAD = /^(?:[#>]+\s*)+/;
const RE_MD_PAIR = /(\*\*|__|~~|\*|_)(\S(?:.*?\S)?)\1/g;
const RE_MD_EDGE = /^[*_~]+|[*_~]+$/g;
const RE_LIST = /^\s*(?:[-•▪·]|\(?\d{1,3}[.)]|[a-z]\))\s+/i;
const RE_LABEL = new RegExp(
  "^\\s*(?:tagline|slogan|headline|option|variant|idea|line|candidate|copy|v\\d+)" +
    "(?:\\s+(?:idea|option|variant|line|candidate))?" +
    "(?:\\s*(?:\\d+|[a-z]|#\\d+))?(?:\\s*\\([^)]{0,40}\\))?(?:\\s*[:：]|\\s+[-–—])\\s*",
  "i",
);
const NONWORD = "[^\\p{L}\\p{N}_]"; // Python's unicode-aware \W
const RE_PLACEHOLDER = new RegExp(
  "\\{\\{.*?\\}\\}|\\$\\{.*?\\}|<[A-Z][A-Z _-]{1,30}>|\\[(?:insert|your|tagline|brand|slogan)[^\\]]*\\]" +
    "|^\\s*todo\\s*:?\\s*$|^\\s*todo:\\s*(?:write|add|insert|tagline|slogan|headline|copy)\\b" +
    "|^\\s*tbd\\s*:?\\s*$|^\\s*tbd:\\s*(?:tagline|slogan|headline|copy)\\b|lorem ipsum" +
    `|^${NONWORD}*(?:(?:your|the|a|an|new|catchy|best|insert) )?` +
    `(?:tagline|slogan|headline|copy|text|title|name|brand|logo|content|description|body)s? goes here${NONWORD}*$` +
    "|^\\s*(?:insert|add|enter) \\w+ here\\s*[.!]?$" +
    "|^\\s*your (?:tagline|slogan|text|copy|headline|logo|name|brand|message|title) here\\s*[.!]?$" +
    "|^\\s*(?:undefined|null|nan|none|n/?a|\\[object object\\])\\s*$",
  "iu",
);
const COPY_WORDS = "(?:tagline|slogan|headline|option|idea|cop(?:y|ies)|suggestion|version)s?";
const RE_LLM_CHATTER = new RegExp(
  "^(?:sure|certainly|okay|of course|absolutely|got it)(?: thing)?[!,.:]\\s+(?:here|i|below|these)\\b" +
    `(?:.*\\b${COPY_WORDS}\\b|.*:\\s*$)` +
    `|^here(?:'s| is| are) (?:\\w+ ){0,4}${COPY_WORDS}\\b` +
    "|^here you go[:!]" +
    "|^as an? (?:ai|language model)\\b" +
    "|^i(?:'m| am) (?:sorry|afraid),? but i can(?:'t|not)\\b" +
    "|^sorry,? (?:but )?i can(?:'t|not) (?:write|generate|create|provide|do that|help with)\\b" +
    "|^i (?:can(?:'t|not)|am unable to) (?:write|generate|create|provide|come up with) " +
    "(?:a |an |the |any |that|this|slogan|tagline)" +
    "|^error[:\\s]+(?:\\d{3}\\s*$|rate limit|loading|message|code)",
  "i",
);
const RE_LETTER = /\p{L}/u;
const RE_LATIN = /\p{Script=Latin}/u;

export const CASINGS = ["none", "lower", "sentence", "lower_nopunct", "canonical"];
export const MAX_WORDS = 17;
export const BLOCKING_FLAGS = new Set(["no_letters", "single_word", "too_long", "placeholder", "llm_chatter"]);

export function normalizeQuotes(text) {
  return text.replace(/[’‘“”´`]/g, (c) => QUOTE_MAP[c]);
}

function collapseWs(text) {
  return text.replace(RE_WS, " ").trim();
}

export function stripTerminalPunctuation(text) {
  let t = collapseWs(normalizeQuotes(text));
  t = t.replace(RE_LEADING, "");
  t = t.replace(RE_TRAILING, "");
  return t.trim();
}

export function stripWrappers(text) {
  let t = normalizeQuotes(text).replace(RE_HTML, " ");
  t = collapseWs(t);
  for (let i = 0; i < 3; i++) {
    let t2 = t.replace(RE_MD_LEAD, "");
    t2 = t2.replace(RE_MD_PAIR, "$2");
    t2 = t2.replace(RE_MD_EDGE, "").trim();
    t2 = t2.replace(RE_LIST, "").replace(RE_LABEL, "").trim();
    t2 = t2.replace(RE_LEADING, "");
    if (t2 === t) break;
    t = t2;
  }
  return t;
}

export function canonicalize(text) {
  return stripTerminalPunctuation(stripWrappers(text).toLowerCase());
}

export function wordTokens(text) {
  return normalizeQuotes(text).replace(/'/g, "").toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
}

export function normalizeText(text, casing = "none") {
  switch (casing) {
    case "none":
      return text;
    case "lower":
      return text.toLowerCase();
    case "lower_nopunct":
      return stripTerminalPunctuation(text.toLowerCase());
    case "canonical":
      return canonicalize(text);
    case "sentence": {
      const t = text.toLowerCase();
      for (let i = 0; i < t.length; i++) {
        if (RE_LETTER.test(t[i])) return t.slice(0, i) + t[i].toUpperCase() + t.slice(i + 1);
      }
      return t;
    }
    default:
      throw new Error(`casing must be one of ${CASINGS.join(", ")}, got ${casing}`);
  }
}

export function hasLetters(text) {
  return RE_LETTER.test(text);
}

export function inputFlags(text, maxWords = MAX_WORDS) {
  const flags = [];
  if (!hasLetters(text)) return ["no_letters"];
  const raw = normalizeQuotes(text);
  const stripped = stripWrappers(raw);
  const words = stripped.split(/\s+/).filter((w) => w && RE_LETTER.test(w));
  if (words.length <= 1) flags.push("single_word");
  if (words.length > maxWords) flags.push("too_long");
  if (RE_PLACEHOLDER.test(raw) || RE_PLACEHOLDER.test(stripped)) flags.push("placeholder");
  if (RE_LLM_CHATTER.test(stripped.trim())) flags.push("llm_chatter");
  const letters = Array.from(stripped).filter((ch) => RE_LETTER.test(ch));
  if (letters.length && letters.filter((ch) => RE_LATIN.test(ch)).length / letters.length < 0.9) {
    flags.push("non_latin");
  }
  if (stripped !== collapseWs(raw)) flags.push("wrapped");
  return flags;
}

// ---------------------------------------------------------------- near-famous check
function trigrams(s) {
  s = s.replace(/ /g, "_");
  const out = new Set();
  if (s.length >= 3) for (let i = 0; i <= s.length - 3; i++) out.add(s.slice(i, i + 3));
  else if (s) out.add(s);
  return out;
}

function jaccard(a, b) {
  if (!a.size && !b.size) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
}

function tokenEditDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)));
    }
    prev = cur;
  }
  return prev[b.length];
}

const SEVERITY_RANK = { copy: 0, parody: 1, possible: 2 };

export function prepareFamous(famous) {
  return famous.map((f) => {
    const toks = wordTokens(canonicalize(f.text));
    return { text: f.text, brand: f.brand || "", toks, tri: trigrams(toks.join(" ")) };
  });
}

/** Per text: {text, brand, edits, trigram, severity} of the closest famous line, or null. */
export function nearFamous(texts, famousPrepared, { maxEdits = 2, trigram = 0.5 } = {}) {
  return texts.map((t) => {
    const ct = wordTokens(canonicalize(t));
    const ctri = trigrams(ct.join(" "));
    let best = null;
    for (const f of famousPrepared) {
      const ft = f.toks;
      if (!ft.length || !ct.length) continue;
      let edits;
      let jac;
      if (Math.abs(ft.length - ct.length) > maxEdits) {
        if (ft.length < 3 || ct.length < 3) continue;
        jac = jaccard(ctri, f.tri);
        if (jac < trigram) continue;
        edits = tokenEditDistance(ct, ft);
      } else {
        edits = tokenEditDistance(ct, ft);
        jac = jaccard(ctri, f.tri);
      }
      const allowed = Math.min(maxEdits, Math.floor((ft.length - 1) / 2));
      let severity;
      if (edits === 0) severity = "copy";
      else if (edits <= allowed || (ft.length >= 3 && ct.length >= 3 && jac >= trigram)) severity = "parody";
      else if (ft.length === 2 && ct.length === 2 && edits === 1 && ct[0] === ft[0]) severity = "possible";
      else continue;
      const rank = [SEVERITY_RANK[severity], edits, -jac];
      if (best === null || rankLess(rank, best.rank)) {
        best = { text: f.text, brand: f.brand, edits, trigram: jac, severity, rank };
      }
    }
    if (best) delete best.rank;
    return best;
  });
}

function rankLess(a, b) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] < b[i];
  }
  return false;
}

// ---------------------------------------------------------------- scoring
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function l2Normalize(v) {
  const n = Math.sqrt(dot(v, v)) || 1;
  return Array.from(v, (x) => x / n);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function calibratorProba(c, features) {
  let z = c.b;
  for (let i = 0; i < c.w.length; i++) z += ((features[i] - c.mu[i]) / c.sd[i]) * c.w[i];
  return sigmoid(z);
}

export function shiftPrior(p, prior) {
  if (prior === 0.5) return p;
  if (!(prior > 0 && prior < 1)) throw new Error("prior must be strictly between 0 and 1");
  const eps = 1e-12;
  const pc = Math.min(Math.max(p, eps), 1 - eps);
  const logit = Math.log(pc / (1 - pc)) + Math.log(prior / (1 - prior));
  return sigmoid(logit);
}

export class WebModel {
  constructor(blob) {
    if (blob.format !== "emoscript-web/1") throw new Error(`unknown model format ${blob.format}`);
    this.blob = blob;
    this.axes = blob.axes;
    this.casing = blob.casing;
    this.calibrator = blob.calibrator;
    this.senseAxis = blob.sense_axis;
    this.senseGate = blob.sense_gate;
    this.embedding = blob.embedding;
    this.famous = prepareFamous(blob.famous || []);
  }

  get k() {
    return this.axes.length;
  }

  /** Text exactly as it goes into the embedding model (normalized, prefixed). */
  modelInput(text) {
    return this.embedding.prefix + normalizeText(text, this.casing);
  }

  scorePerAxis(e) {
    return this.axes.map((a) => dot(e, a));
  }

  score(e) {
    return Math.max(...this.scorePerAxis(e));
  }

  senseScore(e) {
    return this.senseAxis ? dot(e, this.senseAxis) : null;
  }

  proba(e, prior = 0.5) {
    if (!this.calibrator) return null;
    const feats = this.senseAxis ? [this.score(e), this.senseScore(e)] : [this.score(e)];
    return shiftPrior(calibratorProba(this.calibrator, feats), prior);
  }

  senseProba(e) {
    if (!this.senseAxis || !this.senseGate) return null;
    return calibratorProba(this.senseGate, [this.senseScore(e)]);
  }
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function std(xs) {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) * (x - m))));
}

function quantile(xs, q) {
  // numpy default (linear interpolation)
  const s = [...xs].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/**
 * Score a batch exactly like `emoscript score --sort`. `embed(texts)` must return
 * one L2-normalized vector per *model input* (already normalized + prefixed).
 */
export async function scoreTexts(
  model,
  texts,
  embed,
  { threshold = 0.5, prior = 0.5, brief = null, minRel = null, topRel = null, senseGate = true, famousCheck = true } = {},
) {
  if (!(threshold > 0 && threshold < 1)) throw new Error("threshold must be strictly between 0 and 1");
  if (!(prior > 0 && prior < 1)) throw new Error("prior must be strictly between 0 and 1");
  if (brief !== null && !hasLetters(brief)) throw new Error("brief must contain words");
  if (minRel !== null && brief === null) throw new Error("minRel needs a brief");
  if (topRel !== null && (brief === null || !(topRel > 0 && topRel <= 1))) throw new Error("topRel needs a brief and a fraction in (0, 1]");

  const ok = texts.map(hasLetters);
  const inputs = texts.filter((_, i) => ok[i]).map((t) => model.modelInput(t));
  if (brief !== null) inputs.push(model.modelInput(brief));
  const vectors = inputs.length ? (await embed(inputs)).map(l2Normalize) : [];
  const B = brief !== null ? vectors.pop() : null;

  const near = famousCheck ? nearFamous(texts, model.famous) : texts.map(() => null);
  const rows = [];
  let j = 0;
  for (let i = 0; i < texts.length; i++) {
    const t = texts[i];
    const flags = inputFlags(t);
    const row = { text: t, score: null };
    if (!ok[i]) {
      row.skipped = "no letters";
      if (brief !== null) row.brief_relevance = null;
      if (model.senseGate) row.sense = null;
    } else {
      const e = vectors[j++];
      row.score = model.score(e);
      if (brief !== null) row.brief_relevance = dot(e, B);
      if (model.senseGate) row.sense = model.senseProba(e);
      row._prob = model.proba(e, prior);
      row._gate = !senseGate || !model.senseGate || row.sense >= 0.5;
      if (model.k > 1) row.axis = model.scorePerAxis(e).indexOf(row.score);
    }
    if (flags.length) row.flags = flags;
    if (near[i]) row.near_famous = near[i];
    row._blocked = flags.filter((f) => BLOCKING_FLAGS.has(f)).sort();
    if (near[i] && near[i].severity === "copy") row._blocked.push("near_famous:copy");
    rows.push(row);
  }
  let relCut = null;
  if (brief !== null && topRel !== null) {
    const valid = rows.filter((r) => r.score !== null).map((r) => r.brief_relevance);
    if (valid.length) relCut = quantile(valid, 1 - topRel);
  }
  for (const r of rows) {
    const skipped = r.score === null;
    if (!skipped && brief !== null && minRel !== null && r.brief_relevance < minRel) r._blocked.push("off_brief");
    if (!skipped && relCut !== null && r.brief_relevance < relCut) r._blocked.push("off_brief");
    if (!skipped && !r._gate) r._blocked.push("sense_gate");
    if (model.calibrator) {
      r.prob = skipped ? null : r._prob;
      r.pass = !skipped && r.prob >= threshold && r._blocked.length === 0;
    }
    if (r._blocked.length) r.blocked_by = r._blocked;
    delete r._blocked;
    delete r._prob;
    delete r._gate;
  }
  // sort: with a brief, by style AND relevance z-scored over the unblocked rows of this batch
  if (brief !== null) {
    const clean = rows.filter((r) => r.score !== null && !r.blocked_by);
    const s = clean.map((r) => r.score);
    const rel = clean.map((r) => r.brief_relevance);
    const [muS, sdS] = s.length > 1 ? [mean(s), std(s)] : [0, 1];
    const [muR, sdR] = rel.length > 1 ? [mean(rel), std(rel)] : [0, 1];
    for (const r of rows) {
      if (r.score !== null && r.brief_relevance != null) {
        r.rank_score = (r.score - muS) / (sdS + 1e-12) + (r.brief_relevance - muR) / (sdR + 1e-12);
      }
    }
  }
  const keyName = brief !== null ? "rank_score" : "score";
  const key = (r) => [
    r.blocked_by || r.score === null ? 1 : 0,
    -(r[keyName] ?? -Infinity),
    -(r.score ?? -Infinity),
  ];
  return rows
    .map((r, i) => ({ r, i, k: key(r) }))
    .sort((a, b) => {
      for (let n = 0; n < 3; n++) if (a.k[n] !== b.k[n]) return a.k[n] < b.k[n] ? -1 : 1;
      return a.i - b.i;
    })
    .map((x) => x.r);
}

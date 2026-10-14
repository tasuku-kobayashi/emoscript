// Web worker: loads the transformers.js feature-extraction pipeline and embeds
// batches off the main thread. Messages:
//   {type: "load", model, revision, dtype} -> progress events, then {type: "ready", device, dtype, webgpu}
//   {type: "embed", id, inputs, pooling}   -> {type: "embedded", id, vectors} (L2-normalized rows)
// Errors are reported as {type: "error", id?, message}.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;

let extractor = null;
let loaded = null; // {model, revision, dtype, device}

async function hasWebGPU() {
  if (!("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function load(model, revision, dtype, forceDevice = null) {
  if (extractor) {
    await extractor.dispose();
    extractor = null;
  }
  const webgpu = await hasWebGPU();
  if (dtype === "fp16" && !webgpu) throw new Error("fp16 needs WebGPU; this browser has none. Choose fp32 or q8.");
  const device = forceDevice || (webgpu ? "webgpu" : "wasm");
  const seen = new Map();
  const progress_callback = (p) => {
    if (p.status === "progress" && p.file) {
      seen.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
      let done = 0;
      let total = 0;
      for (const v of seen.values()) {
        done += v.loaded;
        total += v.total;
      }
      self.postMessage({ type: "progress", file: p.file, loaded: done, total });
    } else if (p.status === "initiate" || p.status === "done" || p.status === "ready") {
      self.postMessage({ type: "stage", status: p.status, file: p.file || "" });
    }
  };
  const opts = { dtype, device, revision, progress_callback };
  try {
    extractor = await pipeline("feature-extraction", model, opts);
  } catch (err) {
    if (device !== "wasm" && dtype !== "fp16") {
      self.postMessage({ type: "stage", status: "fallback", file: `WebGPU failed (${err.message}); using WASM` });
      extractor = await pipeline("feature-extraction", model, { ...opts, device: "wasm" });
      loaded = { model, revision, dtype, device: "wasm" };
      self.postMessage({ type: "ready", device: "wasm", dtype, webgpu });
      return;
    }
    throw err;
  }
  loaded = { model, revision, dtype, device };
  self.postMessage({ type: "ready", device, dtype, webgpu });
}

async function embed(inputs, pooling) {
  const out = (await extractor(inputs, { pooling, normalize: true })).tolist();
  for (const row of out) {
    for (const x of row) if (!Number.isFinite(x)) throw new Error("the model returned a non-finite embedding");
  }
  return out;
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") {
      await load(msg.model, msg.revision, msg.dtype);
    } else if (msg.type === "embed") {
      if (!extractor) throw new Error("model not loaded");
      let vectors;
      try {
        vectors = await embed(msg.inputs, msg.pooling);
      } catch (err) {
        // a WebGPU device can be lost or reject an op at run time: reload on WASM once and retry
        if (loaded.device !== "webgpu" || loaded.dtype === "fp16") throw err;
        self.postMessage({ type: "stage", status: "fallback", file: `WebGPU failed while embedding (${err.message}); using WASM` });
        await load(loaded.model, loaded.revision, loaded.dtype, "wasm");
        vectors = await embed(msg.inputs, msg.pooling);
      }
      self.postMessage({ type: "embedded", id: msg.id, vectors });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String(err && err.message ? err.message : err) });
  }
};

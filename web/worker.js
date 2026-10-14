// Web worker: loads the transformers.js feature-extraction pipeline and embeds
// batches off the main thread. Messages:
//   {type: "load", model, dtype}         -> progress events, then {type: "ready", device, dtype}
//   {type: "embed", id, inputs, pooling} -> {type: "embedded", id, vectors} (L2-normalized rows)
// Errors are reported as {type: "error", id?, message}.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;

let extractor = null;
let loaded = null; // {model, dtype}

async function hasWebGPU() {
  if (!("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function load(model, dtype) {
  if (extractor && loaded && loaded.model === model && loaded.dtype === dtype) return;
  if (extractor) {
    await extractor.dispose();
    extractor = null;
  }
  const webgpu = await hasWebGPU();
  if (dtype === "fp16" && !webgpu) throw new Error("fp16 needs WebGPU; this browser has none. Choose fp32 or q8.");
  const device = webgpu ? "webgpu" : "wasm";
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
  try {
    extractor = await pipeline("feature-extraction", model, { dtype, device, progress_callback });
  } catch (err) {
    if (device !== "wasm" && dtype !== "fp16") {
      self.postMessage({ type: "stage", status: "fallback", file: `WebGPU failed (${err.message}); using WASM` });
      extractor = await pipeline("feature-extraction", model, { dtype, device: "wasm", progress_callback });
      loaded = { model, dtype };
      self.postMessage({ type: "ready", device: "wasm", dtype });
      return;
    }
    throw err;
  }
  loaded = { model, dtype };
  self.postMessage({ type: "ready", device, dtype });
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") {
      await load(msg.model, msg.dtype);
    } else if (msg.type === "embed") {
      if (!extractor) throw new Error("model not loaded");
      const out = await extractor(msg.inputs, { pooling: msg.pooling, normalize: true });
      self.postMessage({ type: "embedded", id: msg.id, vectors: out.tolist() });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String(err && err.message ? err.message : err) });
  }
};

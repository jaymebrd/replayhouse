import { Store } from "./replayhouse.js";
import { initBench } from "./bench.js";

export const el = (id) => document.getElementById(id);
export const fmtMs = (x) => x >= 1000 ? `${(x / 1000).toFixed(2)}s` : `${Math.round(x)}ms`;

async function resolveBase() {
  try {
    const r = await fetch("./engine/index.js", { method: "HEAD" });
    if (r.ok) return "./engine";
  } catch {}
  return "./node_modules/chdb-wasm/dist";
}

el("load").onclick = async () => {
  el("load").disabled = true;
  el("loadmsg").textContent = "resolving bundle…";
  // coi-serviceworker installs on first visit and reloads the page to take effect;
  // on very old browsers (or file://) it can't establish isolation at all — the mt
  // bundle needs SharedArrayBuffer + crossOriginIsolated, so bail out clearly here
  // rather than letting selectBundle silently fall back / fail deep inside AsyncChdb.
  if (!self.crossOriginIsolated) {
    el("loadmsg").textContent =
      "this page needs cross-origin isolation for the multi-threaded engine — serve it over HTTP";
    el("load").disabled = false;
    return;
  }
  try {
    const base = await resolveBase();
    const { AsyncChdb, selectBundle } = await import(`${base}/index.js`);
    const bundle = selectBundle({ baseUrl: base });
    if (!bundle.supported) {
      el("loadmsg").textContent = bundle.reasons.join("; ");
      el("load").disabled = false;
      return;
    }
    el("dl").hidden = false;
    // worker-side dynamic import resolves relative URLs against the worker script, not the page — absolutize here.
    const moduleUrl = new URL(bundle.moduleUrl, location.href).href;
    const wasmUrl = bundle.wasmUrl ? new URL(bundle.wasmUrl, location.href).href : undefined;
    const db = await AsyncChdb.create({
      moduleUrl, wasmUrl,
      onProgress: (l, t) => {
        const pct = (l / t) * 100;
        if (Number.isFinite(pct)) el("dl").value = pct;
      },
    });
    el("dl").hidden = true;
    el("loadmsg").textContent = "engine ready — a full ClickHouse is now running on this page";
    const conn = await db.connect();
    const store = await Store.open(conn);
    // the wasm pthread pool is finite; defaults (max_threads = hardwareConcurrency,
    // pread_threadpool async reads) exhaust it under concurrent queries and everything
    // starts failing with CANNOT_SCHEDULE_TASK. Cap query threads and read synchronously.
    await store._exec(
      "SET max_threads = 4, max_insert_threads = 1, local_filesystem_read_method = 'pread'");
    for (const s of ["act-scale", "act-data", "act-learn"])
      el(s).setAttribute("aria-disabled", "false");
    initBench({ store, conn });
    // A failed act module must not kill the others — but it must say so, not
    // leave a live-looking dead section.
    await import("./data.js").then((m) => m.initData({ db, store }), (err) => {
      console.error(err);
      el("datamsg").textContent = `this act failed to load: ${err?.message ?? err}`;
    });
    await import("./learn.js").then((m) => m.initLearn({ store }), (err) => {
      console.error(err);
      el("frame").textContent = `this act failed to load: ${err?.message ?? err}`;
    });
  } catch (err) {
    el("loadmsg").textContent = `failed to load the engine: ${err?.message ?? err}`;
    el("load").disabled = false;
    el("dl").hidden = true;
  }
};

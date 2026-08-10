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
    const db = await AsyncChdb.create({
      moduleUrl: bundle.moduleUrl, wasmUrl: bundle.wasmUrl,
      onProgress: (l, t) => { el("dl").value = (l / t) * 100; },
    });
    el("dl").hidden = true;
    el("loadmsg").textContent = "engine ready — a full ClickHouse is now running on this page";
    const conn = await db.connect();
    const store = await Store.open(conn);
    for (const s of ["act-scale", "act-data", "act-learn"])
      el(s).setAttribute("aria-disabled", "false");
    initBench({ store, conn });
    const { initData } = await import("./data.js").catch(() => ({ initData: null }));
    if (initData) initData({ db, store });
    const { initLearn } = await import("./learn.js").catch(() => ({ initLearn: null }));
    if (initLearn) initLearn({ store });
  } catch (err) {
    el("loadmsg").textContent = `failed to load the engine: ${err?.message ?? err}`;
    el("load").disabled = false;
    el("dl").hidden = true;
  }
};

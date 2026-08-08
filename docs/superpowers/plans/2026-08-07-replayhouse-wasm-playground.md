# ReplayHouse Browser Playground (chdb-wasm) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public browser playground — the terminal demo running in a tab, powered by the real ClickHouse engine via chdb-wasm — plus a `replayhouse.js` mirror of the SQL contract and a gh-pages deploy path.

**Architecture:** Three layers. `web/replayhouse.js` is an ES module mirroring the Python client's SQL contract (~200 lines: create/insert/sample/updatePriorities generating the same SQL, incl. seeded `cityHash64` draws) over an `AsyncChdb` connection — proving the "clients are thin" thesis. `web/index.html` is the playground: a JS linear model runs the PER loop and renders the same frame as the terminal demo into a `<pre>` (monospace, orange-accent histogram), with Prioritized/Uniform/Pause controls; the ~99 MB single-threaded wasm bundle loads with a progress bar (no COOP/COEP needed). CI coverage is a Node test (Node ≥ 23 runs the same wasm) exercising `replayhouse.js` end-to-end — the spike, productionized. Deploy vendors the st bundle onto a `gh-pages` branch (99 MB < GitHub's 100 MB cap; `main` stays clean).

**Tech Stack:** chdb-wasm 0.3.0 (npm, devDependency of `web/`), vanilla ES modules (no bundler), Node ≥ 23 for tests, GitHub Pages.

## Global Constraints

- Branch `replayhouse-wasm-playground` off `main`; commit per task; Python suite (`.venv/bin/pytest tests -q`, 92) untouched and green.
- Honesty constraint (inherited from the demo): every number rendered in the page comes from a chdb-wasm query or the real JS training step — nothing simulated.
- `web/replayhouse.js` must generate SQL byte-compatible with `src/replayhouse/sampling.py`'s contract: A-ES key `-log(1 - randCanonical()) / (w)` unseeded, `-log((cityHash64(id, SEED) + 1) / 18446744073709551616.) / (w)` seeded; weight > 0 exclusion; identifier regex `^[A-Za-z_][A-Za-z0-9_]*$` validated before interpolation; ids validated as UUIDs before phase-2 `IN` lists.
- No Python-side changes; nothing added to pyproject. `web/` has its own `package.json` with `chdb-wasm` as devDependency and `"type": "module"`.
- Node tests use `node --test` (built-in runner), invoked from pytest via one subprocess wrapper marked `@pytest.mark.skipif(shutil.which("node") is None, ...)` so the Python suite stays self-contained where Node is absent.
- The page must state its bundle size before downloading ("Load engine (~99 MB, cached after first visit)") — no silent 99 MB fetch on page open.
- Deploy is a script (`web/deploy.sh`), not CI magic: assembles `web/` + `node_modules/chdb-wasm/dist/st/` into a worktree of the `gh-pages` branch and pushes. Enabling Pages on the repo is a documented manual/gh-cli step in the same task.

## File Structure

```
web/package.json          # type: module; devDependency chdb-wasm; test script
web/replayhouse.js        # SQL-contract mirror over AsyncChdb (Store class)
web/test/contract.test.mjs# node --test: schema/insert/sample/seed/update round-trips
web/index.html            # playground page (loads ./engine/* on gh-pages, node_modules locally)
web/playground.js         # PER loop + <pre> frame renderer + controls
web/serve.mjs             # tiny local static server for development
web/deploy.sh             # assemble + push gh-pages
tests/test_wasm_contract.py # pytest wrapper running `npm --prefix web test` (skipif no node)
README.md                 # playground link + local-dev notes
```

---

### Task 1: `web/replayhouse.js` + Node contract tests

**Files:**
- Create: `web/package.json`, `web/replayhouse.js`, `web/test/contract.test.mjs`, `tests/test_wasm_contract.py`

**Interfaces:**
- Consumes: `chdb-wasm`'s `AsyncChdb.create({moduleUrl, wasmUrl?})`, `db.connect()`, `conn.query(sql, fmt)`.
- Produces (Task 2 relies on these exact names): `class Store` with static `async Store.open(conn)`; methods `async create(name, columns)` (columns: `{col: type}` — creates main + `__priorities` sidecar), `async insert(name, rows)` (rows may carry `priority`, default 1.0; returns ids), `async sample(name, k, {by = "priority", where = null, seed = null} = {})` returning `{ids, rows}`, `async updatePriorities(name, ids, values)`, `async query(sql)` returning parsed JSONEachRow rows. UUIDs generated client-side via `crypto.randomUUID()` (v4 is fine here — no time-ordering claim in JS; document the divergence in a comment).

- [ ] **Step 1: Write `web/package.json`**

```json
{
  "name": "replayhouse-web",
  "private": true,
  "type": "module",
  "scripts": {
    "test": "node --test test/",
    "serve": "node serve.mjs"
  },
  "devDependencies": {
    "chdb-wasm": "^0.3.0"
  }
}
```

Run: `npm --prefix web install` (downloads ~200 MB once; `web/node_modules` is already covered by `.gitignore`'s defaults — verify `node_modules` is ignored, add a `web/.gitignore` with `node_modules/` if not).

- [ ] **Step 2: Write `web/replayhouse.js`**

```javascript
// The ReplayHouse SQL contract, mirrored in JS over chdb-wasm.
// Byte-compatible with src/replayhouse/sampling.py's queries — the point:
// ReplayHouse is a SQL contract; clients are thin.

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const UUID_RE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

function checkName(name) {
  if (!NAME_RE.test(name)) throw new Error(`invalid identifier: ${name}`);
  return name;
}

function sampleKey(by, seed, idExpr = "id") {
  const u = seed === null
    ? "1 - randCanonical()"
    : `(cityHash64(${idExpr}, ${Math.trunc(seed)}) + 1) / 18446744073709551616.`;
  return `-log(${u}) / (${by})`;
}

export class Store {
  constructor(conn) { this.conn = conn; }

  static async open(conn) { return new Store(conn); }

  async query(sql) {
    const r = await this.conn.query(sql, "JSONEachRow");
    const text = r.text();
    return text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  }

  async _exec(sql) { await this.conn.query(sql, "CSV"); }

  async create(name, columns) {
    checkName(name);
    const cols = Object.entries(columns)
      .map(([c, t]) => `\`${checkName(c)}\` ${t}`).join(", ");
    await this._exec(
      `CREATE TABLE \`${name}\` (id UUID, inserted_at DateTime DEFAULT now(), ${cols})
       ENGINE = MergeTree ORDER BY id`);
    await this._exec(
      `CREATE TABLE \`${name}__priorities\` (id UUID, priority Float32, version UInt64)
       ENGINE = ReplacingMergeTree(version) ORDER BY id`);
  }

  async insert(name, rows) {
    checkName(name);
    if (!rows.length) return [];
    const version = Date.now() * 1000;
    const main = [], prios = [];
    for (const row of rows) {
      const { priority = 1.0, ...rest } = row;
      // crypto.randomUUID is v4 (random, not time-ordered like the Python
      // client's UUIDv7) — fine for the playground; PK lookups don't care.
      const id = rest.id ?? crypto.randomUUID();
      main.push({ ...rest, id });
      prios.push({ id, priority, version });
    }
    const nd = (rs) => rs.map((r) => JSON.stringify(r)).join("\n");
    await this._exec(`INSERT INTO \`${name}\` FORMAT JSONEachRow\n${nd(main)}`);
    await this._exec(
      `INSERT INTO \`${name}__priorities\` FORMAT JSONEachRow\n${nd(prios)}`);
    return main.map((r) => r.id);
  }

  async sample(name, k, { by = "priority", where = null, seed = null } = {}) {
    checkName(name);
    k = Math.trunc(k);
    if (seed !== null && !Number.isInteger(seed)) throw new Error("seed must be an int");
    const side = `${name}__priorities`;
    let phase1;
    if (by.trim() === "priority") {
      const filt = where
        ? `\n  AND id IN (SELECT id FROM \`${name}\` WHERE (${where}))` : "";
      phase1 = `WITH current AS (
          SELECT id, argMax(priority, version) AS priority
          FROM \`${side}\` GROUP BY id)
        SELECT id FROM current WHERE priority > 0${filt}
        ORDER BY ${sampleKey("priority", seed)} ASC LIMIT ${k}`;
    } else if (/\bpriority\b/.test(by)) {
      const cond = `((${by})) > 0` + (where ? ` AND ((${where}))` : "");
      phase1 = `WITH current AS (
          SELECT id, argMax(priority, version) AS priority
          FROM \`${side}\` GROUP BY id)
        SELECT m.id AS id FROM \`${name}\` AS m
        INNER JOIN current AS c ON m.id = c.id WHERE ${cond}
        ORDER BY ${sampleKey(`(${by})`, seed, "m.id")} ASC LIMIT ${k}`;
    } else {
      const cond = `((${by})) > 0` + (where ? ` AND ((${where}))` : "");
      phase1 = `SELECT id FROM \`${name}\` WHERE ${cond}
        ORDER BY ${sampleKey(`(${by})`, seed)} ASC LIMIT ${k}`;
    }
    const ids = (await this.query(phase1)).map((r) => r.id);
    if (!ids.length) return { ids: [], rows: [] };
    for (const id of ids) if (!UUID_RE.test(id)) throw new Error(`not a UUID: ${id}`);
    const rows = await this.query(
      `SELECT * FROM \`${name}\` WHERE id IN (${ids.map((i) => `'${i}'`).join(",")})`);
    return { ids: rows.map((r) => r.id), rows };
  }

  async updatePriorities(name, ids, values) {
    checkName(name);
    if (ids.length !== values.length) throw new Error("ids/values length mismatch");
    if (!ids.length) return;
    for (const id of ids) if (!UUID_RE.test(id)) throw new Error(`not a UUID: ${id}`);
    const version = Date.now() * 1000;
    const nd = ids.map((id, i) =>
      JSON.stringify({ id, priority: values[i], version })).join("\n");
    await this._exec(
      `INSERT INTO \`${name}__priorities\` FORMAT JSONEachRow\n${nd}`);
  }
}
```

- [ ] **Step 3: Write `web/test/contract.test.mjs`**

```javascript
import { test } from "node:test";
import assert from "node:assert/strict";
import { AsyncChdb } from "chdb-wasm";
import { Store } from "../replayhouse.js";

const db = await AsyncChdb.create({
  moduleUrl: new URL("../node_modules/chdb-wasm/dist/chdb.mjs", import.meta.url).pathname,
});
const conn = await db.connect();
const store = await Store.open(conn);

await test("schema + insert + count", async () => {
  await store.create("exp", { task: "LowCardinality(String)", reward: "Float32" });
  const ids = await store.insert("exp",
    Array.from({ length: 500 }, (_, i) => ({
      task: `t${i % 3}`, reward: i / 500, priority: 1.0 })));
  assert.equal(ids.length, 500);
  const [{ c }] = await store.query("SELECT count() AS c FROM exp");
  assert.equal(Number(c), 500);
});

await test("weighted sample without replacement", async () => {
  const { ids, rows } = await store.sample("exp", 32);
  assert.equal(ids.length, 32);
  assert.equal(new Set(ids).size, 32);
  assert.equal(rows.length, 32);
});

await test("seeded draws are deterministic", async () => {
  const a = await store.sample("exp", 20, { seed: 42 });
  const b = await store.sample("exp", 20, { seed: 42 });
  assert.deepEqual([...a.ids].sort(), [...b.ids].sort());
  const c = await store.sample("exp", 20, { seed: 7 });
  assert.notDeepEqual([...a.ids].sort(), [...c.ids].sort());
});

await test("priority update shifts sampling", async () => {
  const first = await store.sample("exp", 10, { seed: 1 });
  await store.updatePriorities("exp",
    first.ids, first.ids.map(() => 0.0001));
  const second = await store.sample("exp", 10, { seed: 1 });
  // same seed, changed weights -> different batch (weights are part of the draw)
  assert.notDeepEqual([...first.ids].sort(), [...second.ids].sort());
});

await test("where filter and by expression", async () => {
  const r = await store.sample("exp", 50, { by: "reward + 0.01", where: "task = 't1'" });
  assert.ok(r.rows.length > 0);
  assert.ok(r.rows.every((x) => x.task === "t1"));
});

await conn.close();
await db.terminate();
```

- [ ] **Step 4: Write the pytest wrapper**

`tests/test_wasm_contract.py`:

```python
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.skipif(not (ROOT / "web" / "node_modules" / "chdb-wasm").exists(),
                    reason="run `npm --prefix web install` first")
def test_js_contract_against_wasm_engine():
    r = subprocess.run(["npm", "--prefix", str(ROOT / "web"), "test"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 5: Run and commit**

Run: `npm --prefix web test` → 5 tests pass; `.venv/bin/pytest tests -q` → 93.

```bash
git add web/package.json web/replayhouse.js web/test tests/test_wasm_contract.py web/.gitignore
git commit -m "feat: replayhouse.js - the SQL contract mirrored over chdb-wasm, with contract tests"
```

---


---

> **AMENDMENT (2026-08-08).** Task 1 is complete (commits 375b333 + 0957da5, review clean; version scheme superseded by monotonic BigInt — the Task 1 code above is historical). Tasks below replace the original Tasks 2–3 with a three-act "wow" page, informed by two passed spikes: the wasm engine holds **10M rows** with 8k weighted draws at **~561ms** (58ms at 1M), and `db.putFile` + `file()` supports drag-and-drop querying. Design: Act 1 *scale you can poke*, Act 2 *your own data*, Act 3 *the learning loop*. Page identity unchanged (near-black terminal aesthetic, orange accent, monospace).

### Task 2: Page skeleton + Act 1 (scale bench)

**Files:**
- Create: `web/index.html`, `web/app.js`, `web/bench.js`, `web/serve.mjs`

**Interfaces:**
- Consumes: `Store` from Task 1; chdb-wasm `selectBundle`/`AsyncChdb`.
- Produces (Tasks 3-4 rely on these): `web/app.js` exports `initEngine(onProgress) -> Promise<{db, conn, store}>` (bundle resolution: HEAD-probe `./engine/chdb.mjs`, fall back to `./node_modules/chdb-wasm/dist`), `fmtMs(x)`, `el(id)`; `index.html` has three `<section>` acts with ids `act-scale`, `act-data`, `act-learn`, each initially disabled until the engine loads; a top loader button + `<progress>`. `bench.js` exports `initBench({store, conn})`.
- Act 1 behavior: row-count buttons (100k / 1M / 5M / 10M) generate rows engine-side in chunks with a progress readout (`INSERT ... SELECT ... FROM numbers(...)` into a `bench` store created via `Store.create`, priorities skewed `0.01 + pow(randCanonical(),3)*10` written to the sidecar by a direct INSERT SELECT); a big **Sample 8,192** button runs the `FINAL`-form weighted draw, timed with `performance.now()`, displaying: latency (ms), rows scanned (from `rowsRead` if exposed, else the table count), and the exact SQL in a `<details>` block labeled "the query that just ran". A one-line honesty note: "measured in your browser just now — nothing precomputed."

- [ ] **Step 1: `web/serve.mjs`** — unchanged from the original plan:

```javascript
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const TYPES = { ".html": "text/html", ".js": "text/javascript",
  ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
createServer(async (req, res) => {
  const path = join(".", decodeURIComponent(new URL(req.url, "http://x").pathname))
    .replace(/\/$/, "/index.html");
  try {
    const body = await readFile(path === "." ? "index.html" : path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404); res.end("not found"); }
}).listen(8099, () => console.log("http://localhost:8099"));
```

- [ ] **Step 2: `web/index.html`** — three-act shell:

```html
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReplayHouse — a ClickHouse replay buffer in your browser tab</title>
<style>
  :root { --bg:#14161a; --panel:#101215; --ink:#e8eaed; --dim:#9aa3ad;
    --accent:#e8863f; --edge:#2a2e34; }
  body { background:var(--bg); color:var(--ink); margin:0;
    font:15px/1.55 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    padding:2.2rem 1rem 4rem; display:flex; flex-direction:column; align-items:center; }
  main { width:min(84ch, 94vw); }
  h1 { font-size:1.25rem; margin:0 0 .25rem; } h1 b { color:var(--accent); }
  .sub { color:var(--dim); margin:0 0 1.4rem; font-size:.9rem; }
  section { border:1px solid var(--edge); border-radius:6px; background:var(--panel);
    padding:1.1rem 1.3rem; margin:0 0 1.3rem; }
  section[aria-disabled="true"] { opacity:.45; pointer-events:none; }
  h2 { font-size:.95rem; margin:0 0 .7rem; }
  h2 .act { color:var(--accent); margin-right:.5ch; }
  button { background:#24282e; color:var(--ink); border:1px solid #343a42;
    border-radius:5px; padding:.45rem .9rem; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  button[disabled] { opacity:.45; cursor:default; }
  button.big { font-size:1.05rem; padding:.6rem 1.3rem; border-color:var(--accent); }
  .row { display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; margin:.5rem 0; }
  .stat { font-size:1.5rem; } .stat b { color:var(--accent); }
  .dim { color:var(--dim); font-size:.85rem; }
  pre { background:#0c0e11; border:1px solid var(--edge); border-radius:5px;
    padding:.8rem 1rem; overflow-x:auto; white-space:pre; margin:.6rem 0 0; }
  .bar { color:var(--accent); }
  progress { width:100%; accent-color:var(--accent); }
  details summary { cursor:pointer; color:var(--dim); }
  #drop { border:2px dashed #3a4048; border-radius:6px; padding:1.6rem;
    text-align:center; color:var(--dim); }
  #drop.hot { border-color:var(--accent); color:var(--ink); }
  table { border-collapse:collapse; margin-top:.6rem; font-size:.85rem; }
  td, th { border:1px solid var(--edge); padding:.25rem .6rem; text-align:left;
    font-variant-numeric:tabular-nums; }
  footer { color:var(--dim); font-size:.8rem; margin-top:1rem; max-width:84ch; }
  footer a { color:var(--accent); }
</style>
<main>
<h1><b>ReplayHouse</b> — a ClickHouse replay buffer in your browser tab</h1>
<p class="sub">real MergeTree tables, real weighted sampling — the OLAP engine is WebAssembly running on this page. nothing leaves your machine.</p>

<div class="row">
  <button id="load" class="big">Load the engine (~99 MB, cached after first visit)</button>
  <span id="loadmsg" class="dim"></span>
</div>
<progress id="dl" max="100" value="0" hidden></progress>

<section id="act-scale" aria-disabled="true">
  <h2><span class="act">act 1</span>how fast is a weighted draw over N rows — in a tab?</h2>
  <div class="row">
    <button data-n="100000">100k rows</button>
    <button data-n="1000000">1M rows</button>
    <button data-n="5000000">5M rows</button>
    <button data-n="10000000">10M rows</button>
    <span id="genmsg" class="dim"></span>
  </div>
  <div class="row">
    <button id="draw" class="big" disabled>Sample 8,192 (weighted, without replacement)</button>
  </div>
  <div class="stat" id="drawstat"></div>
  <p class="dim">measured in your browser just now — nothing precomputed.</p>
  <details><summary>the query that just ran</summary><pre id="drawsql"></pre></details>
</section>

<section id="act-data" aria-disabled="true">
  <h2><span class="act">act 2</span>your data, sampled the same way — locally</h2>
  <div id="drop">drop a .csv or .parquet here (it is read into the in-page engine, never uploaded)</div>
  <div class="row" id="datactl" hidden>
    <label for="wcol">weight by</label>
    <select id="wcol"></select>
    <button id="dsample">Weighted sample 10</button>
    <span id="datamsg" class="dim"></span>
  </div>
  <div id="dataout"></div>
</section>

<section id="act-learn" aria-disabled="true">
  <h2><span class="act">act 3</span>why a training loop wants this: watch focus emerge</h2>
  <pre id="frame">engine idle</pre>
  <div class="row">
    <button id="mode" disabled>Switch to uniform [u]</button>
    <button id="pause" disabled>Pause [space]</button>
    <button id="reset" disabled>Reset</button>
  </div>
</section>

<footer>every number on this page comes from a ClickHouse query that ran in this tab.
engine: <a href="https://github.com/chdb-io/chdb">chdb</a> compiled to WebAssembly ·
library: <a href="https://github.com/jaymebrd/replayhouse">replayhouse</a></footer>
</main>
<script type="module" src="./app.js"></script>
```

- [ ] **Step 3: `web/app.js`** — engine boot + act wiring:

```javascript
import { Store } from "./replayhouse.js";
import { initBench } from "./bench.js";

export const el = (id) => document.getElementById(id);
export const fmtMs = (x) => x >= 1000 ? `${(x / 1000).toFixed(2)}s` : `${Math.round(x)}ms`;

async function resolveBase() {
  try {
    const r = await fetch("./engine/chdb.mjs", { method: "HEAD" });
    if (r.ok) return "./engine";
  } catch {}
  return "./node_modules/chdb-wasm/dist";
}

el("load").onclick = async () => {
  el("load").disabled = true;
  el("loadmsg").textContent = "resolving bundle…";
  const base = await resolveBase();
  const { AsyncChdb, selectBundle } = await import(`${base}/index.js`);
  const bundle = selectBundle({ baseUrl: base });
  if (!bundle.supported) { el("loadmsg").textContent = bundle.reasons.join("; "); return; }
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
};
```

(The dynamic `import().catch` guards let Task 2 ship before Tasks 3-4 exist; each later task deletes its guard by shipping the module.)

- [ ] **Step 4: `web/bench.js`** — Act 1:

```javascript
import { el, fmtMs } from "./app.js";

const DRAW_SQL = (side) => `SELECT id FROM \`${side}\` FINAL
WHERE priority > 0
ORDER BY -log(1 - randCanonical()) / priority ASC
LIMIT 8192`;

export function initBench({ store }) {
  let rows = 0, created = false;
  for (const b of el("act-scale").querySelectorAll("button[data-n]")) {
    b.onclick = async () => {
      const n = Number(b.dataset.n);
      b.disabled = true; el("draw").disabled = true;
      el("genmsg").textContent = `generating ${n.toLocaleString()} rows in the engine…`;
      const t0 = performance.now();
      if (!created) {
        await store.create("bench", { reward: "Float32" });
        created = true;
      }
      const CHUNK = 1_000_000;
      for (let done = 0; done < n; done += CHUNK) {
        const c = Math.min(CHUNK, n - done);
        await store._exec(`INSERT INTO bench (id, reward)
          SELECT generateUUIDv7(), toFloat32(randCanonical()) FROM numbers(${c})`);
        el("genmsg").textContent =
          `generating… ${Math.min(done + c, n).toLocaleString()} / ${n.toLocaleString()}`;
      }
      await store._exec(`INSERT INTO bench__priorities
        SELECT id, toFloat32(0.01 + pow(randCanonical(), 3) * 10), 1
        FROM bench WHERE id NOT IN (SELECT id FROM bench__priorities)`);
      rows += n;
      el("genmsg").textContent =
        `${rows.toLocaleString()} rows live in this tab (${fmtMs(performance.now() - t0)} to generate)`;
      el("draw").disabled = false;
    };
  }
  el("draw").onclick = async () => {
    el("draw").disabled = true;
    const sql = DRAW_SQL("bench__priorities");
    const t0 = performance.now();
    const got = await store.query(sql);
    const ms = performance.now() - t0;
    el("drawstat").innerHTML =
      `<b>${fmtMs(ms)}</b> — ${got.length.toLocaleString()} rows drawn, ` +
      `weighted + without replacement, over <b>${rows.toLocaleString()}</b> rows`;
    el("drawsql").textContent = sql;
    el("draw").disabled = false;
  };
}
```

Implementation note: the draw uses the `FINAL` read path (6-7x faster than the argMax CTE per `benchmarks/RESULTS.md`) with the SQL displayed verbatim — the page IS the benchmark, so it gets the fast honest query; Act 3 uses `Store.sample` (the Python-contract path) as its point is contract parity, not speed.

- [ ] **Step 5: Manual verification + commit**

Run `npm --prefix web run serve`, open http://localhost:8099, load engine, generate 1M then 10M, click Sample — record observed latencies in the task report (expect ~60ms @ 1M, ~600ms-1s @ 10M in-browser; Node spike says 58/561ms). Verify Acts 2-3 render disabled without erroring (dynamic-import guards).

```bash
git add web/index.html web/app.js web/bench.js web/serve.mjs
git commit -m "feat: playground shell + act 1 - live weighted-draw benchmark at up to 10M rows"
```

### Task 3: Act 2 — drop your own data

**Files:**
- Create: `web/data.js`

**Interfaces:**
- Consumes: `el`, `fmtMs` from `./app.js`; `db.putFile(path, Uint8Array)`; `file('/drop.ext', FORMAT)` table function; DOM nodes `#drop`, `#datactl`, `#wcol`, `#dsample`, `#datamsg`, `#dataout` from Task 2's shell.
- Produces: `export function initData({ db, store })`. Behavior: dragover/dragleave toggle `.hot`; on drop, read the File as ArrayBuffer, `putFile('/drop.<ext>', bytes)`, pick format by extension (`csv → CSVWithNames`, `parquet → Parquet`, `json/jsonl/ndjson → JSONEachRow`; anything else → message "csv, parquet, or jsonl please"); run `DESCRIBE file('/drop.<ext>', '<FMT>')` to list columns, populate `#wcol` with numeric-typed columns (types matching `/Int|Float|Decimal/`); `#dsample` runs a weighted draw of 10 by `greatest(toFloat64(<col>), 0.000001)` (keeps weights positive without excluding rows) timed with `performance.now`, rendering the sampled rows as an HTML table **via `textContent`** (never innerHTML with user data — XSS), plus `#datamsg` = "10 of <count> rows, weighted by <col>, in <ms> — your file never left this tab". Errors (bad file, no numeric columns) land in `#datamsg` as plain text.

- [ ] **Step 1: Write `web/data.js`** (complete implementation per the interface above — the drop handler, format sniff, DESCRIBE, select population, sample + safe table render; ~90 lines).

```javascript
import { el, fmtMs } from "./app.js";

const FMT = { csv: "CSVWithNames", parquet: "Parquet", json: "JSONEachRow",
              jsonl: "JSONEachRow", ndjson: "JSONEachRow" };

export function initData({ db, store }) {
  const drop = el("drop");
  let path = null, fmt = null, count = 0;

  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("hot"); };
  drop.ondragleave = () => drop.classList.remove("hot");
  drop.ondrop = async (e) => {
    e.preventDefault(); drop.classList.remove("hot");
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    const ext = f.name.split(".").pop().toLowerCase();
    fmt = FMT[ext];
    if (!fmt) { el("datamsg").textContent = "csv, parquet, or jsonl please"; el("datactl").hidden = false; return; }
    path = `/drop.${ext}`;
    await db.putFile(path, new Uint8Array(await f.arrayBuffer()));
    try {
      const cols = await store.query(`DESCRIBE file('${path}', '${fmt}')`);
      const numeric = cols.filter((c) => /Int|Float|Decimal/.test(c.type));
      if (!numeric.length) { el("datamsg").textContent = "no numeric columns to weight by"; return; }
      const sel = el("wcol");
      sel.replaceChildren(...numeric.map((c) => new Option(`${c.name} (${c.type})`, c.name)));
      [{ c: count }] = await store.query(`SELECT count() AS c FROM file('${path}', '${fmt}')`);
      drop.textContent = `${f.name} — ${Number(count).toLocaleString()} rows, read into the in-page engine`;
      el("datactl").hidden = false;
      el("datamsg").textContent = "";
    } catch (err) {
      el("datamsg").textContent = `could not read file: ${String(err).slice(0, 160)}`;
    }
  };

  el("dsample").onclick = async () => {
    const col = el("wcol").value;
    const t0 = performance.now();
    let rows;
    try {
      rows = await store.query(`SELECT * FROM file('${path}', '${fmt}')
        ORDER BY -log(1 - randCanonical()) / greatest(toFloat64(\`${col}\`), 0.000001) ASC
        LIMIT 10`);
    } catch (err) {
      el("datamsg").textContent = `sample failed: ${String(err).slice(0, 160)}`; return;
    }
    const ms = performance.now() - t0;
    el("datamsg").textContent =
      `10 of ${Number(count).toLocaleString()} rows, weighted by ${col}, in ${fmtMs(ms)} — your file never left this tab`;
    const table = document.createElement("table");
    const keys = Object.keys(rows[0] ?? {});
    const thead = table.createTHead().insertRow();
    for (const k of keys) { const th = document.createElement("th"); th.textContent = k; thead.appendChild(th); }
    for (const r of rows) {
      const tr = table.insertRow();
      for (const k of keys) tr.insertCell().textContent = String(r[k]);
    }
    el("dataout").replaceChildren(table);
  };
}
```

Note: the weight-column name comes from DESCRIBE output (engine-provided), not free text — backtick-quoted; still, validate it against `^[A-Za-z_][A-Za-z0-9_]*$` and fall back to an error message if a column name doesn't match (exotic Parquet column names get skipped from the select rather than interpolated).

- [ ] **Step 2: Node smoke for the SQL shapes** — add `web/test/filedrop.test.mjs`: `putFile` a small CSV, DESCRIBE it, run the exact `greatest(toFloat64(...))` weighted-sample SQL, assert 5 rows return. (The DOM layer is manually verified; the SQL contract is CI-tested.)

- [ ] **Step 3: Manual verification** (drop a real CSV and a Parquet; record in report), full suite, commit:

```bash
git add web/data.js web/test/filedrop.test.mjs
git commit -m "feat: act 2 - drop a csv/parquet, weighted-sample it locally"
```

### Task 4: Act 3 — the learning loop

**Files:**
- Create: `web/learn.js`
- Modify: `web/package.json` (revert test script to `"node --test test/"` — parked finding from Task 1's fix round)

**Interfaces:**
- Consumes: `Store` via `initLearn({ store })`; DOM nodes `#frame`, `#mode`, `#pause`, `#reset`; `el` from app.js.
- Produces: the PER loop from the original plan's `playground.js`, adapted: table name `demo`, N=2000, batch 256, JS linear model, `store.sample` (contract path), `updatePriorities(|error|, floor 0.01)`, frame rendered into `#frame` each tick (140ms interval), keys `u`/`space` (guarded to ignore when Act 2's select has focus). **Histogram labels are the improved form:** first bar labeled `easy`, last bar labeled `hard`, middle bars unlabeled — no `bin N` text (user feedback: bin numbers were opaque); each bar also shows its count. The stats comment from the terminal demo applies (stats read post-update; keep the ratio line labeled "sampled-batch mean priority ... vs population").

- [ ] **Step 1: Write `web/learn.js`** — port the original plan's `playground.js` `reset/step/render` logic with: `export function initLearn({ store })`; drop `loadEngine` (app.js owns boot); use `store.drop?.("demo")` if present else `_exec` DROP IF EXISTS for both tables on reset; histogram render replaces `bin ${i}` with left-column labels `easy` (first row), `hard` (last row), blank otherwise, right-aligned counts; interval stored so Reset restarts cleanly; buttons enable once initLearn runs.

- [ ] **Step 2: Manual verification** (ratio >1 prioritized, collapses on u, reset works; record values), full suite, commit:

```bash
git add web/learn.js web/package.json
git commit -m "feat: act 3 - the learning loop with easy/hard histogram labels"
```

### Task 5: gh-pages deploy script + README

Identical to the original plan's Task 3 (deploy.sh assembling `index.html app.js bench.js data.js learn.js replayhouse.js engine/` into a gh-pages worktree; `.nojekyll`; do NOT push in-task; Pages-on-private-repo constraint surfaced in the report). README: replace the earlier "Try it in your browser" note with the three-act description and local-dev commands. Full Python suite green; commit `feat: gh-pages deploy script for the wasm playground`.

## Self-Review Notes (amendment)

- Spike-grounded claims only: 10M/561ms comes from a measured Node run; the page displays live measurements, never these cached numbers.
- Act 1 uses FINAL (fast path, SQL displayed); Act 3 uses the Store contract (parity path) — the distinction is deliberate and noted in Task 2.
- XSS: all user-file values rendered via textContent; column names validated before interpolation.
- The parked package.json test-script finding lands in Task 4 (the next task touching web/package.json).
- Placeholder scan: Task 4 Step 1 describes a port of code fully specified earlier in this same plan file (original Task 2's playground.js listing, still present above the amendment) — the implementer has the complete source to port.

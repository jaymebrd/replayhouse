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

### Task 2: Playground page + local server

**Files:**
- Create: `web/index.html`, `web/playground.js`, `web/serve.mjs`

**Interfaces:**
- Consumes: `Store` from Task 1 (exact API above); `chdb-wasm`'s `selectBundle({baseUrl})` + `AsyncChdb.create({moduleUrl, wasmUrl, onProgress})`.
- Produces: a page that (1) shows a "Load engine (~99 MB, cached after first visit)" button with a progress bar, (2) after load runs the PER loop: 2000 rows of `y = 2*x1 - x2`, JS linear model (plain SGD, ~15 lines), `sample(256)` → train → `updatePriorities(|error|)`, (3) renders the same frame as the terminal demo into a `<pre>` (step, mode, loss + sparkline, 10-bin histogram from a live sidecar query, ratio + top-decile lines), (4) controls: Prioritized/Uniform toggle, Pause, Reset, plus `u`/`space` keys. Engine base URL resolves `./engine/` when present (gh-pages layout) falling back to `./node_modules/chdb-wasm/dist/` (local dev).

- [ ] **Step 1: Write `web/serve.mjs`** (local dev only — plain static server, no headers needed for the st bundle)

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

- [ ] **Step 2: Write `web/index.html`**

Monospace, near-black page matching the terminal demo's look (single deliberate theme — it's a terminal); orange accent `#e8863f`; a `<pre id="frame">` as the screen; buttons `Load engine`, `Prioritized/Uniform`, `Pause`, `Reset`; a `<progress>` element for the download; footer linking the GitHub repo and stating "every number on this page comes from a ClickHouse query running in this tab." Full markup in this step (compact — the visual identity is the terminal frame itself):

```html
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReplayHouse — prioritized replay in your browser</title>
<style>
  :root { --bg: #14161a; --ink: #e8eaed; --dim: #9aa3ad; --accent: #e8863f; }
  body { background: var(--bg); color: var(--ink); margin: 0;
    font: 15px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 2rem 1rem; box-sizing: border-box; }
  h1 { font-size: 1.05rem; font-weight: 600; margin: 0 0 .2rem; }
  h1 b { color: var(--accent); }
  .sub { color: var(--dim); font-size: .85rem; margin: 0 0 1.2rem; }
  #frame { background: #101215; border: 1px solid #2a2e34; border-radius: 6px;
    padding: 1rem 1.2rem; min-width: min(76ch, 92vw); min-height: 30ch;
    overflow-x: auto; white-space: pre; }
  .bar { color: var(--accent); }
  .controls { display: flex; gap: .6rem; margin: 1rem 0; flex-wrap: wrap; }
  button { background: #24282e; color: var(--ink); border: 1px solid #343a42;
    border-radius: 5px; padding: .45rem .9rem; font: inherit; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  button[disabled] { opacity: .45; cursor: default; }
  progress { width: min(76ch, 92vw); accent-color: var(--accent); }
  footer { color: var(--dim); font-size: .8rem; margin-top: 1.4rem;
    max-width: 76ch; text-align: center; }
  footer a { color: var(--accent); }
</style>
<h1><b>ReplayHouse</b> — prioritized replay, live in your tab</h1>
<p class="sub">the terminal demo, but the ClickHouse engine is WebAssembly running right here</p>
<pre id="frame">press “Load engine” — chdb-wasm is ~99 MB (cached by your browser after the first visit)</pre>
<progress id="dl" max="100" value="0" hidden></progress>
<div class="controls">
  <button id="load">Load engine (~99 MB)</button>
  <button id="mode" disabled>Switch to uniform [u]</button>
  <button id="pause" disabled>Pause [space]</button>
  <button id="reset" disabled>Reset</button>
</div>
<footer>every number above comes from a ClickHouse query running in this tab —
no server, no simulation. <a href="https://github.com/jaymebrd/replayhouse">github.com/jaymebrd/replayhouse</a></footer>
<script type="module" src="./playground.js"></script>
```

- [ ] **Step 3: Write `web/playground.js`**

The PER loop + renderer (mirrors `examples/demo.py`'s frame; SPARK/histogram/ratio logic ported):

```javascript
import { Store } from "./replayhouse.js";

const N = 2000, BATCH = 256, BINS = 10, SPARK = "▁▂▃▄▅▆▇█";
const $ = (id) => document.getElementById(id);
let store, state, timer;

async function loadEngine() {
  $("load").disabled = true;
  const local = "./node_modules/chdb-wasm/dist";
  const deployed = "./engine";
  const base = await fetch(`${deployed}/chdb.mjs`, { method: "HEAD" })
    .then((r) => (r.ok ? deployed : local)).catch(() => local);
  const { AsyncChdb, selectBundle } = await import(`${base}/index.js`);
  const bundle = selectBundle({ baseUrl: base });
  if (!bundle.supported) { $("frame").textContent = bundle.reasons.join("; "); return; }
  $("dl").hidden = false;
  const db = await AsyncChdb.create({
    moduleUrl: bundle.moduleUrl, wasmUrl: bundle.wasmUrl,
    onProgress: (l, t) => { $("dl").value = (l / t) * 100; },
  });
  $("dl").hidden = true;
  store = await Store.open(await db.connect());
  await reset();
  for (const id of ["mode", "pause", "reset"]) $(id).disabled = false;
  timer = setInterval(step, 140);
}

async function reset() {
  state = { step: 0, mode: "prioritized", paused: false, losses: [],
            w: [0, 0], b: 0 };
  await store._exec?.("DROP TABLE IF EXISTS exp");
  await store._exec?.("DROP TABLE IF EXISTS exp__priorities");
  await store.create("exp", { x1: "Float32", x2: "Float32", y: "Float32" });
  const rows = Array.from({ length: N }, () => {
    const x1 = Math.random() * 2 - 1, x2 = Math.random() * 2 - 1;
    return { x1, x2, y: 2 * x1 - x2 + (Math.random() - 0.5) * 0.1, priority: 1.0 };
  });
  await store.insert("exp", rows);
}

async function step() {
  if (state.paused) return;
  const by = state.mode === "prioritized" ? "priority" : "1";
  const { ids, rows } = await store.sample("exp", BATCH, { by });
  let loss = 0; const errs = []; const lr = 0.05;
  let gw1 = 0, gw2 = 0, gb = 0;
  for (const r of rows) {
    const pred = state.w[0] * r.x1 + state.w[1] * r.x2 + state.b;
    const e = pred - r.y;
    loss += e * e; errs.push(Math.max(Math.abs(e), 0.01));
    gw1 += 2 * e * r.x1; gw2 += 2 * e * r.x2; gb += 2 * e;
  }
  loss /= rows.length;
  state.w[0] -= lr * gw1 / rows.length; state.w[1] -= lr * gw2 / rows.length;
  state.b -= lr * gb / rows.length;
  await store.updatePriorities("exp", ids, errs);
  state.step += 1; state.losses.push(loss);
  await render(new Set(ids));
}

async function render(sampledIds) {
  const ps = await store.query(
    "SELECT id, argMax(priority, version) AS p FROM exp__priorities GROUP BY id");
  const values = ps.map((r) => Number(r.p)).sort((a, b) => a - b);
  const lo = values[0], hi = Math.max(values.at(-1), lo + 1e-9);
  const hist = Array(BINS).fill(0);
  for (const v of values)
    hist[Math.min(BINS - 1, Math.floor(((v - lo) / (hi - lo)) * BINS))] += 1;
  const byId = new Map(ps.map((r) => [r.id, Number(r.p)]));
  const sampled = [...sampledIds].map((i) => byId.get(i)).filter((x) => x != null);
  const popMean = values.reduce((a, b) => a + b, 0) / values.length;
  const sMean = sampled.reduce((a, b) => a + b, 0) / Math.max(sampled.length, 1);
  const cut = values[Math.floor(values.length * 0.9)];
  const topShare = sampled.filter((v) => v >= cut).length / Math.max(sampled.length, 1);

  const tail = state.losses.slice(-56);
  const mn = Math.min(...tail), mx = Math.max(Math.max(...tail), mn + 1e-9);
  const spark = tail.map((v) =>
    SPARK[Math.round(((v - mn) / (mx - mn)) * (SPARK.length - 1))]).join("");
  const peak = Math.max(...hist, 1);
  const bars = hist.map((c, i) =>
    `  bin ${String(i).padEnd(2)} ${String(c).padStart(5)} ` +
    `<span class="bar">${"█".repeat(Math.round((c / peak) * 46))}</span>`).join("\n");
  $("frame").innerHTML =
    `<b>ReplayHouse: prioritized replay — ClickHouse running in this tab</b>\n` +
    `step ${String(state.step).padEnd(6)} mode <b>${state.mode}</b>` +
    `${state.paused ? "   [paused]" : ""}\n\n` +
    `loss ${state.losses.at(-1).toFixed(4).padStart(8)}  ${spark}\n\n` +
    `priority histogram (live query; range ${lo.toFixed(2)}–${hi.toFixed(2)})\n` +
    `${bars}\n\n` +
    `sampled-batch mean priority ${sMean.toFixed(3)} vs population ` +
    `${popMean.toFixed(3)}  (<b>${(sMean / Math.max(popMean, 1e-9)).toFixed(2)}x</b>)\n` +
    `share of batch from top-decile priority: <b>${Math.round(topShare * 100)}%</b>`;
}

$("load").onclick = loadEngine;
$("mode").onclick = toggleMode;
$("pause").onclick = togglePause;
$("reset").onclick = () => reset();
function toggleMode() {
  state.mode = state.mode === "prioritized" ? "uniform" : "prioritized";
  $("mode").textContent = state.mode === "prioritized"
    ? "Switch to uniform [u]" : "Switch to prioritized [u]";
}
function togglePause() {
  state.paused = !state.paused;
  $("pause").textContent = state.paused ? "Resume [space]" : "Pause [space]";
}
addEventListener("keydown", (e) => {
  if (!store || e.target.tagName === "BUTTON") return;
  if (e.key === "u") toggleMode();
  if (e.key === " ") { e.preventDefault(); togglePause(); }
});
```

Implementation note: `Store` exposes `_exec` from Task 1 (it's on the class); if the reset-drop pattern needs a public method, add `async drop(name)` to `Store` (drops both tables with `IF EXISTS`) and use it here instead — reviewer's choice, prefer the public `drop`.

- [ ] **Step 4: Manual verification (documented in the report)**

Run: `npm --prefix web run serve` then open `http://localhost:8099`, click Load engine, verify: frame animates, ratio > 1 in prioritized mode, toggling uniform drops it, Reset works. Record observed ratio values in the task report. (No headless-browser CI for the page itself — the contract tests carry CI; state this in the report.)

- [ ] **Step 5: Commit**

```bash
git add web/index.html web/playground.js web/serve.mjs web/replayhouse.js
git commit -m "feat: browser playground - the PER demo over chdb-wasm in a tab"
```

---

### Task 3: gh-pages deploy + README

**Files:**
- Create: `web/deploy.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: the `web/` tree from Tasks 1-2; `node_modules/chdb-wasm/dist/st/` (single-threaded bundle: `chdb.mjs`, `chdb.wasm`, `index.js` glue — copy whatever `dist/st/` contains).
- Produces: `web/deploy.sh` that builds a `gh-pages` worktree containing `index.html`, `playground.js`, `replayhouse.js`, and `engine/` (the st bundle), commits, and pushes; README "Try it in your browser" line.

- [ ] **Step 1: Write `web/deploy.sh`**

```bash
#!/usr/bin/env bash
# Assemble and push the playground to the gh-pages branch.
# The ~99MB wasm lives ONLY on gh-pages, never on main.
set -euo pipefail
cd "$(dirname "$0")/.."

test -d web/node_modules/chdb-wasm || { echo "run: npm --prefix web install"; exit 1; }

WT=$(mktemp -d)
git worktree add "$WT" gh-pages 2>/dev/null || {
  git worktree add --detach "$WT"
  git -C "$WT" checkout --orphan gh-pages
  git -C "$WT" rm -rf --quiet . 2>/dev/null || true
}

rm -rf "$WT"/{index.html,playground.js,replayhouse.js,engine}
cp web/index.html web/playground.js web/replayhouse.js "$WT/"
mkdir -p "$WT/engine"
cp -R web/node_modules/chdb-wasm/dist/st/. "$WT/engine/"
cp web/node_modules/chdb-wasm/dist/index.js "$WT/engine/" 2>/dev/null || true
touch "$WT/.nojekyll"

git -C "$WT" add -A
git -C "$WT" commit -m "deploy playground $(git rev-parse --short HEAD)" || echo "nothing to deploy"
git -C "$WT" push -u origin gh-pages
git worktree remove "$WT"
echo "enable Pages once with:"
echo "  gh api repos/{owner}/{repo}/pages -X POST -f 'source[branch]=gh-pages' -f 'source[path]=/'"
```

Run: `chmod +x web/deploy.sh`. Do NOT run the deploy in this task — pushing gh-pages and enabling Pages is the controller/user's release step (the repo is currently private; Pages on a private repo requires a paid plan or making the repo public — surface this in the report).

- [ ] **Step 2: README**

Add under the Examples GIF caption:

```markdown
**Try it in your browser:** the same demo runs on ClickHouse compiled to
WebAssembly — no install — once deployed via `web/deploy.sh` (see
[`web/`](web/)). Locally: `npm --prefix web install && npm --prefix web run serve`.
```

- [ ] **Step 3: Full Python suite + commit**

Run: `.venv/bin/pytest tests -q` (93 with node present).

```bash
git add web/deploy.sh README.md
git commit -m "feat: gh-pages deploy script for the wasm playground"
```

---

## Self-Review Notes

- **Honesty constraint:** every rendered stat in `playground.js` derives from `store.query`/`sample`/`updatePriorities` against the wasm engine; the JS model's loss is the real training loss. No simulated numbers.
- **Contract fidelity:** `sampleKey` and the three phase-1 branches mirror `sampling.py` exactly (incl. `m.id` in join mode); divergences are documented in-code (UUIDv4 vs v7, `Date.now()*1000` versions).
- **Type consistency:** `Store.open/create/insert/sample/updatePriorities/query` names match between Task 1 tests and Task 2 usage; the one flagged seam (`_exec` vs a public `drop`) is explicitly delegated to Task 2's implementer with a stated preference.
- **Deploy honesty:** the 99 MB bundle stays off `main`; Pages-on-private-repo constraint is surfaced rather than hidden.
- **Placeholder scan:** clean.

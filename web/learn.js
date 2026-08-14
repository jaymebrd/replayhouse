import { el } from "./app.js";

// The PER loop from the terminal demo, ported to the browser and trained
// live over a chdb-wasm table. `#frame` is set via innerHTML instead of
// textContent so the sparkline/histogram bars can carry a <span class="bar">
// — safe here only because every interpolated value below is a
// program-generated number (loss, priority stats, counts) or a fixed label
// string ("easy"/"hard"); no user-supplied text ever reaches this frame.

const N = 2000, BATCH = 256, BINS = 10, SPARK = "▁▂▃▄▅▆▇█";
let store, state, timer, gen = 0;

async function reset() {
  // clearTimeout alone can't stop a loop that is mid-step (its timer already
  // fired) — it would reschedule itself after this reset started its own loop,
  // leaking one extra concurrent loop per reset. The generation counter makes
  // stale loops (and their pending priority writes) die on their next check.
  const g = ++gen;
  if (timer) clearTimeout(timer);
  state = { step: 0, mode: "prioritized", paused: false, losses: [], w: [0, 0], b: 0 };
  await store._exec("DROP TABLE IF EXISTS demo");
  await store._exec("DROP TABLE IF EXISTS demo__priorities");
  await store.create("demo", { x1: "Float32", x2: "Float32", y: "Float32" });
  const rows = Array.from({ length: N }, () => {
    const x1 = Math.random() * 2 - 1, x2 = Math.random() * 2 - 1;
    return { x1, x2, y: 2 * x1 - x2 + (Math.random() - 0.5) * 0.1, priority: 1.0 };
  });
  await store.insert("demo", rows);
  el("mode").textContent = "Switch to uniform [u]";
  el("pause").textContent = "Pause [space]";
  await render(new Set());
  if (g !== gen) return;
  timer = setTimeout(() => loop(g), 140);
}

// Self-scheduling instead of setInterval: a step is several engine queries and can
// outlast the 140ms cadence — overlapping ticks would pile up unboundedly on the
// single engine queue and starve the other acts' queries.
async function loop(g) {
  if (g !== gen) return;
  const t0 = performance.now();
  try {
    if (!state.paused) await step(g);
  } catch (err) {
    // e.g. a step in flight while reset() drops the tables — keep the loop alive
    console.warn("learn step failed:", err?.message ?? err);
  }
  if (g !== gen) return;
  timer = setTimeout(() => loop(g), Math.max(20, 140 - (performance.now() - t0)));
}

async function step(g) {
  if (state.paused) return;
  const by = state.mode === "prioritized" ? "priority" : "1";
  const { ids, rows } = await store.sample("demo", BATCH, { by });
  if (g !== gen) return; // reset() happened underneath us — don't write stale priorities
  let loss = 0;
  const errs = [];
  const lr = 0.05;
  let gw1 = 0, gw2 = 0, gb = 0;
  for (const r of rows) {
    const pred = state.w[0] * r.x1 + state.w[1] * r.x2 + state.b;
    const e = pred - r.y;
    loss += e * e;
    errs.push(Math.max(Math.abs(e), 0.01));
    gw1 += 2 * e * r.x1;
    gw2 += 2 * e * r.x2;
    gb += 2 * e;
  }
  loss /= rows.length;
  state.w[0] -= lr * gw1 / rows.length;
  state.w[1] -= lr * gw2 / rows.length;
  state.b -= lr * gb / rows.length;
  await store.updatePriorities("demo", ids, errs);
  state.step += 1;
  state.losses.push(loss);
  await render(new Set(ids));
}

async function render(sampledIds) {
  const ps = await store.query(
    "SELECT id, argMax(priority, version) AS p FROM demo__priorities GROUP BY id");
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
  // First bar labeled "easy" (lowest priority), last labeled "hard"
  // (highest priority), middle bars unlabeled — bin numbers were opaque
  // to users, the label tells the story instead.
  const bars = hist.map((c, i) => {
    const label = i === 0 ? "easy" : i === BINS - 1 ? "hard" : "";
    return `  ${label.padEnd(4)} ${String(c).padStart(5)} ` +
      `<span class="bar">${"█".repeat(Math.round((c / peak) * 46))}</span>`;
  }).join("\n");
  el("frame").innerHTML =
    `<b>ReplayHouse: prioritized replay — ClickHouse running in this tab</b>\n` +
    `step ${String(state.step).padEnd(6)} mode <b>${state.mode}</b>` +
    `${state.paused ? "   [paused]" : ""}\n\n` +
    `loss ${(state.losses.at(-1) ?? 0).toFixed(4).padStart(8)}  ${spark}\n\n` +
    `priority histogram (live query; range ${lo.toFixed(2)}–${hi.toFixed(2)})\n` +
    `${bars}\n\n` +
    `sampled-batch mean priority ${sMean.toFixed(3)} vs population ` +
    `${popMean.toFixed(3)}  (<b>${(sMean / Math.max(popMean, 1e-9)).toFixed(2)}x</b>)\n` +
    `share of batch from top-decile priority: <b>${Math.round(topShare * 100)}%</b>`;
}

function toggleMode() {
  state.mode = state.mode === "prioritized" ? "uniform" : "prioritized";
  el("mode").textContent = state.mode === "prioritized"
    ? "Switch to uniform [u]" : "Switch to prioritized [u]";
}

function togglePause() {
  state.paused = !state.paused;
  el("pause").textContent = state.paused ? "Resume [space]" : "Pause [space]";
}

export function initLearn({ store: s }) {
  store = s;
  el("mode").disabled = false;
  el("pause").disabled = false;
  el("reset").disabled = false;
  el("mode").onclick = toggleMode;
  el("pause").onclick = togglePause;
  el("reset").onclick = () => reset();
  addEventListener("keydown", (e) => {
    if (!store) return;
    const t = e.target;
    const tag = t?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "BUTTON") return;
    if (e.key === "u") toggleMode();
    if (e.key === " ") { e.preventDefault(); togglePause(); }
  });
  reset();
}

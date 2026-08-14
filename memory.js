import { el, stage } from "./app.js";
import { RES, N, BATCH, makeNet, forward, trainRows } from "./net.js";
import { quantize, assemble } from "./gif.js";
import { defaultScene } from "./race.js";

// Act 1: why replay buffers exist. A spotlight sweeps across the photo — the
// present, which an agent only lives once. One student trains only on the
// moment; the other records every moment into the store (a replay buffer) and
// studies its memories by priority. As the spotlight moves on, the first
// student's past visibly rots; the second's painting holds.

const W = 16;          // spotlight width, in pixel columns
const COL_MS = 700;    // sweep speed: one new column per COL_MS
const SHARP = 0.08;    // a pixel is "sharp" when mean |channel error| < this
const CODA_MS = 8000;  // after the stream ends, replay gets a short study coda
const PRIO_FLOOR = 0.003;

let store, gen = 0;

export function initMemory({ store: s }) {
  store = s;
  el("mreplay").disabled = false;
  el("mreplay").onclick = () => startStream();
  startStream();
}

async function startStream() {
  stage.stop?.(); // one act drives the engine at a time
  const g = ++gen;
  stage.stop = () => { gen += 1; };
  el("mreplay").textContent = "Restart the stream";
  const oldGif = el("mgif");
  if (oldGif.href) URL.revokeObjectURL(oldGif.href);
  oldGif.hidden = true;
  oldGif.removeAttribute("href");

  const srcCanvas = defaultScene();
  el("mstream").getContext("2d").drawImage(srcCanvas, 0, 0);
  const src = srcCanvas.getContext("2d").getImageData(0, 0, RES, RES).data;
  const target = new Float32Array(N * 3);
  for (let p = 0; p < N; p++)
    for (let c = 0; c < 3; c++) target[p * 3 + c] = src[p * 4 + c] / 255;

  el("memstat").textContent = "opening the experience stream…";
  await store._exec("DROP TABLE IF EXISTS stream");
  await store._exec("DROP TABLE IF EXISTS stream__priorities");
  if (g !== gen) return;
  await store.create("stream",
    { x: "UInt16", y: "UInt16", r: "Float32", g: "Float32", b: "Float32" });
  if (g !== gen) return;

  const netNow = makeNet(7), netMem = makeNet(7); // identical twins
  const prioView = new Float32Array(N); // zero until a pixel is experienced
  let inserted = 0, step = 0, queries = 0, stored = 0;
  let pctNow = 0, pctMem = 0;
  let heatBusy = false, lastHeat = 0, codaAt = null, done = false;
  const t0 = performance.now();

  // the buffer only ever holds what has been lived — insert columns as the
  // spotlight reveals them
  async function insertCols(upto) {
    if (upto <= inserted) return;
    const rows = [];
    for (let cx = inserted; cx < upto; cx++) {
      for (let y = 0; y < RES; y++) {
        const p = y * RES + cx;
        rows.push({ x: cx, y,
          r: target[p * 3], g: target[p * 3 + 1], b: target[p * 3 + 2], priority: 1.0 });
      }
    }
    inserted = upto;
    await store.insert("stream", rows);
    queries += 2;
    stored += rows.length;
  }

  function render(canvas, fn) {
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(RES, RES);
    for (let p = 0; p < N; p++) fn(p, img.data, p * 4);
    ctx.putImageData(img, 0, 0);
  }

  function paint(lo) {
    // the stream: the future is near-black, the present bright, the past dim —
    // the point being that only memory still has the past
    render(el("mstream"), (p, d, o) => {
      const x = p % RES;
      const k = x >= inserted ? 0.10 : x >= lo ? 1.0 : 0.4;
      d[o] = target[p * 3] * 255 * k;
      d[o + 1] = target[p * 3 + 1] * 255 * k;
      d[o + 2] = target[p * 3 + 2] * 255 * k;
      d[o + 3] = 255;
    });
    render(el("mnow"), (p, d, o) => {
      const c = forward(netNow, p);
      d[o] = c[0] * 255; d[o + 1] = c[1] * 255; d[o + 2] = c[2] * 255; d[o + 3] = 255;
    });
    render(el("mmem"), (p, d, o) => {
      const c = forward(netMem, p);
      d[o] = c[0] * 255; d[o + 1] = c[1] * 255; d[o + 2] = c[2] * 255; d[o + 3] = 255;
    });
    // the buffer: black where nothing is stored yet; stored memories as dim
    // grayscale with the current priorities glowing on top
    const blur = new Float32Array(N);
    for (let y = 0; y < RES; y++) {
      for (let x = 0; x < RES; x++) {
        let s = 0, m = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= RES) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= RES) continue;
            s += prioView[yy * RES + xx]; m += 1;
          }
        }
        blur[y * RES + x] = s / m;
      }
    }
    const sorted = Float32Array.from(blur).sort();
    const p95 = Math.max(sorted[Math.floor(N * 0.95)] || 1, 0.03);
    render(el("mbuf"), (p, d, o) => {
      if (p % RES >= inserted) { d[o] = d[o + 1] = d[o + 2] = 0; d[o + 3] = 255; return; }
      const lum = (target[p * 3] * 0.3 + target[p * 3 + 1] * 0.6 + target[p * 3 + 2] * 0.1) * 60;
      const v = Math.min(Math.sqrt(blur[p] / p95), 1);
      d[o] = Math.min(255, lum + 225 * v);
      d[o + 1] = Math.min(255, lum + 120 * v);
      d[o + 2] = Math.min(255, lum + 40 * v);
      d[o + 3] = 255;
    });
  }

  // % of experienced pixels each net currently has sharp
  function evalNets() {
    let sharpNow = 0, sharpMem = 0, m = 0;
    for (let p = 0; p < N; p += 8) {
      if (p % RES >= inserted) continue;
      m += 1;
      let c = forward(netNow, p), e = 0;
      for (let i = 0; i < 3; i++) e += Math.abs(c[i] - target[p * 3 + i]);
      if (e / 3 < SHARP) sharpNow += 1;
      c = forward(netMem, p); e = 0;
      for (let i = 0; i < 3; i++) e += Math.abs(c[i] - target[p * 3 + i]);
      if (e / 3 < SHARP) sharpMem += 1;
    }
    pctNow = Math.round((sharpNow / Math.max(m, 1)) * 100);
    pctMem = Math.round((sharpMem / Math.max(m, 1)) * 100);
  }

  async function refreshHeat() {
    if (heatBusy) return;
    heatBusy = true;
    try {
      const rs = await store.query(
        `SELECT m.x AS x, m.y AS y, argMax(s.priority, s.version) AS p
         FROM stream AS m INNER JOIN stream__priorities AS s ON m.id = s.id
         GROUP BY m.x, m.y`);
      queries += 1;
      if (g !== gen) return;
      for (const r of rs) prioView[Number(r.y) * RES + Number(r.x)] = Number(r.p);
    } finally { heatBusy = false; }
  }

  // --- recorder (same pattern as the race act) ---
  const rec = (() => {
    const PANEL = 192, GAP = 8, CAPTION = 40;
    const c = document.createElement("canvas");
    c.width = PANEL * 4 + GAP * 5;
    c.height = PANEL + GAP * 2 + CAPTION;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    return { c, ctx, PANEL, GAP, frames: [], intervalMs: 1000, last: 0 };
  })();
  function captureFrame(caption) {
    const { c, ctx, PANEL, GAP } = rec;
    ctx.fillStyle = "#f1f2ee";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.font = "13px ui-monospace, Menlo, monospace";
    [["mstream", "the stream (the present)"], ["mnow", "no buffer"],
     ["mmem", "replay buffer"], ["mbuf", "the buffer, filling up"]]
      .forEach(([id, label], i) => {
        const x = GAP + i * (PANEL + GAP);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(el(id), x, GAP, PANEL, PANEL);
        ctx.fillStyle = "#5d6660";
        ctx.textAlign = "center";
        ctx.fillText(label, x + PANEL / 2, GAP + PANEL + 16);
      });
    ctx.fillStyle = "#b65515";
    ctx.textAlign = "left";
    ctx.fillText(caption, GAP, GAP + PANEL + 34);
    rec.frames.push(quantize(ctx.getImageData(0, 0, c.width, c.height)));
    if (rec.frames.length >= 120) {
      rec.frames = rec.frames.filter((_, i) => i % 2 === 0);
      rec.intervalMs *= 2;
    }
  }

  paint(0);
  while (g === gen && !done) {
    try {
      const now = performance.now();
      const secs = (now - t0) / 1000;
      const virtualFront = W + Math.floor((now - t0) / COL_MS);
      await insertCols(Math.min(RES, virtualFront));
      if (g !== gen) return;
      const lo = Math.min(Math.max(0, virtualFront - W), RES);
      const streamLive = lo < RES;
      const [bNow, bMem] = await Promise.all([
        streamLive
          ? store.sample("stream", BATCH, { by: "1", where: `x >= ${lo} AND x < ${inserted}` })
          : Promise.resolve(null),
        store.sample("stream", BATCH, { by: "priority" }),
      ]);
      if (g !== gen) return;
      if (bNow) trainRows(netNow, bNow.rows);
      const errs = trainRows(netMem, bMem.rows);
      await store.updatePriorities("stream", bMem.ids,
        Array.from(errs, (e) => Math.max(e * e, PRIO_FLOOR)));
      queries += streamLive ? 5 : 3;
      step += 1;
      if (g !== gen) return;
      if (step % 4 === 0) evalNets();
      paint(lo);
      if (now - rec.last >= rec.intervalMs) {
        rec.last = now;
        captureFrame(`replayhouse — an experience stream · t=${secs.toFixed(0)}s · ` +
          `${stored.toLocaleString()} memories in the buffer`);
      }
      if (now - lastHeat > 1000) { lastHeat = now; refreshHeat(); }
      if (streamLive) {
        el("memstat").textContent =
          `the present: columns ${lo}–${inserted} · ${stored.toLocaleString()} ` +
          `experiences stored · of everything lived so far: no-buffer ` +
          `${pctNow}% sharp, replay ${pctMem}%`;
      } else {
        codaAt = codaAt ?? now;
        el("memstat").textContent =
          `the stream has ended — replay is still studying its memories · ` +
          `no-buffer ${pctNow}% sharp, replay ${pctMem}%`;
        if (now - codaAt > CODA_MS) {
          done = true;
          evalNets();
          paint(lo);
          const line = `🏁 the stream is over: replay holds ${pctMem}% of the photo ` +
            `sharp — the no-buffer student kept ${pctNow}% and can never revisit the rest`;
          el("memstat").textContent =
            `${line} · ${stored.toLocaleString()} experiences, ` +
            `${queries.toLocaleString()} real queries`;
          captureFrame(line.replace(/^🏁 /, ""));
          setTimeout(() => {
            const bytes = assemble(rec.frames, { delayCs: 12, holdCs: 300 });
            const a = el("mgif");
            if (a.href) URL.revokeObjectURL(a.href);
            a.href = URL.createObjectURL(new Blob([bytes], { type: "image/gif" }));
            a.textContent =
              `Download the stream as a GIF (${(bytes.length / 1e6).toFixed(1)} MB)`;
            a.hidden = false;
          }, 50);
        }
      }
    } catch (err) {
      if (g !== gen) return;
      console.warn("stream step failed:", err?.message ?? err);
      await new Promise((r) => setTimeout(r, 300));
    }
  }
}

import { el, stage } from "./app.js";
import { quantize, assemble } from "./gif.js";
import { RES, N, BATCH, makeNet, forward, trainRows } from "./net.js";

// The hero act: two identical neural nets race to paint a photo, pixel by pixel.
// One draws its training batches uniformly, the other by error priority — and
// every batch is a real weighted draw from a MergeTree table in this tab
// (store.sample), every error a real write-back (store.updatePriorities), and
// the "where it studies" heatmap a live query over the priority sidecar.

const PRIO_FLOOR = 0.003;
// The finish line: a learner is done when its worst-decile pixels are sharp
// (this is what the eye judges — aggregate PSNR is dominated by easy pixels).
const FINISH_DB = 22, STEP_CAP = 900, TIME_CAP_S = 150;

// ---------- the act ----------
let store, gen = 0;

export function initRace({ store: s }) {
  store = s;
  const drop = el("rdrop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("hot"); };
  drop.ondragleave = () => drop.classList.remove("hot");
  drop.ondrop = async (e) => {
    e.preventDefault(); drop.classList.remove("hot");
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    try {
      const bmp = await createImageBitmap(f);
      const c = document.createElement("canvas");
      c.width = RES; c.height = RES;
      const side = Math.min(bmp.width, bmp.height); // center square crop
      c.getContext("2d").drawImage(bmp,
        (bmp.width - side) / 2, (bmp.height - side) / 2, side, side, 0, 0, RES, RES);
      drop.textContent = `${f.name} — racing to paint it`;
      await startRace(c);
    } catch (err) {
      el("racestat").textContent = `could not read that image: ${err?.message ?? err}`;
    }
  };
  el("rpause").disabled = false;
  el("rpause").textContent = "Start the race";
  el("rpause").onclick = () => startRace(defaultScene());
}

export { defaultScene };

// A default scene drawn in-canvas (ours, no licensing) — crisp text and hard
// edges are exactly the detail prioritized sampling visibly wins on.
function defaultScene() {
  const c = document.createElement("canvas");
  c.width = RES; c.height = RES;
  const x = c.getContext("2d");
  const sky = x.createLinearGradient(0, 0, 0, RES);
  sky.addColorStop(0, "#274b8f"); sky.addColorStop(0.55, "#b0577e"); sky.addColorStop(0.75, "#e8863f");
  x.fillStyle = sky; x.fillRect(0, 0, RES, RES);
  x.fillStyle = "#f6d38b";
  x.beginPath(); x.arc(RES * 0.68, RES * 0.42, RES * 0.13, 0, 7); x.fill();
  x.fillStyle = "#1c3a2c";
  x.beginPath(); x.moveTo(0, RES * 0.78); x.quadraticCurveTo(RES * 0.35, RES * 0.58, RES, RES * 0.8);
  x.lineTo(RES, RES); x.lineTo(0, RES); x.fill();
  x.fillStyle = "#0f2018";
  x.beginPath(); x.moveTo(0, RES * 0.92); x.quadraticCurveTo(RES * 0.6, RES * 0.78, RES, RES * 0.94);
  x.lineTo(RES, RES); x.lineTo(0, RES); x.fill();
  x.fillStyle = "#fff";
  x.font = `bold ${Math.round(RES * 0.16)}px sans-serif`;
  x.fillText("Replay", RES * 0.06, RES * 0.23);
  x.fillText("House", RES * 0.06, RES * 0.41);
  return c;
}

let paused = false;

async function startRace(srcCanvas) {
  stage.stop?.(); // one act drives the engine at a time
  const g = ++gen;
  stage.stop = () => { gen += 1; };
  paused = false;
  const oldGif = el("rgif");
  if (oldGif.href) URL.revokeObjectURL(oldGif.href);
  oldGif.hidden = true;
  oldGif.removeAttribute("href");
  el("rpause").textContent = "Pause";
  el("rpause").onclick = () => {
    paused = !paused;
    el("rpause").textContent = paused ? "Resume" : "Pause";
  };

  // target pixels (client copy is only for the "original" panel + PSNR readout;
  // the learners' training data comes back from the store)
  el("rorig").getContext("2d").drawImage(srcCanvas, 0, 0);
  const src = srcCanvas.getContext("2d").getImageData(0, 0, RES, RES).data;
  const target = new Float32Array(N * 3);
  for (let p = 0; p < N; p++)
    for (let c = 0; c < 3; c++) target[p * 3 + c] = src[p * 4 + c] / 255;

  el("racestat").textContent = "loading pixels into the store…";
  await store._exec("DROP TABLE IF EXISTS photo");
  await store._exec("DROP TABLE IF EXISTS photo__priorities");
  if (g !== gen) return;
  await store.create("photo",
    { x: "UInt16", y: "UInt16", r: "Float32", g: "Float32", b: "Float32" });
  const rows = [];
  for (let p = 0; p < N; p++) {
    rows.push({ x: p % RES, y: Math.floor(p / RES),
      r: target[p * 3], g: target[p * 3 + 1], b: target[p * 3 + 2], priority: 1.0 });
  }
  // ids come back in row order, so pixelIds[p] is pixel p's row id — this is
  // what lets the brush write priorities for exact pixels
  const pixelIds = await store.insert("photo", rows);
  if (g !== gen) return;

  el("racesql").textContent =
    `SELECT id FROM (SELECT id, argMax(priority, version) AS priority\n` +
    `  FROM photo__priorities GROUP BY id)\nWHERE priority > 0\n` +
    `ORDER BY -log(1 - randCanonical()) / priority ASC\nLIMIT ${BATCH}`;

  const netU = makeNet(7), netP = makeNet(7); // identical twins
  const prioView = new Float32Array(N).fill(1); // latest heatmap query result
  let step = 0, queries = 0, samples = 0;
  let evalU = { overall: 0, hard: 0 }, evalP = { overall: 0, hard: 0 };
  let hitU = null, hitP = null; // when each learner crossed the finish line
  let done = false;
  let heatBusy = false, lastHeat = 0, lastOptimize = performance.now();
  const t0 = performance.now();

  function render(canvas, fn) {
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(RES, RES);
    for (let p = 0; p < N; p++) fn(p, img.data, p * 4);
    ctx.putImageData(img, 0, 0);
  }
  // --- race recorder: one composite frame per second, quantized at capture
  // (4 bytes/px -> 1) so a long race stays bounded; assembled into a GIF at
  // the flag. Every frame is the live canvases — nothing re-rendered.
  const rec = (() => {
    const PANEL = 192, GAP = 8, CAPTION = 40;
    const c = document.createElement("canvas");
    c.width = PANEL * 4 + GAP * 5;
    c.height = PANEL + GAP * 2 + CAPTION;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;
    return { c, ctx, PANEL, GAP, frames: [], intervalMs: 1000, last: 0 };
  })();

  function captureFrame(caption) {
    const { c, ctx, PANEL, GAP } = rec;
    ctx.fillStyle = "#f1f2ee";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.font = "13px ui-monospace, Menlo, monospace";
    [["rorig", "the photo"], ["runi", "studies random pixels"],
     ["rpri", "studies its mistakes"], ["rheat", "the mistake ledger"]]
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
    if (rec.frames.length >= 120) { // halve into a coarser timelapse
      rec.frames = rec.frames.filter((_, i) => i % 2 === 0);
      rec.intervalMs *= 2;
    }
  }

  function paint() {
    render(el("runi"), (p, d, o) => {
      const c = forward(netU, p);
      d[o] = c[0] * 255; d[o + 1] = c[1] * 255; d[o + 2] = c[2] * 255; d[o + 3] = 255;
    });
    render(el("rpri"), (p, d, o) => {
      const c = forward(netP, p);
      d[o] = c[0] * 255; d[o + 1] = c[1] * 255; d[o + 2] = c[2] * 255; d[o + 3] = 255;
    });
    // blur the raw per-pixel priorities into attention blobs, and glow them
    // over a dim grayscale of the photo — raw speckle told the viewer nothing
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
    render(el("rheat"), (p, d, o) => {
      const lum = (target[p * 3] * 0.3 + target[p * 3 + 1] * 0.6 + target[p * 3 + 2] * 0.1) * 60;
      const v = Math.min(Math.sqrt(blur[p] / p95), 1);
      d[o] = Math.min(255, lum + 225 * v);
      d[o + 1] = Math.min(255, lum + 120 * v);
      d[o + 2] = Math.min(255, lum + 40 * v);
      d[o + 3] = 255;
    });
  }

  // overall PSNR plus PSNR of the worst decile of pixels (each net's own worst)
  function evalNet(net) {
    const pe = [];
    let se = 0;
    for (let p = 0; p < N; p += 8) {
      const o = forward(net, p);
      let e2 = 0;
      for (let c = 0; c < 3; c++) { const e = o[c] - target[p * 3 + c]; e2 += e * e; }
      se += e2; pe.push(e2);
    }
    pe.sort((a, b) => b - a);
    const nHard = Math.max(1, Math.floor(pe.length * 0.1));
    let hs = 0;
    for (let i = 0; i < nHard; i++) hs += pe[i];
    return {
      overall: -10 * Math.log10(se / (pe.length * 3)),
      hard: -10 * Math.log10(hs / (nHard * 3)),
    };
  }

  async function refreshHeat() {
    if (heatBusy) return;
    heatBusy = true;
    try {
      const rs = await store.query(
        `SELECT m.x AS x, m.y AS y, argMax(s.priority, s.version) AS p
         FROM photo AS m INNER JOIN photo__priorities AS s ON m.id = s.id
         GROUP BY m.x, m.y`);
      queries += 1;
      if (g !== gen) return;
      for (const r of rs) prioView[Number(r.y) * RES + Number(r.x)] = Number(r.p);
    } finally { heatBusy = false; }
  }

  // --- the brush: click/drag any panel to mark "study this now" — a real
  // priority write; the next weighted draw floods with those pixels and the
  // patch snaps into focus. The learner's own error write-backs then take
  // over again, so the boost self-corrects once the region is mastered.
  const brush = { pending: new Set(), dragging: false };
  function brushAt(canvas, ev) {
    const rect = canvas.getBoundingClientRect();
    const bx = Math.floor(((ev.clientX - rect.left) / rect.width) * RES);
    const by = Math.floor(((ev.clientY - rect.top) / rect.height) * RES);
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        if (dx * dx + dy * dy > 9) continue;
        const x = bx + dx, y = by + dy;
        if (x < 0 || x >= RES || y < 0 || y >= RES) continue;
        const p = y * RES + x;
        brush.pending.add(p);
        prioView[p] = Math.max(prioView[p], 5); // instant ledger echo; the next
      }                                          // live query overwrites with truth
    }
  }
  for (const id of ["rorig", "rpri", "rheat"]) {
    const c = el(id);
    c.style.cursor = "crosshair";
    c.onpointerdown = (ev) => { brush.dragging = true; brushAt(c, ev); ev.preventDefault(); };
    c.onpointermove = (ev) => { if (brush.dragging) brushAt(c, ev); };
  }
  addEventListener("pointerup", () => { brush.dragging = false; });

  paint();
  while (g === gen && !done) {
    if (paused) { await new Promise((r) => setTimeout(r, 120)); continue; }
    try {
      if (brush.pending.size) {
        const ps = [...brush.pending];
        brush.pending.clear();
        await store.updatePriorities("photo",
          ps.map((p) => pixelIds[p]), ps.map(() => 8.0));
        queries += 1;
        window.__brushFlushed = (window.__brushFlushed ?? 0) + ps.length;
      }
      const [u, p] = await Promise.all([
        store.sample("photo", BATCH, { by: "1" }),
        store.sample("photo", BATCH, { by: "priority" }),
      ]);
      if (g !== gen) return;
      trainRows(netU, u.rows);
      const errs = trainRows(netP, p.rows);
      await store.updatePriorities("photo", p.ids,
        Array.from(errs, (e) => Math.max(e * e, PRIO_FLOOR)));
      // each sample is 2 queries (weighted id draw + fetch), plus the write-back
      queries += 5; samples += u.rows.length + p.rows.length; step += 1;
      if (g !== gen) return;
      const now = performance.now();
      const secs = (now - t0) / 1000;
      if (step % 3 === 0) {
        evalU = evalNet(netU); evalP = evalNet(netP);
        if (!hitU && evalU.hard >= FINISH_DB) hitU = { step, secs };
        if (!hitP && evalP.hard >= FINISH_DB) hitP = { step, secs };
      }
      paint();
      if (now - rec.last >= rec.intervalMs) {
        rec.last = now;
        captureFrame(`replayhouse photo race · t=${secs.toFixed(0)}s · ` +
          `${queries.toLocaleString()} ClickHouse queries in this tab`);
      }
      if (now - lastHeat > 1000) { lastHeat = now; refreshHeat(); }
      if (now - lastOptimize > 20000) {
        lastOptimize = now;
        // compact the append-only priority history so draws stay fast — what a
        // real deployment's background merges do, forced eagerly here
        store._exec("OPTIMIZE TABLE photo__priorities FINAL")
          .then(() => { queries += 1; }, () => {});
      }
      if ((hitU && hitP) || step >= STEP_CAP || secs > TIME_CAP_S) {
        finishRace();
        return;
      }
      const pct = (e) => Math.max(0, Math.min(100, Math.round((e.hard / FINISH_DB) * 100)));
      el("racestat").textContent =
        `step ${step} · ${Math.round(samples / secs).toLocaleString()} pixels/s drawn ` +
        `from the table · ${queries.toLocaleString()} queries · blurriest patches: ` +
        `random ${pct(evalU)}% sharp, mistake-led ${pct(evalP)}% — first to 100% wins`;
    } catch (err) {
      if (g !== gen) return;
      console.warn("race step failed:", err?.message ?? err);
      await new Promise((r) => setTimeout(r, 300));
    }
  }

  function finishRace() {
    done = true;
    paint();
    refreshHeat();
    const t = (h) => (h ? `${h.secs.toFixed(0)}s` : "didn't finish");
    let line;
    if (hitU && hitP) {
      const [win, lose, wName, lName] = hitP.secs <= hitU.secs
        ? [hitP, hitU, "the mistake-student", "the random student"]
        : [hitU, hitP, "the random student", "the mistake-student"];
      line = `🏁 ${wName} got every patch sharp in ${t(win)} — ${lName} needed ` +
        `${t(lose)} (${(lose.secs / win.secs).toFixed(1)}x longer)`;
    } else {
      line = `🏁 time! mistake-student ${t(hitP)} vs random student ${t(hitU)}`;
    }
    el("racestat").textContent =
      `${line} · ${samples.toLocaleString()} samples, ${queries.toLocaleString()} ` +
      `real queries through the store`;
    el("rpause").textContent = "Race again";
    el("rpause").onclick = () => startRace(srcCanvas);
    captureFrame(line.replace(/^🏁 /, ""));
    setTimeout(() => { // let the verdict paint before the ~1s encode
      const bytes = assemble(rec.frames, { delayCs: 12, holdCs: 300 });
      const a = el("rgif");
      if (a.href) URL.revokeObjectURL(a.href);
      a.href = URL.createObjectURL(new Blob([bytes], { type: "image/gif" }));
      a.textContent =
        `Download the race as a GIF (${(bytes.length / 1e6).toFixed(1)} MB)`;
      a.hidden = false;
    }, 50);
  }
}

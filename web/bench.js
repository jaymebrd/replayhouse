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

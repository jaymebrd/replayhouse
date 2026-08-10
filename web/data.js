import { el, fmtMs } from "./app.js";

const FMT = { csv: "CSVWithNames", parquet: "Parquet", json: "JSONEachRow",
              jsonl: "JSONEachRow", ndjson: "JSONEachRow" };

const VALID_COL_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

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
      const numeric = cols.filter((c) => /Int|Float|Decimal/.test(c.type) && VALID_COL_NAME.test(c.name));
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
    if (!VALID_COL_NAME.test(col)) {
      el("datamsg").textContent = `invalid column name: ${col}`;
      return;
    }
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

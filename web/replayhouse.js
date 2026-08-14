// The ReplayHouse SQL contract, mirrored in JS over chdb-wasm.
// Query-shape compatible with src/replayhouse/sampling.py (same key formulas,
// exclusions, and validation; whitespace differs, and create() skips the Python
// client's PARTITION BY / COMMENT — playground tables are ephemeral). The point:
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
  #lastVersion = 0n;

  constructor(conn) { this.conn = conn; }

  static async open(conn) { return new Store(conn); }

  _nextVersion() {
    const now = BigInt(Date.now()) * 1_000_000n;   // ms -> ns scale, comparable to Python's time_ns()
    this.#lastVersion = now > this.#lastVersion ? now : this.#lastVersion + 1n;
    return this.#lastVersion.toString();           // serialize as string; ClickHouse parses quoted UInt64
  }

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
    const version = this._nextVersion();
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
    // by/where are trusted raw SQL (same contract as the Python client) — never wire user input into them without your own validation.
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
    // 8k inline UUIDs ≈ 310KB, over the default 256KB max_query_size — fetch in
    // chunks, mirroring the Python client's _FETCH_CHUNK in table.py.
    const rows = [];
    for (let i = 0; i < ids.length; i += 4000) {
      const chunk = ids.slice(i, i + 4000);
      rows.push(...await this.query(
        `SELECT * FROM \`${name}\` WHERE id IN (${chunk.map((x) => `'${x}'`).join(",")})`));
    }
    return { ids: rows.map((r) => r.id), rows };
  }

  async updatePriorities(name, ids, values) {
    checkName(name);
    if (ids.length !== values.length) throw new Error("ids/values length mismatch");
    if (!ids.length) return;
    for (const id of ids) if (!UUID_RE.test(id)) throw new Error(`not a UUID: ${id}`);
    const version = this._nextVersion();
    const nd = ids.map((id, i) =>
      JSON.stringify({ id, priority: values[i], version })).join("\n");
    await this._exec(
      `INSERT INTO \`${name}__priorities\` FORMAT JSONEachRow\n${nd}`);
  }
}

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

await test("rapid last-write-wins via monotonic versions", async () => {
  await store.create("exp2", { task: "LowCardinality(String)", reward: "Float32" });
  const ids = await store.insert("exp2", Array.from({ length: 5 }, (_, i) => ({ task: `t${i}`, reward: i / 10 })));
  assert.equal(ids.length, 5);
  // two back-to-back updatePriorities with no awaited work between; later call must win due to monotonic versioning
  const p1 = store.updatePriorities("exp2", ids, ids.map(() => 5.0));
  const p2 = store.updatePriorities("exp2", ids, ids.map(() => 0.5));
  await p1;
  await p2;
  const priorities = await store.query(
    `SELECT id, argMax(priority, version) AS priority FROM exp2__priorities GROUP BY id`);
  priorities.forEach((row) => {
    assert.equal(parseFloat(row.priority), 0.5, `expected priority 0.5, got ${row.priority}`);
  });
});

await test("large draws chunk the phase-2 fetch", async () => {
  // k > 4000 forces multiple IN-list fetches — inlining all ids in one query
  // blows the default 256KB max_query_size (mirrors Python's _FETCH_CHUNK).
  await store.create("exp3", { reward: "Float32" });
  await store.insert("exp3",
    Array.from({ length: 4500 }, (_, i) => ({ reward: i / 4500, priority: 1.0 })));
  const { ids, rows } = await store.sample("exp3", 4200);
  assert.equal(ids.length, 4200);
  assert.equal(new Set(ids).size, 4200);
  assert.equal(rows.length, 4200);
});

await conn.close();
await db.terminate();

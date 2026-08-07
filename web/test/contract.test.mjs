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

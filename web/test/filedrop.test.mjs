import { test } from "node:test";
import assert from "node:assert/strict";
import { AsyncChdb } from "chdb-wasm";
import { Store } from "../replayhouse.js";

const db = await AsyncChdb.create({
  moduleUrl: new URL("../node_modules/chdb-wasm/dist/chdb.mjs", import.meta.url).pathname,
});
const conn = await db.connect();
const store = await Store.open(conn);

await test("file drop: putFile → DESCRIBE → weighted sample with greatest()", async () => {
  // Create a small CSV with numeric columns
  const csv = "id,value,weight\n1,100,1.5\n2,200,2.5\n3,300,0.5\n4,400,3.0\n5,500,1.0";
  const bytes = new TextEncoder().encode(csv);

  // putFile the CSV
  await db.putFile("/test.csv", bytes);

  // DESCRIBE the file to get column info
  const describe = await store.query(`DESCRIBE file('/test.csv', 'CSVWithNames')`);
  const numericCols = describe.filter((c) => /Int|Float|Decimal/.test(c.type));
  assert.ok(numericCols.length > 0, "should have numeric columns");

  // Get row count
  const [{ c: count }] = await store.query(`SELECT count() AS c FROM file('/test.csv', 'CSVWithNames')`);
  assert.equal(Number(count), 5, "CSV should have 5 rows");

  // Run weighted sample with greatest() - the exact SQL shape from the implementation
  const weightCol = "weight"; // from describe results
  const rows = await store.query(`SELECT * FROM file('/test.csv', 'CSVWithNames')
    ORDER BY -log(1 - randCanonical()) / greatest(toFloat64(\`${weightCol}\`), 0.000001) ASC
    LIMIT 10`);

  assert.ok(Array.isArray(rows), "sample result should be an array");
  assert.equal(rows.length, 5, "all 5 rows should be in result when sampling 10 from 5");
  assert.ok(rows.every(r => r.id && r.value && r.weight !== undefined),
    "each row should have id, value, and weight");
});

await conn.close();
await db.terminate();

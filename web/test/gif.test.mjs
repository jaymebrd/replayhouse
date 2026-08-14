import { test } from "node:test";
import assert from "node:assert/strict";
import { quantize, assemble } from "../gif.js";

function frame(w, h, rgb) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = rgb[0]; data[i * 4 + 1] = rgb[1]; data[i * 4 + 2] = rgb[2];
    data[i * 4 + 3] = 255;
  }
  return quantize({ data, width: w, height: h });
}

await test("gif structure: header, loop ext, frames, trailer", () => {
  const f1 = frame(8, 6, [255, 0, 0]), f2 = frame(8, 6, [0, 0, 255]);
  const g = assemble([f1, f2], { delayCs: 10, holdCs: 100 });
  assert.equal(String.fromCharCode(...g.slice(0, 6)), "GIF89a");
  assert.equal(g[6] | (g[7] << 8), 8);  // width
  assert.equal(g[8] | (g[9] << 8), 6);  // height
  assert.equal(g[10], 0xF7);            // 256-entry global color table
  assert.equal(g.at(-1), 0x3B);         // trailer
  // exactly two image descriptors
  let seps = 0;
  for (let i = 0; i < g.length - 9; i++)
    if (g[i] === 0x2C && g[i + 1] === 0 && g[i + 2] === 0 && g[i + 3] === 0 && g[i + 4] === 0
        && (g[i + 5] | (g[i + 6] << 8)) === 8) seps += 1;
  assert.equal(seps, 2);
});

await test("quantize maps solid colors to stable palette cells", () => {
  const a = frame(4, 4, [255, 255, 255]).indices;
  const b = frame(4, 4, [0, 0, 0]).indices;
  assert.ok(a.every((v) => v === a[0]));
  assert.ok(b.every((v) => v === 0));
  assert.notEqual(a[0], b[0]);
});

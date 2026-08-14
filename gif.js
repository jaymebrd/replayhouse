// Minimal dependency-free GIF89a encoder for the race recorder.
// Fixed 6x7x6 color cube (252 colors) + Floyd-Steinberg dithering + standard
// LZW. Frames are quantized at capture time (cheap, and 4 bytes/px -> 1).

const PAL = new Uint8Array(256 * 3);
{
  const rl = [0, 51, 102, 153, 204, 255];
  const gl = [0, 43, 85, 128, 170, 213, 255];
  const bl = [0, 51, 102, 153, 204, 255];
  let i = 0;
  for (const r of rl) for (const g of gl) for (const b of bl) {
    PAL[i * 3] = r; PAL[i * 3 + 1] = g; PAL[i * 3 + 2] = b; i += 1;
  }
  // 252 used; the last 4 entries stay black padding
}

const clamp = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

// RGBA ImageData -> palette indices with Floyd-Steinberg error diffusion.
export function quantize({ data, width, height }) {
  const idx = new Uint8Array(width * height);
  // per-channel running error rows
  let cur = new Float32Array((width + 2) * 3);
  let nxt = new Float32Array((width + 2) * 3);
  for (let y = 0; y < height; y++) {
    nxt.fill(0);
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4, e = (x + 1) * 3;
      const r = clamp(data[o] + cur[e]);
      const g = clamp(data[o + 1] + cur[e + 1]);
      const b = clamp(data[o + 2] + cur[e + 2]);
      const ri = Math.round(r / 51), gi = Math.round((g * 6) / 255), bi = Math.round(b / 51);
      const q = (ri * 7 + gi) * 6 + bi;
      idx[y * width + x] = q;
      const er = r - PAL[q * 3], eg = g - PAL[q * 3 + 1], eb = b - PAL[q * 3 + 2];
      cur[e + 3] += (er * 7) / 16; cur[e + 4] += (eg * 7) / 16; cur[e + 5] += (eb * 7) / 16;
      nxt[e - 3] += (er * 3) / 16; nxt[e - 2] += (eg * 3) / 16; nxt[e - 1] += (eb * 3) / 16;
      nxt[e] += (er * 5) / 16; nxt[e + 1] += (eg * 5) / 16; nxt[e + 2] += (eb * 5) / 16;
      nxt[e + 3] += er / 16; nxt[e + 4] += eg / 16; nxt[e + 5] += eb / 16;
    }
    [cur, nxt] = [nxt, cur];
  }
  return { width, height, indices: idx };
}

function lzw(indices, out) {
  const CLEAR = 256, EOI = 257;
  let codeSize = 9, next = 258;
  let dict = new Map();
  let acc = 0, accBits = 0;
  const bytes = [];
  const emit = (code) => {
    acc |= code << accBits;
    accBits += codeSize;
    while (accBits >= 8) { bytes.push(acc & 255); acc >>>= 8; accBits -= 8; }
  };
  emit(CLEAR);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i];
    const key = prefix * 256 + k;
    const hit = dict.get(key);
    if (hit !== undefined) { prefix = hit; continue; }
    emit(prefix);
    dict.set(key, next);
    next += 1;
    if (next === (1 << codeSize) + 1 && codeSize < 12) codeSize += 1;
    if (next === 4096) {
      emit(CLEAR);
      codeSize = 9; next = 258; dict = new Map();
    }
    prefix = k;
  }
  emit(prefix);
  emit(EOI);
  if (accBits > 0) bytes.push(acc & 255);
  // pack into <=255-byte sub-blocks
  out.push(8); // LZW minimum code size
  for (let i = 0; i < bytes.length; i += 255) {
    const n = Math.min(255, bytes.length - i);
    out.push(n);
    for (let j = 0; j < n; j++) out.push(bytes[i + j]);
  }
  out.push(0);
}

// frames: [{width, height, indices}] (all same size), delayCs per frame,
// holdCs extra delay on the last frame. Returns a Uint8Array.
export function assemble(frames, { delayCs = 8, holdCs = 250 } = {}) {
  const { width: w, height: h } = frames[0];
  const out = [];
  const u16 = (v) => { out.push(v & 255, (v >> 8) & 255); };
  for (const c of "GIF89a") out.push(c.charCodeAt(0));
  u16(w); u16(h);
  out.push(0xF7, 0, 0); // GCT present, 256 entries, 8-bit color
  for (const v of PAL) out.push(v);
  // NETSCAPE loop-forever extension
  out.push(0x21, 0xFF, 11);
  for (const c of "NETSCAPE2.0") out.push(c.charCodeAt(0));
  out.push(3, 1, 0, 0, 0);
  frames.forEach((f, i) => {
    const delay = i === frames.length - 1 ? delayCs + holdCs : delayCs;
    out.push(0x21, 0xF9, 4, 0x04, delay & 255, (delay >> 8) & 255, 0, 0); // GCE
    out.push(0x2C); u16(0); u16(0); u16(f.width); u16(f.height); out.push(0); // descriptor
    lzw(f.indices, out);
  });
  out.push(0x3B);
  return Uint8Array.from(out);
}

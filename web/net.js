// The student: a small Fourier-feature MLP that learns (x, y) -> color, plus
// its trainer. Shared by both demo acts so the learners are identical twins —
// the acts differ only in which experiences they feed them.

export const RES = 96, N = RES * RES, BATCH = 1024, SUB = 256;
const FEATS = 96, HID = 64, FDIM = FEATS * 2, LR = 2e-3;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussFrom(r) {
  return () => {
    const u = Math.max(r(), 1e-12), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// Fourier feature matrix, shared by every net — two scales, else all speckle.
const B = new Float32Array(FEATS * 2);
{
  const g = gaussFrom(mulberry32(1234));
  for (let f = 0; f < FEATS; f++) {
    const s = f < FEATS / 2 ? 3.0 : 10.0;
    B[f * 2] = g() * s; B[f * 2 + 1] = g() * s;
  }
}
const feat = new Float32Array(N * FDIM);
for (let p = 0; p < N; p++) {
  const x = ((p % RES) / RES) * 2 - 1, y = (Math.floor(p / RES) / RES) * 2 - 1;
  for (let f = 0; f < FEATS; f++) {
    const t = 2 * Math.PI * (B[f * 2] * x + B[f * 2 + 1] * y);
    feat[p * FDIM + f] = Math.sin(t);
    feat[p * FDIM + FEATS + f] = Math.cos(t);
  }
}

export function makeNet(seed) {
  const g = gaussFrom(mulberry32(seed));
  const init = (n, fanIn) => {
    const a = new Float32Array(n), s = Math.sqrt(2 / fanIn);
    for (let i = 0; i < n; i++) a[i] = g() * s;
    return a;
  };
  const zeros = (n) => new Float32Array(n);
  const net = {
    W1: init(HID * FDIM, FDIM), b1: zeros(HID),
    W2: init(HID * HID, HID), b2: zeros(HID),
    W3: init(3 * HID, HID), b3: zeros(3),
    t: 0,
  };
  for (const k of ["W1", "b1", "W2", "b2", "W3", "b3"]) {
    net["m" + k] = zeros(net[k].length);
    net["v" + k] = zeros(net[k].length);
  }
  return net;
}

// ---------- forward / backward (shared scratch, single-threaded) ----------
const h1 = new Float32Array(HID), h2 = new Float32Array(HID), out = new Float32Array(3);
const d2 = new Float32Array(HID), d1 = new Float32Array(HID), dout = new Float32Array(3);
export function forward(net, p) {
  const { W1, b1, W2, b2, W3, b3 } = net;
  const fo = p * FDIM;
  for (let i = 0; i < HID; i++) {
    let s = b1[i];
    const w = i * FDIM;
    for (let j = 0; j < FDIM; j++) s += W1[w + j] * feat[fo + j];
    h1[i] = s > 0 ? s : 0;
  }
  for (let i = 0; i < HID; i++) {
    let s = b2[i];
    const w = i * HID;
    for (let j = 0; j < HID; j++) s += W2[w + j] * h1[j];
    h2[i] = s > 0 ? s : 0;
  }
  for (let i = 0; i < 3; i++) {
    let s = b3[i];
    const w = i * HID;
    for (let j = 0; j < HID; j++) s += W3[w + j] * h2[j];
    out[i] = 1 / (1 + Math.exp(-s));
  }
  return out;
}

let gW1, gb1, gW2, gb2, gW3, gb3;
function zeroGrads() {
  gW1 = gW1 ?? new Float32Array(HID * FDIM); gW1.fill(0);
  gb1 = gb1 ?? new Float32Array(HID); gb1.fill(0);
  gW2 = gW2 ?? new Float32Array(HID * HID); gW2.fill(0);
  gb2 = gb2 ?? new Float32Array(HID); gb2.fill(0);
  gW3 = gW3 ?? new Float32Array(3 * HID); gW3.fill(0);
  gb3 = gb3 ?? new Float32Array(3); gb3.fill(0);
}

function backward(net, p, r, g, b) {
  forward(net, p);
  const tgt = [r, g, b];
  let err = 0;
  for (let i = 0; i < 3; i++) {
    const e = out[i] - tgt[i];
    err += Math.abs(e);
    dout[i] = 2 * e * out[i] * (1 - out[i]);
  }
  const { W2, W3 } = net;
  for (let j = 0; j < HID; j++) {
    let s = 0;
    for (let i = 0; i < 3; i++) s += W3[i * HID + j] * dout[i];
    d2[j] = h2[j] > 0 ? s : 0;
  }
  for (let j = 0; j < HID; j++) {
    let s = 0;
    for (let i = 0; i < HID; i++) s += W2[i * HID + j] * d2[i];
    d1[j] = h1[j] > 0 ? s : 0;
  }
  for (let i = 0; i < 3; i++) {
    gb3[i] += dout[i];
    const w = i * HID;
    for (let j = 0; j < HID; j++) gW3[w + j] += dout[i] * h2[j];
  }
  for (let i = 0; i < HID; i++) {
    gb2[i] += d2[i];
    const w = i * HID;
    for (let j = 0; j < HID; j++) gW2[w + j] += d2[i] * h1[j];
  }
  const fo = p * FDIM;
  for (let i = 0; i < HID; i++) {
    gb1[i] += d1[i];
    const w = i * FDIM;
    if (d1[i] !== 0)
      for (let j = 0; j < FDIM; j++) gW1[w + j] += d1[i] * feat[fo + j];
  }
  return err / 3;
}

function adam(net, k, g, scale) {
  const w = net[k], m = net["m" + k], v = net["v" + k];
  const b1 = 0.9, b2 = 0.999, eps = 1e-8, t = net.t;
  const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
  // decay the step size as training matures — a fixed lr keeps taking big
  // noisy steps at convergence and the painted image visibly shimmers
  const lr = LR / (1 + t / 600);
  for (let i = 0; i < w.length; i++) {
    const gi = g[i] * scale;
    m[i] = b1 * m[i] + (1 - b1) * gi;
    v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    w[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
  }
}

// One drawn batch trains as several Adam sub-steps: the draw is the expensive
// engine roundtrip, the optimizer steps are nearly free.
export function trainRows(net, rows) {
  const errs = new Float32Array(rows.length);
  for (let off = 0; off < rows.length; off += SUB) {
    zeroGrads();
    const end = Math.min(off + SUB, rows.length);
    for (let i = off; i < end; i++) {
      const row = rows[i];
      const p = Number(row.y) * RES + Number(row.x);
      errs[i] = backward(net, p, Number(row.r), Number(row.g), Number(row.b));
    }
    net.t += 1;
    const scale = 1 / (end - off);
    adam(net, "W1", gW1, scale); adam(net, "b1", gb1, scale);
    adam(net, "W2", gW2, scale); adam(net, "b2", gb2, scale);
    adam(net, "W3", gW3, scale); adam(net, "b3", gb3, scale);
  }
  return errs;
}

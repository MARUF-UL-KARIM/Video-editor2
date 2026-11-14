// Planar geometry for the tracker: 3×3 transforms, robust model fitting (RANSAC) and quad helpers.
// Everything here is pure and allocation-light so it can run per frame inside the tracking worker.
import type { TrackMotionModel } from '../../types';

/** Row-major 3×3 matrix. */
export type Mat3 = Float64Array;

export const identity = (): Mat3 => new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

export function mul(a: Mat3, b: Mat3): Mat3 {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}

export function invert(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-14) return null;
  const k = 1 / det;
  return new Float64Array([A * k, (c * h - b * i) * k, (b * f - c * e) * k, B * k, (a * i - c * g) * k, (c * d - a * f) * k, C * k, (b * g - a * h) * k, (a * e - b * d) * k]);
}

export function applyX(m: Mat3, x: number, y: number): number {
  return (m[0] * x + m[1] * y + m[2]) / (m[6] * x + m[7] * y + m[8]);
}
export function applyY(m: Mat3, x: number, y: number): number {
  return (m[3] * x + m[4] * y + m[5]) / (m[6] * x + m[7] * y + m[8]);
}

/** Local similarity (uniform scale + rotation) of a transform around a point, from its Jacobian. */
export function localSimilarity(m: Mat3, x: number, y: number): { scale: number; rotation: number } {
  const X = m[0] * x + m[1] * y + m[2];
  const Y = m[3] * x + m[4] * y + m[5];
  const W = m[6] * x + m[7] * y + m[8];
  const w2 = W * W || 1e-12;
  const j00 = (m[0] * W - X * m[6]) / w2;
  const j01 = (m[1] * W - X * m[7]) / w2;
  const j10 = (m[3] * W - Y * m[6]) / w2;
  const j11 = (m[4] * W - Y * m[7]) / w2;
  const det = j00 * j11 - j01 * j10;
  return { scale: Math.sqrt(Math.abs(det)) || 1, rotation: Math.atan2(j10 - j01, j00 + j11) };
}

/** In-place Gaussian elimination with partial pivoting. `A` is n×n row-major, `b` length n. Solution is left in `b`. */
export function solveLinear(A: Float64Array, b: Float64Array, n: number): boolean {
  for (let c = 0; c < n; c++) {
    let piv = c;
    let best = Math.abs(A[c * n + c]);
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(A[r * n + c]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-12) return false;
    if (piv !== c) {
      for (let k = c; k < n; k++) {
        const t = A[c * n + k];
        A[c * n + k] = A[piv * n + k];
        A[piv * n + k] = t;
      }
      const t = b[c];
      b[c] = b[piv];
      b[piv] = t;
    }
    const d = A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / d;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * b[k];
    b[r] = s / A[r * n + r];
  }
  return true;
}

/** Deterministic PRNG so the same footage always produces the same track. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Pts = ArrayLike<number>;

// ───────────────────────────── least-squares fits ─────────────────────────────
// src / dst are interleaved x,y arrays; idx selects which correspondences to use.

function fitTranslation(src: Pts, dst: Pts, idx: ArrayLike<number>, n: number): Mat3 | null {
  if (n < 1) return null;
  let dx = 0;
  let dy = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    dx += dst[i] - src[i];
    dy += dst[i + 1] - src[i + 1];
  }
  return new Float64Array([1, 0, dx / n, 0, 1, dy / n, 0, 0, 1]);
}

function fitSimilarity(src: Pts, dst: Pts, idx: ArrayLike<number>, n: number): Mat3 | null {
  if (n < 2) return null;
  let mx = 0, my = 0, ux = 0, uy = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    mx += src[i];
    my += src[i + 1];
    ux += dst[i];
    uy += dst[i + 1];
  }
  mx /= n; my /= n; ux /= n; uy /= n;
  let a = 0, b = 0, d = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    const xs = src[i] - mx, ys = src[i + 1] - my, xd = dst[i] - ux, yd = dst[i + 1] - uy;
    a += xs * xd + ys * yd;
    b += xs * yd - ys * xd;
    d += xs * xs + ys * ys;
  }
  if (d < 1e-9) return null;
  const c = a / d;
  const s = b / d;
  if (c * c + s * s < 1e-6) return null;
  return new Float64Array([c, -s, ux - (c * mx - s * my), s, c, uy - (s * mx + c * my), 0, 0, 1]);
}

function fitAffine(src: Pts, dst: Pts, idx: ArrayLike<number>, n: number): Mat3 | null {
  if (n < 3) return null;
  let mx = 0, my = 0, ux = 0, uy = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    mx += src[i]; my += src[i + 1]; ux += dst[i]; uy += dst[i + 1];
  }
  mx /= n; my /= n; ux /= n; uy /= n;
  let sxx = 0, sxy = 0, syy = 0, sxu = 0, syu = 0, sxv = 0, syv = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    const x = src[i] - mx, y = src[i + 1] - my, u = dst[i] - ux, v = dst[i + 1] - uy;
    sxx += x * x; sxy += x * y; syy += y * y;
    sxu += x * u; syu += y * u; sxv += x * v; syv += y * v;
  }
  const det = sxx * syy - sxy * sxy;
  if (Math.abs(det) < 1e-9 * Math.max(1, sxx * syy)) return null;
  const a = (sxu * syy - syu * sxy) / det;
  const b = (syu * sxx - sxu * sxy) / det;
  const c = (sxv * syy - syv * sxy) / det;
  const d = (syv * sxx - sxv * sxy) / det;
  if (a * d - b * c < 1e-4) return null; // mirrored or collapsed
  return new Float64Array([a, b, ux - a * mx - b * my, c, d, uy - c * mx - d * my, 0, 0, 1]);
}

const AtA = new Float64Array(64);
const Atb = new Float64Array(8);

function fitHomography(src: Pts, dst: Pts, idx: ArrayLike<number>, n: number): Mat3 | null {
  if (n < 4) return null;
  // Hartley normalisation keeps the normal equations well conditioned.
  let mx = 0, my = 0, ux = 0, uy = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    mx += src[i]; my += src[i + 1]; ux += dst[i]; uy += dst[i + 1];
  }
  mx /= n; my /= n; ux /= n; uy /= n;
  let ds = 0, dd = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    ds += Math.hypot(src[i] - mx, src[i + 1] - my);
    dd += Math.hypot(dst[i] - ux, dst[i + 1] - uy);
  }
  if (ds < 1e-9 || dd < 1e-9) return null;
  const ss = (Math.SQRT2 * n) / ds;
  const sd = (Math.SQRT2 * n) / dd;
  AtA.fill(0);
  Atb.fill(0);
  const row = new Float64Array(8);
  const acc = (rhs: number) => {
    for (let r = 0; r < 8; r++) {
      const v = row[r];
      if (v === 0) continue;
      Atb[r] += v * rhs;
      for (let c = r; c < 8; c++) AtA[r * 8 + c] += v * row[c];
    }
  };
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    const x = (src[i] - mx) * ss, y = (src[i + 1] - my) * ss, u = (dst[i] - ux) * sd, v = (dst[i + 1] - uy) * sd;
    row[0] = x; row[1] = y; row[2] = 1; row[3] = 0; row[4] = 0; row[5] = 0; row[6] = -u * x; row[7] = -u * y;
    acc(u);
    row[0] = 0; row[1] = 0; row[2] = 0; row[3] = x; row[4] = y; row[5] = 1; row[6] = -v * x; row[7] = -v * y;
    acc(v);
  }
  for (let r = 0; r < 8; r++) for (let c = 0; c < r; c++) AtA[r * 8 + c] = AtA[c * 8 + r];
  for (let r = 0; r < 8; r++) AtA[r * 8 + r] += 1e-10;
  if (!solveLinear(AtA, Atb, 8)) return null;
  const hn = new Float64Array([Atb[0], Atb[1], Atb[2], Atb[3], Atb[4], Atb[5], Atb[6], Atb[7], 1]);
  const ts = new Float64Array([ss, 0, -mx * ss, 0, ss, -my * ss, 0, 0, 1]);
  const tdInv = new Float64Array([1 / sd, 0, ux, 0, 1 / sd, uy, 0, 0, 1]);
  const h = mul(tdInv, mul(hn, ts));
  if (Math.abs(h[8]) < 1e-12) return null;
  for (let i = 0; i < 9; i++) h[i] /= h[8];
  // reject mirrored / collapsed solutions and anything that folds the points through the horizon
  if (h[0] * h[4] - h[1] * h[3] < 1e-4) return null;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 2;
    if (h[6] * src[i] + h[7] * src[i + 1] + 1 < 0.05) return null;
  }
  return h;
}

export const MODEL_MIN: Record<TrackMotionModel, number> = { translation: 1, similarity: 2, affine: 3, perspective: 4 };

export function fitModel(model: TrackMotionModel, src: Pts, dst: Pts, idx: ArrayLike<number>, n: number): Mat3 | null {
  switch (model) {
    case 'translation':
      return fitTranslation(src, dst, idx, n);
    case 'similarity':
      return fitSimilarity(src, dst, idx, n);
    case 'affine':
      return fitAffine(src, dst, idx, n);
    default:
      return fitHomography(src, dst, idx, n);
  }
}

export interface RansacResult {
  H: Mat3;
  inlier: Uint8Array;
  count: number;
  /** RMS reprojection error of the inliers, px */
  error: number;
}

/**
 * Robust fit of a planar motion model to point correspondences.
 * `use` selects the candidate indices (already filtered by the forward-backward check).
 */
export function ransac(model: TrackMotionModel, src: Pts, dst: Pts, use: Int32Array, nUse: number, thresh: number, maxIter: number, rand: () => number): RansacResult | null {
  const m = MODEL_MIN[model];
  if (nUse < m) return null;
  const t2 = thresh * thresh;
  const total = src.length >> 1;
  const sample = new Int32Array(m);
  let best: Mat3 | null = null;
  let bestCount = 0;
  let bestErr = Infinity;
  const score = (H: Mat3): [number, number] => {
    let c = 0;
    let e = 0;
    for (let k = 0; k < nUse; k++) {
      const i = use[k] * 2;
      const w = H[6] * src[i] + H[7] * src[i + 1] + H[8];
      const dx = (H[0] * src[i] + H[1] * src[i + 1] + H[2]) / w - dst[i];
      const dy = (H[3] * src[i] + H[4] * src[i + 1] + H[5]) / w - dst[i + 1];
      const d = dx * dx + dy * dy;
      if (d <= t2) {
        c++;
        e += d;
      }
    }
    return [c, e];
  };
  let iters = nUse === m ? 1 : maxIter;
  for (let it = 0; it < iters; it++) {
    for (let k = 0; k < m; k++) {
      let pick = 0;
      let ok = false;
      for (let guard = 0; guard < 20 && !ok; guard++) {
        pick = use[Math.min(nUse - 1, Math.floor(rand() * nUse))];
        ok = true;
        for (let j = 0; j < k; j++) if (sample[j] === pick) ok = false;
      }
      sample[k] = pick;
    }
    const H = fitModel(model, src, dst, sample, m);
    if (!H) continue;
    const [c, e] = score(H);
    if (c > bestCount || (c === bestCount && e < bestErr)) {
      best = H;
      bestCount = c;
      bestErr = e;
      const w = c / nUse;
      if (w > 0.999) break;
      const need = Math.log(1 - 0.995) / Math.log(1 - Math.pow(w, m) + 1e-12);
      if (Number.isFinite(need)) iters = Math.min(iters, Math.max(it + 1, Math.ceil(need)));
    }
  }
  if (!best || bestCount < m) return null;
  // local optimisation: refit on the consensus set, twice
  const inlier = new Uint8Array(total);
  const set = new Int32Array(nUse);
  let H = best;
  let count = 0;
  let err = 0;
  for (let round = 0; round < 3; round++) {
    count = 0;
    err = 0;
    for (let k = 0; k < nUse; k++) {
      const i = use[k] * 2;
      const w = H[6] * src[i] + H[7] * src[i + 1] + H[8];
      const dx = (H[0] * src[i] + H[1] * src[i + 1] + H[2]) / w - dst[i];
      const dy = (H[3] * src[i] + H[4] * src[i + 1] + H[5]) / w - dst[i + 1];
      const d = dx * dx + dy * dy;
      if (d <= t2) {
        set[count++] = use[k];
        err += d;
      }
    }
    if (round === 2 || count < m) break;
    const refit = fitModel(model, src, dst, set, count);
    if (!refit) break;
    H = refit;
  }
  if (count < m) return null;
  for (let k = 0; k < count; k++) inlier[set[k]] = 1;
  return { H, inlier, count, error: Math.sqrt(err / count) };
}

// ───────────────────────────── quads ─────────────────────────────
// A quad is 8 numbers: TL, TR, BR, BL (x,y each).

export type Quad = ArrayLike<number>;

export const quadCx = (q: Quad, o = 0) => (q[o] + q[o + 2] + q[o + 4] + q[o + 6]) / 4;
export const quadCy = (q: Quad, o = 0) => (q[o + 1] + q[o + 3] + q[o + 5] + q[o + 7]) / 4;

/** Signed shoelace area; `ax` scales x so normalised coordinates can be measured in square pixels. */
export function quadArea(q: Quad, o = 0, ax = 1): number {
  let a = 0;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    a += q[o + i * 2] * ax * q[o + j * 2 + 1] - q[o + j * 2] * ax * q[o + i * 2 + 1];
  }
  return a / 2;
}

/** Characteristic size: square root of the area. */
export const quadSize = (q: Quad, o = 0, ax = 1) => Math.sqrt(Math.abs(quadArea(q, o, ax))) || 1e-6;

/** Orientation of the quad's horizontal mid-axis (left edge centre → right edge centre), radians. */
export function quadAngle(q: Quad, o = 0, ax = 1): number {
  const lx = (q[o] + q[o + 6]) / 2, ly = (q[o + 1] + q[o + 7]) / 2;
  const rx = (q[o + 2] + q[o + 4]) / 2, ry = (q[o + 3] + q[o + 5]) / 2;
  return Math.atan2(ry - ly, (rx - lx) * ax);
}

/** Build a rotated rectangle. Sizes are in the same units as the centre; `ax` (= width/height of the frame) keeps it rectangular in pixels when coordinates are normalised. */
export function rectQuad(cx: number, cy: number, w: number, h: number, rot: number, ax = 1): number[] {
  const c = Math.cos(rot), s = Math.sin(rot);
  const out: number[] = [];
  const hw = (w * ax) / 2, hh = h / 2;
  for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    const x = sx * hw, y = sy * hh;
    out.push(cx + (x * c - y * s) / ax, cy + x * s + y * c);
  }
  return out;
}

export function pointInQuad(q: Quad, x: number, y: number): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    const cr = (q[j * 2] - q[i * 2]) * (y - q[i * 2 + 1]) - (q[j * 2 + 1] - q[i * 2 + 1]) * (x - q[i * 2]);
    if (Math.abs(cr) < 1e-12) continue;
    const s = cr > 0 ? 1 : -1;
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

/** Inside the ellipse inscribed in a (parallelogram-ish) quad. */
export function pointInEllipse(q: Quad, x: number, y: number): boolean {
  const cx = quadCx(q), cy = quadCy(q);
  const ex = (q[2] + q[4] - q[0] - q[6]) / 4, ey = (q[3] + q[5] - q[1] - q[7]) / 4; // half horizontal axis
  const fx = (q[6] + q[4] - q[0] - q[2]) / 4, fy = (q[7] + q[5] - q[1] - q[3]) / 4; // half vertical axis
  const det = ex * fy - ey * fx;
  if (Math.abs(det) < 1e-12) return false;
  const dx = x - cx, dy = y - cy;
  const u = (dx * fy - dy * fx) / det;
  const v = (dy * ex - dx * ey) / det;
  return u * u + v * v <= 1;
}

const IDX4 = new Int32Array([0, 1, 2, 3]);
/** Exact homography taking one quad onto another. */
export function quadToQuad(src: Quad, dst: Quad): Mat3 | null {
  return fitHomography(src, dst, IDX4, 4);
}

export function transformQuad(H: Mat3, q: Quad, out: Float64Array | number[] = new Float64Array(8)): Float64Array | number[] {
  for (let i = 0; i < 4; i++) {
    const x = q[i * 2], y = q[i * 2 + 1];
    const w = H[6] * x + H[7] * y + H[8];
    out[i * 2] = (H[0] * x + H[1] * y + H[2]) / w;
    out[i * 2 + 1] = (H[3] * x + H[4] * y + H[5]) / w;
  }
  return out;
}

export function median(values: number[] | Float64Array | Float32Array, n = values.length): number {
  if (n <= 0) return 0;
  const a = Array.prototype.slice.call(values, 0, n) as number[];
  a.sort((x, y) => x - y);
  const m = n >> 1;
  return n % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

export const wrapAngle = (a: number) => {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
};

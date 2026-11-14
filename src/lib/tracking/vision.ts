// Low-level vision primitives used by the trackers: image pyramids, Scharr gradients, Shi–Tomasi corners,
// pyramidal Lucas–Kanade optical flow and normalised cross-correlation template matching.
//
// The LK solver mirrors the one in the reference tracker (per-point inverse-compositional solve over an image
// pyramid, Scharr derivatives with zero padding, minimum-eigenvalue gating, oscillation damping and a final
// photometric error), implemented on the CPU so it runs anywhere a worker does.

export interface Level {
  w: number;
  h: number;
  img: Float32Array;
  gx: Float32Array | null;
  gy: Float32Array | null;
}
export type Pyramid = Level[];

export function grayFromRGBA(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Float32Array {
  const n = w * h;
  const out = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) out[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return out;
}

function pyrDown(src: Level): Level {
  const { w, h, img } = src;
  const w2 = (w + 1) >> 1;
  const h2 = (h + 1) >> 1;
  const tmp = new Float32Array(w2 * h);
  const r = (v: number, n: number) => (v < 0 ? -v : v >= n ? 2 * n - 2 - v : v);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x2 = 0; x2 < w2; x2++) {
      const x = x2 * 2;
      let v: number;
      if (x >= 2 && x + 2 < w) v = img[row + x - 2] + 4 * img[row + x - 1] + 6 * img[row + x] + 4 * img[row + x + 1] + img[row + x + 2];
      else v = img[row + r(x - 2, w)] + 4 * img[row + r(x - 1, w)] + 6 * img[row + x] + 4 * img[row + r(x + 1, w)] + img[row + r(x + 2, w)];
      tmp[y * w2 + x2] = v;
    }
  }
  const out = new Float32Array(w2 * h2);
  for (let y2 = 0; y2 < h2; y2++) {
    const y = y2 * 2;
    const a = r(y - 2, h) * w2, b = r(y - 1, h) * w2, c = y * w2, d = r(y + 1, h) * w2, e = r(y + 2, h) * w2;
    for (let x = 0; x < w2; x++) out[y2 * w2 + x] = (tmp[a + x] + 4 * tmp[b + x] + 6 * tmp[c + x] + 4 * tmp[d + x] + tmp[e + x]) / 256;
  }
  return { w: w2, h: h2, img: out, gx: null, gy: null };
}

/** `levels` is the index of the coarsest level (0 = just the base image). */
export function buildPyramid(gray: Float32Array, w: number, h: number, levels: number): Pyramid {
  const pyr: Pyramid = [{ w, h, img: gray, gx: null, gy: null }];
  for (let l = 1; l <= levels; l++) {
    const prev = pyr[l - 1];
    if (Math.min(prev.w, prev.h) < 24) break;
    pyr.push(pyrDown(prev));
  }
  return pyr;
}

/** Scharr derivatives (normalised by 32), zero at the border. Computed once per level and cached on it. */
export function ensureGradients(L: Level) {
  if (L.gx) return;
  const { w, h, img } = L;
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    const o = y * w;
    for (let x = 1; x < w - 1; x++) {
      const i = o + x;
      const tl = img[i - w - 1], tr = img[i - w + 1], bl = img[i + w - 1], br = img[i + w + 1];
      gx[i] = (3 * (tr + br - tl - bl) + 10 * (img[i + 1] - img[i - 1])) / 32;
      gy[i] = (3 * (bl + br - tl - tr) + 10 * (img[i + w] - img[i - w])) / 32;
    }
  }
  L.gx = gx;
  L.gy = gy;
}

function sampleClamped(a: Float32Array, w: number, h: number, x: number, y: number): number {
  if (x < 0) x = 0;
  else if (x > w - 1) x = w - 1;
  if (y < 0) y = 0;
  else if (y > h - 1) y = h - 1;
  const x0 = x | 0, y0 = y | 0;
  const x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
  const fx = x - x0, fy = y - y0;
  const p = a[y0 * w + x0], q = a[y0 * w + x1], r = a[y1 * w + x0], s = a[y1 * w + x1];
  return (p + (q - p) * fx) * (1 - fy) + (r + (s - r) * fx) * fy;
}

export const bilinear = sampleClamped;

export interface LKParams {
  /** half window: the patch is (2r+1)² */
  radius: number;
  iterations: number;
  epsilon: number;
  /** minimum mean eigenvalue of the structure tensor (intensity²/px²) for a patch to be trackable */
  minEigen: number;
}

const MAX_PATCH = 31 * 31;
const tI = new Float32Array(MAX_PATCH);
const tGx = new Float32Array(MAX_PATCH);
const tGy = new Float32Array(MAX_PATCH);

/**
 * Track `n` points from pyramid A to pyramid B.
 * pts / out are interleaved x,y at level-0 resolution. status[i] = 1 when the point was found; err[i] is the mean
 * absolute photometric difference of the matched patch. `guess` optionally seeds the search (e.g. a predicted position).
 */
export function trackLK(A: Pyramid, B: Pyramid, pts: Float32Array, n: number, out: Float32Array, status: Uint8Array, err: Float32Array, p: LKParams, guess?: Float32Array | null) {
  const top = Math.min(A.length, B.length) - 1;
  const r = p.radius;
  const side = 2 * r + 1;
  const area = side * side;
  const eps2 = p.epsilon * p.epsilon;
  for (let l = 0; l <= top; l++) ensureGradients(A[l]);
  for (let i = 0; i < n; i++) {
    const px = pts[i * 2], py = pts[i * 2 + 1];
    const sc = 1 / (1 << top);
    let nx = (guess ? guess[i * 2] : px) * sc;
    let ny = (guess ? guess[i * 2 + 1] : py) * sc;
    let ok = false;
    let photo = 0;
    for (let level = top; level >= 0; level--) {
      const LA = A[level], LB = B[level];
      const w = LA.w, h = LA.h;
      const s = 1 / (1 << level);
      const sx = px * s, sy = py * s;
      if (level !== top) {
        nx *= 2;
        ny *= 2;
      }
      const img = LA.img, gx = LA.gx!, gy = LA.gy!;
      // template patch + structure tensor
      let a = 0, b = 0, c = 0;
      const ix = Math.floor(sx), iy = Math.floor(sy);
      const fx = sx - ix, fy = sy - iy;
      const inside = ix - r >= 0 && iy - r >= 0 && ix + r + 1 < w && iy + r + 1 < h;
      let k = 0;
      if (inside) {
        const w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
        for (let dy = -r; dy <= r; dy++) {
          let o = (iy + dy) * w + ix - r;
          for (let dx = -r; dx <= r; dx++, o++, k++) {
            tI[k] = img[o] * w00 + img[o + 1] * w01 + img[o + w] * w10 + img[o + w + 1] * w11;
            const g1 = gx[o] * w00 + gx[o + 1] * w01 + gx[o + w] * w10 + gx[o + w + 1] * w11;
            const g2 = gy[o] * w00 + gy[o + 1] * w01 + gy[o + w] * w10 + gy[o + w + 1] * w11;
            tGx[k] = g1;
            tGy[k] = g2;
            a += g1 * g1;
            b += g1 * g2;
            c += g2 * g2;
          }
        }
      } else {
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++, k++) {
            tI[k] = sampleClamped(img, w, h, sx + dx, sy + dy);
            const g1 = sampleClamped(gx, w, h, sx + dx, sy + dy);
            const g2 = sampleClamped(gy, w, h, sx + dx, sy + dy);
            tGx[k] = g1;
            tGy[k] = g2;
            a += g1 * g1;
            b += g1 * g2;
            c += g2 * g2;
          }
        }
      }
      const eigen = (a + c - Math.sqrt((a - c) * (a - c) + 4 * b * b)) / (2 * area);
      const det = a * c - b * b;
      if (eigen < p.minEigen || det <= 1e-8) {
        if (level === 0) ok = false;
        continue;
      }
      const imgB = LB.img;
      let ldx = 0, ldy = 0;
      for (let it = 0; it < p.iterations; it++) {
        if (nx < -r || ny < -r || nx >= w + r || ny >= h + r) break;
        let sumX = 0, sumY = 0;
        const jx = Math.floor(nx), jy = Math.floor(ny);
        const gxf = nx - jx, gyf = ny - jy;
        k = 0;
        if (jx - r >= 0 && jy - r >= 0 && jx + r + 1 < w && jy + r + 1 < h) {
          const w00 = (1 - gxf) * (1 - gyf), w01 = gxf * (1 - gyf), w10 = (1 - gxf) * gyf, w11 = gxf * gyf;
          for (let dy = -r; dy <= r; dy++) {
            let o = (jy + dy) * w + jx - r;
            for (let dx = -r; dx <= r; dx++, o++, k++) {
              const d = imgB[o] * w00 + imgB[o + 1] * w01 + imgB[o + w] * w10 + imgB[o + w + 1] * w11 - tI[k];
              sumX += tGx[k] * d;
              sumY += tGy[k] * d;
            }
          }
        } else {
          for (let dy = -r; dy <= r; dy++) {
            for (let dx = -r; dx <= r; dx++, k++) {
              const d = sampleClamped(imgB, w, h, nx + dx, ny + dy) - tI[k];
              sumX += tGx[k] * d;
              sumY += tGy[k] * d;
            }
          }
        }
        const ddx = (b * sumY - c * sumX) / det;
        const ddy = (b * sumX - a * sumY) / det;
        nx += ddx;
        ny += ddy;
        if (ddx * ddx + ddy * ddy <= eps2) break;
        if (it > 0 && Math.abs(ddx + ldx) < 0.01 && Math.abs(ddy + ldy) < 0.01) {
          nx -= ddx * 0.5;
          ny -= ddy * 0.5;
          break;
        }
        ldx = ddx;
        ldy = ddy;
      }
      if (level === 0) {
        if (!(nx >= 0 && ny >= 0 && nx <= w - 1 && ny <= h - 1)) {
          ok = false;
          break;
        }
        let acc = 0;
        k = 0;
        for (let dy = -r; dy <= r; dy += 2) {
          for (let dx = -r; dx <= r; dx += 2) {
            acc += Math.abs(sampleClamped(imgB, w, h, nx + dx, ny + dy) - tI[(dy + r) * side + dx + r]);
            k++;
          }
        }
        photo = acc / k;
        ok = true;
      }
    }
    if (ok) {
      out[i * 2] = nx;
      out[i * 2 + 1] = ny;
      status[i] = 1;
      err[i] = photo;
    } else {
      out[i * 2] = px;
      out[i * 2 + 1] = py;
      status[i] = 0;
      err[i] = 1e6;
    }
  }
}

export interface CornerOptions {
  max: number;
  /** fraction of the strongest response a corner must reach */
  quality: number;
  minDist: number;
  /** bounding box to search (level-0 px) */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** optional predicate (region shape / mask) */
  accept?: (x: number, y: number) => boolean;
  /** existing points to keep away from (interleaved), and how many */
  existing?: Float32Array;
  nExisting?: number;
}

/** Shi–Tomasi "good features to track": minimum eigenvalue of the 3×3 structure tensor, then distance-based NMS. */
export function detectCorners(L: Level, o: CornerOptions): Float32Array {
  ensureGradients(L);
  const { w, h } = L;
  const gx = L.gx!, gy = L.gy!;
  const x0 = Math.max(2, Math.floor(o.x0)), y0 = Math.max(2, Math.floor(o.y0));
  const x1 = Math.min(w - 3, Math.ceil(o.x1)), y1 = Math.min(h - 3, Math.ceil(o.y1));
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  if (bw < 3 || bh < 3 || o.max <= 0) return new Float32Array(0);
  const resp = new Float32Array(bw * bh);
  let maxR = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let a = 0, b = 0, c = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const row = (y + dy) * w + x;
        for (let dx = -1; dx <= 1; dx++) {
          const g1 = gx[row + dx], g2 = gy[row + dx];
          a += g1 * g1;
          b += g1 * g2;
          c += g2 * g2;
        }
      }
      const e = (a + c - Math.sqrt((a - c) * (a - c) + 4 * b * b)) * 0.5;
      resp[(y - y0) * bw + (x - x0)] = e;
      if (e > maxR) maxR = e;
    }
  }
  if (maxR <= 1e-6) return new Float32Array(0);
  const thr = Math.max(maxR * o.quality, 4);
  const cand: number[] = [];
  for (let y = 1; y < bh - 1; y++) {
    for (let x = 1; x < bw - 1; x++) {
      const i = y * bw + x;
      const v = resp[i];
      if (v < thr) continue;
      if (v < resp[i - 1] || v < resp[i + 1] || v < resp[i - bw] || v < resp[i + bw] || v < resp[i - bw - 1] || v < resp[i - bw + 1] || v < resp[i + bw - 1] || v < resp[i + bw + 1]) continue;
      if (o.accept && !o.accept(x + x0, y + y0)) continue;
      cand.push(i);
    }
  }
  cand.sort((p, q) => resp[q] - resp[p]);
  const cell = Math.max(1, o.minDist);
  const gw = Math.ceil(w / cell) + 1, gh = Math.ceil(h / cell) + 1;
  const grid = new Map<number, number[]>();
  const put = (x: number, y: number) => {
    const key = Math.floor(y / cell) * gw + Math.floor(x / cell);
    const list = grid.get(key);
    if (list) list.push(x, y);
    else grid.set(key, [x, y]);
  };
  const free = (x: number, y: number) => {
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell);
    const d2 = o.minDist * o.minDist;
    for (let yy = cy - 1; yy <= cy + 1; yy++) {
      if (yy < 0 || yy >= gh) continue;
      for (let xx = cx - 1; xx <= cx + 1; xx++) {
        const list = grid.get(yy * gw + xx);
        if (!list) continue;
        for (let k = 0; k < list.length; k += 2) {
          const dx = list[k] - x, dy = list[k + 1] - y;
          if (dx * dx + dy * dy < d2) return false;
        }
      }
    }
    return true;
  };
  if (o.existing) for (let k = 0; k < (o.nExisting ?? o.existing.length >> 1); k++) put(o.existing[k * 2], o.existing[k * 2 + 1]);
  const out: number[] = [];
  for (const i of cand) {
    if (out.length >= o.max * 2) break;
    const x = (i % bw) + x0, y = Math.floor(i / bw) + y0;
    if (!free(x, y)) continue;
    put(x, y);
    out.push(x, y);
  }
  return new Float32Array(out);
}

// ───────────────────────────── template matching ─────────────────────────────

/** Sample an S×S patch through a quad (bilinear in the quad's own frame) and normalise it to zero mean / unit norm. */
export function extractPatch(L: Level, q: ArrayLike<number>, S: number, out: Float32Array = new Float32Array(S * S), dx = 0, dy = 0): { patch: Float32Array; std: number } {
  let sum = 0;
  let k = 0;
  for (let j = 0; j < S; j++) {
    const v = (j + 0.5) / S;
    const lx = q[0] + (q[6] - q[0]) * v, ly = q[1] + (q[7] - q[1]) * v;
    const rx = q[2] + (q[4] - q[2]) * v, ry = q[3] + (q[5] - q[3]) * v;
    for (let i = 0; i < S; i++, k++) {
      const u = (i + 0.5) / S;
      const val = sampleClamped(L.img, L.w, L.h, lx + (rx - lx) * u + dx, ly + (ry - ly) * u + dy);
      out[k] = val;
      sum += val;
    }
  }
  const mean = sum / k;
  let sq = 0;
  for (let i = 0; i < k; i++) {
    out[i] -= mean;
    sq += out[i] * out[i];
  }
  const norm = Math.sqrt(sq);
  if (norm > 1e-6) for (let i = 0; i < k; i++) out[i] /= norm;
  return { patch: out, std: norm / Math.sqrt(k) };
}

export function correlate(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Bounded re-detection: slide the quad (translation only) inside ±radius and return the offset with the best
 * normalised cross-correlation against a reference template.
 */
export function searchTemplate(L: Level, q: ArrayLike<number>, tpl: Float32Array, S: number, radius: number, step: number): { dx: number; dy: number; score: number } {
  const scratch = new Float32Array(S * S);
  let best = { dx: 0, dy: 0, score: -1 };
  const scan = (cx: number, cy: number, rad: number, st: number) => {
    for (let dy = cy - rad; dy <= cy + rad + 1e-6; dy += st) {
      for (let dx = cx - rad; dx <= cx + rad + 1e-6; dx += st) {
        const { patch, std } = extractPatch(L, q, S, scratch, dx, dy);
        if (std < 1.5) continue;
        const sc = correlate(patch, tpl);
        if (sc > best.score) best = { dx, dy, score: sc };
      }
    }
  };
  scan(0, 0, radius, step);
  if (best.score > -1 && step > 1) scan(best.dx, best.dy, step, Math.max(1, step / 4));
  return best;
}

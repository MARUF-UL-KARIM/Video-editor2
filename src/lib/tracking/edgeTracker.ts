// Edge Track: edge-detection filter + collective multi-dot tracking inside the tracking circle.
//
// How it works:
//  1. ROI-local edge detection (3×3 Gaussian denoise → Scharr gradients → Canny non-maximum suppression
//     → adaptive hysteresis thresholding) strictly inside the tracking circle.
//  2. Places tracking dots directly along the detected 1px edge ridges, balanced across 8 angular sectors
//     so diverse edge orientations constrain 2D motion + scale collectively (eliminating the aperture problem).
//  3. Tracks only a cropped local pyramid around the predicted circle position (typically 5–10× fewer pixels
//     than full-frame pyramid trackers) with velocity warm-start Lucas–Kanade + edge-alignment verification.
//  4. Combines all edge dots collectively via RANSAC + Huber-weighted IRLS directly against the reference
//     constellation, yielding rock-steady sub-pixel accuracy without cumulative frame-to-frame drift.

import type { TrackMask, TrackMotionModel } from '../../types';
import {
  applyX,
  applyY,
  identity,
  invert,
  localSimilarity,
  median,
  mul,
  mulberry32,
  pointInEllipse,
  quadCx,
  quadCy,
  quadSize,
  ransac,
  transformQuad,
  type Mat3,
} from './math';
import {
  POINT_INLIER,
  POINT_LOST,
  POINT_OUTLIER,
  POINT_RESCUED,
  type MaskImage,
  type StepOut,
  type TrackerFrame,
} from './pointTracker';
import {
  buildPyramid,
  correlate,
  ensureGradients,
  extractPatch,
  searchTemplate,
  trackLK,
  type LKParams,
  type Pyramid,
} from './vision';

export interface EdgeDetectOptions {
  maxDots: number;
  /** 10–100, default 55 */
  sensitivity: number;
  minDist?: number;
  existing?: Float32Array;
  nExisting?: number;
  accept?: (x: number, y: number) => boolean;
  /** when > 0, return up to this many evenly sampled detected edge pixel coordinates [x, y, ...] */
  maxPreviewEdges?: number;
  /** inward shrink fraction so LK windows stay inside the circle (default 0.06) */
  shrink?: number;
}

export interface EdgeDetectResult {
  /** interleaved x, y coordinates of tracking dots placed along detected edges */
  dots: Float32Array;
  /** interleaved unit gradient normals (nx, ny) for each dot */
  normals: Float32Array;
  /** normalized edge-strength weight for each dot (0.5–1.5) */
  weights: Float32Array;
  /** interleaved x, y coordinates of detected edge pixels for live visual overlay */
  edgePixels: Float32Array;
  /** total count of detected edge pixels inside the circle */
  edgeCount: number;
  /** overall edge contrast quality inside the circle, 0–1 */
  quality: number;
}

/** Fast cropped grayscale extraction from an RGBA frame buffer. */
export function grayFromRGBACrop(
  rgba: Uint8ClampedArray | Uint8Array,
  fullW: number,
  ox: number,
  oy: number,
  cw: number,
  ch: number,
): Float32Array {
  const out = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) {
    const srcRow = (oy + y) * fullW + ox;
    const dstRow = y * cw;
    for (let x = 0; x < cw; x++) {
      const j = (srcRow + x) * 4;
      out[dstRow + x] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
    }
  }
  return out;
}

/**
 * Detect edges strictly inside the tracking circle using a fast Canny-style pipeline
 * (3×3 Gaussian → Scharr gradients → gradient-normal NMS → adaptive hysteresis)
 * and place well-distributed tracking dots along the detected edges.
 */
export function detectEdgesInCircle(
  gray: Float32Array,
  w: number,
  h: number,
  quad: ArrayLike<number>,
  opts: EdgeDetectOptions,
): EdgeDetectResult {
  const empty: EdgeDetectResult = {
    dots: new Float32Array(0),
    normals: new Float32Array(0),
    weights: new Float32Array(0),
    edgePixels: new Float32Array(0),
    edgeCount: 0,
    quality: 0,
  };
  if (opts.maxDots <= 0 || w < 8 || h < 8) return empty;

  const cx = quadCx(quad);
  const cy = quadCy(quad);
  const qSize = quadSize(quad);

  const minX = Math.min(quad[0], quad[2], quad[4], quad[6]);
  const maxX = Math.max(quad[0], quad[2], quad[4], quad[6]);
  const minY = Math.min(quad[1], quad[3], quad[5], quad[7]);
  const maxY = Math.max(quad[1], quad[3], quad[5], quad[7]);

  const x0 = Math.max(2, Math.floor(minX));
  const y0 = Math.max(2, Math.floor(minY));
  const x1 = Math.min(w - 3, Math.ceil(maxX));
  const y1 = Math.min(h - 3, Math.ceil(maxY));
  const bw = x1 - x0 + 1;
  const bh = y1 - y0 + 1;
  if (bw < 5 || bh < 5) return empty;

  // 1. Fast 3×3 Gaussian smoothing inside ROI (+1px border so gradients stay clean)
  const sw = bw + 2;
  const sh = bh + 2;
  const smooth = new Float32Array(sw * sh);
  for (let ly = 0; ly < sh; ly++) {
    const gy = y0 - 1 + ly;
    const r0 = (gy - 1) * w;
    const r1 = gy * w;
    const r2 = (gy + 1) * w;
    const dst = ly * sw;
    for (let lx = 0; lx < sw; lx++) {
      const gx = x0 - 1 + lx;
      smooth[dst + lx] =
        (gray[r0 + gx - 1] +
          2 * gray[r0 + gx] +
          gray[r0 + gx + 1] +
          2 * gray[r1 + gx - 1] +
          4 * gray[r1 + gx] +
          2 * gray[r1 + gx + 1] +
          gray[r2 + gx - 1] +
          2 * gray[r2 + gx] +
          gray[r2 + gx + 1]) *
        0.0625;
    }
  }

  // 2. Scharr 3×3 gradients + magnitude + quantized orientation inside the circle
  const nPix = bw * bh;
  const gxBuf = new Float32Array(nPix);
  const gyBuf = new Float32Array(nPix);
  const magBuf = new Float32Array(nPix);
  const dirBuf = new Uint8Array(nPix);
  const inCircle = new Uint8Array(nPix);

  const shrink = opts.shrink ?? 0.06;
  const invShrink = shrink > 0 && shrink < 0.45 ? 1 / (1 - shrink) : 1;

  for (let ly = 0; ly < bh; ly++) {
    const sy = ly + 1;
    const sRow = sy * sw;
    const oRow = ly * bw;
    const py = y0 + ly;
    for (let lx = 0; lx < bw; lx++) {
      const px = x0 + lx;
      // Full ellipse test (used for edge contour & NMS neighborhood)
      if (!pointInEllipse(quad, px, py)) continue;
      const sx = lx + 1;
      const idx = oRow + lx;
      const i = sRow + sx;
      const tl = smooth[i - sw - 1];
      const tr = smooth[i - sw + 1];
      const bl = smooth[i + sw - 1];
      const br = smooth[i + sw + 1];
      const gx = (3 * (tr + br - tl - bl) + 10 * (smooth[i + 1] - smooth[i - 1])) * 0.03125;
      const gy = (3 * (bl + br - tl - tr) + 10 * (smooth[i + sw] - smooth[i - sw])) * 0.03125;
      const mag = Math.hypot(gx, gy);
      gxBuf[idx] = gx;
      gyBuf[idx] = gy;
      magBuf[idx] = mag;
      inCircle[idx] = 1;

      // Quantize gradient normal into 4 sectors: 0 (0°), 1 (45°), 2 (90°), 3 (135°)
      const ax = Math.abs(gx);
      const ay = Math.abs(gy);
      if (ay <= ax * 0.41421356) dirBuf[idx] = 0;
      else if (ax <= ay * 0.41421356) dirBuf[idx] = 2;
      else dirBuf[idx] = (gx > 0) === (gy > 0) ? 1 : 3;
    }
  }

  // 3. Canny Non-Maximum Suppression (1px ridge thinning)
  const nms = new Float32Array(nPix);
  let maxMag = 0;
  let sumMag = 0;
  let nzCount = 0;

  for (let ly = 1; ly < bh - 1; ly++) {
    const row = ly * bw;
    for (let lx = 1; lx < bw - 1; lx++) {
      const idx = row + lx;
      if (!inCircle[idx]) continue;
      const m = magBuf[idx];
      if (m < 2.5) continue;
      const d = dirBuf[idx];
      let m1 = 0;
      let m2 = 0;
      if (d === 0) {
        m1 = magBuf[idx - 1];
        m2 = magBuf[idx + 1];
      } else if (d === 2) {
        m1 = magBuf[idx - bw];
        m2 = magBuf[idx + bw];
      } else if (d === 1) {
        m1 = magBuf[idx - bw - 1];
        m2 = magBuf[idx + bw + 1];
      } else {
        m1 = magBuf[idx - bw + 1];
        m2 = magBuf[idx + bw - 1];
      }
      if (m >= m1 && m >= m2) {
        nms[idx] = m;
        if (m > maxMag) maxMag = m;
        sumMag += m;
        nzCount++;
      }
    }
  }

  if (maxMag < 4.0 || nzCount === 0) return empty;

  // 4. Adaptive hysteresis thresholding controlled by sensitivity (10..100)
  const sens = Math.min(100, Math.max(10, opts.sensitivity));
  const tNorm = (sens - 10) / 90; // 0 = strict, 1 = sensitive
  const meanNms = sumMag / nzCount;
  // Blend peak-relative and mean-relative threshold so one bright specular glint doesn't hide real edges
  const peakFrac = 0.40 - 0.30 * tNorm; // 0.40 down to 0.10
  const meanMult = 2.0 - 1.15 * tNorm; // 2.0 down to 0.85
  const highThr = Math.max(4.8, Math.min(maxMag * peakFrac, meanNms * meanMult + 3.5));
  const lowThr = Math.max(2.8, highThr * 0.44);

  // 0 = none, 1 = weak, 2 = strong edge
  const edgeState = new Uint8Array(nPix);
  const queue = new Int32Array(nPix);
  let qHead = 0;
  let qTail = 0;

  for (let ly = 1; ly < bh - 1; ly++) {
    const row = ly * bw;
    for (let lx = 1; lx < bw - 1; lx++) {
      const idx = row + lx;
      const v = nms[idx];
      if (v >= highThr) {
        edgeState[idx] = 2;
        queue[qTail++] = idx;
      } else if (v >= lowThr) {
        edgeState[idx] = 1;
      }
    }
  }

  // 8-connected hysteresis trace
  while (qHead < qTail) {
    const cur = queue[qHead++];
    const lx = cur % bw;
    const ly = (cur - lx) / bw;
    for (let dy = -1; dy <= 1; dy++) {
      const ny = ly + dy;
      if (ny < 1 || ny >= bh - 1) continue;
      const nRow = ny * bw;
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = lx + dx;
        if (nx < 1 || nx >= bw - 1) continue;
        const nIdx = nRow + nx;
        if (edgeState[nIdx] === 1) {
          edgeState[nIdx] = 2;
          queue[qTail++] = nIdx;
        }
      }
    }
  }

  // Collect all strong edge pixels + score each candidate for dot placement
  const allEdges: number[] = [];
  const sectors: { idx: number; x: number; y: number; score: number; mag: number }[][] = [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
  ];

  for (let ly = 1; ly < bh - 1; ly++) {
    const row = ly * bw;
    const py = y0 + ly;
    for (let lx = 1; lx < bw - 1; lx++) {
      const idx = row + lx;
      if (edgeState[idx] !== 2) continue;
      const px = x0 + lx;
      allEdges.push(px, py);

      // Slight inward margin for tracking dots so LK patches stay on the target
      if (invShrink > 1) {
        const tx = cx + (px - cx) * invShrink;
        const ty = cy + (py - cy) * invShrink;
        if (!pointInEllipse(quad, tx, ty)) continue;
      }
      if (opts.accept && !opts.accept(px, py)) continue;

      // Local 3×3 structure tensor minor eigenvalue boosts corners & curved edge junctions
      let sxx = 0;
      let sxy = 0;
      let syy = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const rr = (ly + dy) * bw + lx;
        for (let dx = -1; dx <= 1; dx++) {
          const g1 = gxBuf[rr + dx];
          const g2 = gyBuf[rr + dx];
          sxx += g1 * g1;
          sxy += g1 * g2;
          syy += g2 * g2;
        }
      }
      const lambdaMin = Math.max(0, (sxx + syy - Math.sqrt((sxx - syy) * (sxx - syy) + 4 * sxy * sxy)) * 0.5);
      const mag = magBuf[idx];
      const score = mag * (1 + 0.28 * Math.sqrt(lambdaMin));

      const ang = Math.atan2(py - cy, px - cx);
      const sec = ((Math.floor(((ang + Math.PI) / (2 * Math.PI)) * 8) % 8) + 8) & 7;
      sectors[sec].push({ idx, x: px, y: py, score, mag });
    }
  }

  const edgeCount = allEdges.length >> 1;
  if (edgeCount === 0) return empty;

  // Subsample edge pixels for live visual overlay if requested
  let edgePixels = new Float32Array(0);
  if (opts.maxPreviewEdges && opts.maxPreviewEdges > 0) {
    const stride = Math.max(1, Math.ceil(edgeCount / opts.maxPreviewEdges));
    const sampled: number[] = [];
    for (let i = 0; i < edgeCount; i += stride) {
      sampled.push(allEdges[i * 2], allEdges[i * 2 + 1]);
    }
    edgePixels = new Float32Array(sampled);
  }

  // 5. Sector-balanced spatial NMS dot placement along detected edges
  for (let s = 0; s < 8; s++) sectors[s].sort((a, b) => b.score - a.score);

  const circlePerimeter = Math.PI * qSize;
  const autoMinDist = Math.max(3.5, Math.min(22, (circlePerimeter / Math.max(8, opts.maxDots)) * 0.72));
  const minDist = opts.minDist ?? autoMinDist;
  const d2Min = minDist * minDist;
  const cell = Math.max(1, minDist);
  const gw = Math.ceil(w / cell) + 1;
  const gh = Math.ceil(h / cell) + 1;
  const grid = new Map<number, number[]>();

  const put = (x: number, y: number) => {
    const key = Math.floor(y / cell) * gw + Math.floor(x / cell);
    const list = grid.get(key);
    if (list) list.push(x, y);
    else grid.set(key, [x, y]);
  };

  const isFree = (x: number, y: number) => {
    const gx = Math.floor(x / cell);
    const gy = Math.floor(y / cell);
    for (let yy = gy - 1; yy <= gy + 1; yy++) {
      if (yy < 0 || yy >= gh) continue;
      for (let xx = gx - 1; xx <= gx + 1; xx++) {
        const list = grid.get(yy * gw + xx);
        if (!list) continue;
        for (let k = 0; k < list.length; k += 2) {
          const dx = list[k] - x;
          const dy = list[k + 1] - y;
          if (dx * dx + dy * dy < d2Min) return false;
        }
      }
    }
    return true;
  };

  if (opts.existing) {
    const ne = opts.nExisting ?? opts.existing.length >> 1;
    for (let k = 0; k < ne; k++) put(opts.existing[k * 2], opts.existing[k * 2 + 1]);
  }

  const pickedDots: number[] = [];
  const pickedNormals: number[] = [];
  const pickedMags: number[] = [];
  const cursors = new Int32Array(8);

  // Round-robin across the 8 angular sectors so dots hug the full contour collectively
  let progressed = true;
  while (pickedMags.length < opts.maxDots && progressed) {
    progressed = false;
    for (let s = 0; s < 8 && pickedMags.length < opts.maxDots; s++) {
      const list = sectors[s];
      while (cursors[s] < list.length) {
        const cand = list[cursors[s]++];
        if (!isFree(cand.x, cand.y)) continue;
        put(cand.x, cand.y);
        pickedDots.push(cand.x, cand.y);
        const invM = cand.mag > 1e-6 ? 1 / cand.mag : 0;
        pickedNormals.push(gxBuf[cand.idx] * invM, gyBuf[cand.idx] * invM);
        pickedMags.push(cand.mag);
        progressed = true;
        break;
      }
    }
  }

  const count = pickedMags.length;
  const weights = new Float32Array(count);
  if (count > 0) {
    let meanW = 0;
    for (let i = 0; i < count; i++) meanW += pickedMags[i];
    meanW = Math.max(1e-6, meanW / count);
    for (let i = 0; i < count; i++) {
      weights[i] = Math.min(1.6, Math.max(0.5, pickedMags[i] / meanW));
    }
  }

  // Quality combines dot coverage and average edge contrast
  const dotCoverage = Math.min(1, count / Math.max(6, Math.min(opts.maxDots, 16)));
  const contrastScore = Math.min(1, maxMag / 28);
  const quality = dotCoverage * (0.35 + 0.65 * contrastScore);

  return {
    dots: new Float32Array(pickedDots),
    normals: new Float32Array(pickedNormals),
    weights,
    edgePixels,
    edgeCount,
    quality,
  };
}

// ───────────────────────────── EdgeTracker worker engine ─────────────────────────────

export interface EdgeTrackerOptions {
  model: TrackMotionModel;
  followSize: boolean;
  sensitivity: number;
  maxDots: number;
  mask: TrackMask;
}

interface CroppedPyr {
  ox: number;
  oy: number;
  w: number;
  h: number;
  pyr: Pyramid;
}

const TPL_SIZE = 20;

/**
 * Huber-weighted IRLS similarity fit from reference offsets `src` to current positions `dst`.
 * Uses all inlier edge dots collectively so individual jitter cancels out (~1/√K).
 */
function fitWeightedSimilarityIRLS(
  src: Float32Array,
  dst: Float32Array,
  weights: Float32Array,
  idx: Int32Array,
  count: number,
  huberDelta: number,
): Mat3 | null {
  if (count < 2) return null;
  const rw = new Float64Array(count);
  for (let k = 0; k < count; k++) rw[k] = weights[idx[k]] || 1;

  let H: Mat3 | null = null;
  for (let iter = 0; iter < 3; iter++) {
    let wsum = 0;
    let mx = 0;
    let my = 0;
    let ux = 0;
    let uy = 0;
    for (let k = 0; k < count; k++) {
      const i = idx[k] * 2;
      const w = rw[k];
      wsum += w;
      mx += w * src[i];
      my += w * src[i + 1];
      ux += w * dst[i];
      uy += w * dst[i + 1];
    }
    if (wsum < 1e-9) return H;
    mx /= wsum;
    my /= wsum;
    ux /= wsum;
    uy /= wsum;

    let a = 0;
    let b = 0;
    let d = 0;
    for (let k = 0; k < count; k++) {
      const i = idx[k] * 2;
      const w = rw[k];
      const xs = src[i] - mx;
      const ys = src[i + 1] - my;
      const xd = dst[i] - ux;
      const yd = dst[i + 1] - uy;
      a += w * (xs * xd + ys * yd);
      b += w * (xs * yd - ys * xd);
      d += w * (xs * xs + ys * ys);
    }
    if (d < 1e-9) return H;
    const c = a / d;
    const s = b / d;
    const sc2 = c * c + s * s;
    if (sc2 < 0.04 || sc2 > 25) return H;
    H = new Float64Array([c, -s, ux - (c * mx - s * my), s, c, uy - (s * mx + c * my), 0, 0, 1]);

    if (iter < 2) {
      for (let k = 0; k < count; k++) {
        const i = idx[k] * 2;
        const px = H[0] * src[i] + H[1] * src[i + 1] + H[2];
        const py = H[3] * src[i] + H[4] * src[i + 1] + H[5];
        const r = Math.hypot(px - dst[i], py - dst[i + 1]);
        const huber = r <= huberDelta ? 1 : huberDelta / r;
        rw[k] = (weights[idx[k]] || 1) * huber;
      }
    }
  }
  return H;
}

/** Huber-weighted IRLS translation fit across all collective inlier edge dots. */
function fitWeightedTranslationIRLS(
  src: Float32Array,
  dst: Float32Array,
  weights: Float32Array,
  idx: Int32Array,
  count: number,
  huberDelta: number,
): Mat3 | null {
  if (count < 1) return null;
  const rw = new Float64Array(count);
  for (let k = 0; k < count; k++) rw[k] = weights[idx[k]] || 1;
  let tx = 0;
  let ty = 0;
  for (let iter = 0; iter < 3; iter++) {
    let wsum = 0;
    let sx = 0;
    let sy = 0;
    for (let k = 0; k < count; k++) {
      const i = idx[k] * 2;
      const w = rw[k];
      wsum += w;
      sx += w * (dst[i] - src[i]);
      sy += w * (dst[i + 1] - src[i + 1]);
    }
    if (wsum < 1e-9) break;
    tx = sx / wsum;
    ty = sy / wsum;
    if (iter < 2) {
      for (let k = 0; k < count; k++) {
        const i = idx[k] * 2;
        const r = Math.hypot(src[i] + tx - dst[i], src[i + 1] + ty - dst[i + 1]);
        const huber = r <= huberDelta ? 1 : huberDelta / r;
        rw[k] = (weights[idx[k]] || 1) * huber;
      }
    }
  }
  return new Float64Array([1, 0, tx, 0, 1, ty, 0, 0, 1]);
}

export class EdgeTracker {
  private o: EdgeTrackerOptions;
  private prevCrop!: CroppedPyr;
  private frameW = 0;
  private frameH = 0;
  private pts = new Float32Array(0);
  /** Reference-frame coordinates of each edge dot (absolute pose → zero cumulative drift) */
  private off = new Float32Array(0);
  /** Reference unit gradient normals of each edge dot */
  private normals = new Float32Array(0);
  /** Edge-strength weights for collective IRLS voting */
  private weights = new Float32Array(0);
  private n = 0;
  /** Reference frame → current frame transform */
  private M: Mat3 = identity();
  private ref = new Float64Array(8);
  private motion = new Float64Array(8);
  private roi = new Float64Array(8);
  /** Smoothed collective velocity (px/frame) for warm-starting LK */
  private vx = 0;
  private vy = 0;
  private tpl: Float32Array | null = null;
  private searching = 0;
  private unlike = 0;
  private frameNo = 0;
  private lk!: LKParams;
  private fbMax = 1.2;
  private thr = 1.8;
  private rand = mulberry32(0xed6e7ac);

  constructor(o: EdgeTrackerOptions) {
    this.o = o;
  }

  /** Worker skips full-frame pyramid construction; EdgeTracker builds only a tiny local ROI crop pyramid! */
  get pyramidLevels() {
    return 0;
  }

  /** Initial edge dots in frame pixel coords [x, y, status, ...] for immediate frame-0 feedback. */
  initialPoints(): Float32Array {
    const out = new Float32Array(this.n * 3);
    for (let i = 0; i < this.n; i++) {
      out[i * 3] = this.pts[i * 2];
      out[i * 3 + 1] = this.pts[i * 2 + 1];
      out[i * 3 + 2] = POINT_INLIER;
    }
    return out;
  }

  private maskOk(mask: MaskImage | null, x: number, y: number): boolean {
    if (!mask || this.o.mask === 'none') return true;
    const mx = Math.min(mask.w - 1, Math.max(0, Math.floor((x / this.frameW) * mask.w)));
    const my = Math.min(mask.h - 1, Math.max(0, Math.floor((y / this.frameH) * mask.h)));
    const v = mask.data[my * mask.w + mx];
    return this.o.mask === 'subject' ? v > 128 : v < 96;
  }

  /** Extract a fast local pyramid covering only the tracking circle plus a generous motion margin. */
  private makeCrop(f: TrackerFrame, cx: number, cy: number, qSize: number, extraMargin = 0): CroppedPyr {
    const half = Math.ceil(Math.max(qSize * 0.95 + 42, 68) + extraMargin);
    const x0 = Math.max(0, Math.floor(cx - half));
    const y0 = Math.max(0, Math.floor(cy - half));
    const x1 = Math.min(f.w, Math.ceil(cx + half));
    const y1 = Math.min(f.h, Math.ceil(cy + half));
    const cw = Math.max(24, x1 - x0);
    const ch = Math.max(24, y1 - y0);
    const ox = Math.min(x0, Math.max(0, f.w - cw));
    const oy = Math.min(y0, Math.max(0, f.h - ch));
    const gray = f.rgba
      ? grayFromRGBACrop(f.rgba, f.w, ox, oy, cw, ch)
      : (() => {
          const out = new Float32Array(cw * ch);
          for (let y = 0; y < ch; y++) out.set(f.gray.subarray((oy + y) * f.w + ox, (oy + y) * f.w + ox + cw), y * cw);
          return out;
        })();
    const levels = qSize > 95 ? 3 : 2;
    return { ox, oy, w: cw, h: ch, pyr: buildPyramid(gray, cw, ch, levels) };
  }

  private toCropQuad(crop: CroppedPyr, q: ArrayLike<number>): Float64Array {
    const out = new Float64Array(8);
    for (let i = 0; i < 4; i++) {
      out[i * 2] = q[i * 2] - crop.ox;
      out[i * 2 + 1] = q[i * 2 + 1] - crop.oy;
    }
    return out;
  }

  private seedEdges(crop: CroppedPyr, mask: MaskImage | null, maxAdd: number) {
    if (maxAdd <= 0) return;
    const localQuad = this.toCropQuad(crop, this.roi);
    const existingLocal = new Float32Array(this.n * 2);
    for (let i = 0; i < this.n; i++) {
      existingLocal[i * 2] = this.pts[i * 2] - crop.ox;
      existingLocal[i * 2 + 1] = this.pts[i * 2 + 1] - crop.oy;
    }

    const run = (useMask: boolean) =>
      detectEdgesInCircle(crop.pyr[0].img, crop.w, crop.h, localQuad, {
        maxDots: maxAdd,
        sensitivity: this.o.sensitivity,
        existing: existingLocal,
        nExisting: this.n,
        accept: useMask && mask && this.o.mask !== 'none' ? (lx, ly) => this.maskOk(mask, lx + crop.ox, ly + crop.oy) : undefined,
        shrink: 0.05,
      });

    let res = run(true);
    if ((res.dots.length >> 1) < 4 && mask && this.o.mask !== 'none') {
      const all = run(false);
      if (all.dots.length > res.dots.length * 2) res = all;
    }

    // If sensitivity was very strict and yielded very few dots on a softer area, relax once automatically
    if (this.n === 0 && (res.dots.length >> 1) < 4 && this.o.sensitivity < 85) {
      res = detectEdgesInCircle(crop.pyr[0].img, crop.w, crop.h, localQuad, {
        maxDots: maxAdd,
        sensitivity: Math.min(95, this.o.sensitivity + 28),
        existing: existingLocal,
        nExisting: this.n,
        shrink: 0.04,
      });
    }

    const m = res.dots.length >> 1;
    if (!m) return;

    const inv = invert(this.M) ?? identity();
    const pts = new Float32Array((this.n + m) * 2);
    const off = new Float32Array((this.n + m) * 2);
    const normals = new Float32Array((this.n + m) * 2);
    const weights = new Float32Array(this.n + m);

    pts.set(this.pts.subarray(0, this.n * 2));
    off.set(this.off.subarray(0, this.n * 2));
    normals.set(this.normals.subarray(0, this.n * 2));
    weights.set(this.weights.subarray(0, this.n));

    for (let i = 0; i < m; i++) {
      const gx = res.dots[i * 2] + crop.ox;
      const gy = res.dots[i * 2 + 1] + crop.oy;
      pts[(this.n + i) * 2] = gx;
      pts[(this.n + i) * 2 + 1] = gy;
      off[(this.n + i) * 2] = applyX(inv, gx, gy);
      off[(this.n + i) * 2 + 1] = applyY(inv, gx, gy);
      normals[(this.n + i) * 2] = res.normals[i * 2];
      normals[(this.n + i) * 2 + 1] = res.normals[i * 2 + 1];
      weights[this.n + i] = res.weights[i];
    }

    this.pts = pts;
    this.off = off;
    this.normals = normals;
    this.weights = weights;
    this.n += m;
  }

  private extractCropPatch(crop: CroppedPyr, q: ArrayLike<number>, out?: Float32Array) {
    const localQ = this.toCropQuad(crop, q);
    const size = quadSize(localQ);
    const lvl = Math.max(0, Math.min(crop.pyr.length - 1, Math.round(Math.log2(Math.max(1, size / 28)))));
    const s = 1 / (1 << lvl);
    const sq = new Float64Array(8);
    for (let i = 0; i < 8; i++) sq[i] = localQ[i] * s;
    return { ...extractPatch(crop.pyr[lvl], sq, TPL_SIZE, out), lvl, sq };
  }

  init(f: TrackerFrame, quad: ArrayLike<number>): string | null {
    this.frameW = f.w;
    this.frameH = f.h;
    const k = Math.max(f.w, f.h) / 480;
    this.lk = { radius: 6, iterations: 10, epsilon: 0.02, minEigen: 0.85 };
    this.fbMax = 1.15 * Math.max(1, k);
    this.thr = 1.65 * Math.max(1, k);

    for (let i = 0; i < 8; i++) this.ref[i] = this.motion[i] = this.roi[i] = quad[i];
    this.M = identity();
    this.pts = new Float32Array(0);
    this.off = new Float32Array(0);
    this.normals = new Float32Array(0);
    this.weights = new Float32Array(0);
    this.n = 0;
    this.vx = 0;
    this.vy = 0;
    this.searching = 0;
    this.unlike = 0;
    this.frameNo = 0;

    const cx = quadCx(this.roi);
    const cy = quadCy(this.roi);
    const qSize = quadSize(this.roi);
    const crop = this.makeCrop(f, cx, cy, qSize);

    this.seedEdges(crop, f.mask, this.o.maxDots);
    if (this.n < 2) return 'low-detail';

    const t = this.extractCropPatch(crop, this.roi);
    this.tpl = t.std > 1.5 ? t.patch : null;
    this.prevCrop = crop;
    return null;
  }

  private solve(currCrop: CroppedPyr, shift: [number, number] | null) {
    const n = this.n;
    const A = this.prevCrop;
    const B = currCrop;
    const srcLocal = new Float32Array(n * 2);
    const guessLocal = new Float32Array(n * 2);
    const fwdLocal = new Float32Array(n * 2);
    const backLocal = new Float32Array(n * 2);
    const st = new Uint8Array(n);
    const st2 = new Uint8Array(n);
    const err = new Float32Array(n);
    const err2 = new Float32Array(n);
    const status = new Uint8Array(n).fill(POINT_LOST);
    const fwdGlobal = new Float32Array(n * 2);
    const use = new Int32Array(n);
    let nUse = 0;

    const predDx = shift ? shift[0] : this.vx;
    const predDy = shift ? shift[1] : this.vy;

    for (let i = 0; i < n; i++) {
      const gx = this.pts[i * 2];
      const gy = this.pts[i * 2 + 1];
      srcLocal[i * 2] = gx - A.ox;
      srcLocal[i * 2 + 1] = gy - A.oy;
      guessLocal[i * 2] = gx + predDx - B.ox;
      guessLocal[i * 2 + 1] = gy + predDy - B.oy;
    }

    const loose = this.searching > 0 ? 1.85 : 1;
    if (n > 0) {
      trackLK(A.pyr, B.pyr, srcLocal, n, fwdLocal, st, err, this.lk, guessLocal);
      trackLK(B.pyr, A.pyr, fwdLocal, n, backLocal, st2, err2, this.lk, srcLocal);

      ensureGradients(B.pyr[0]);
      const bgx = B.pyr[0].gx!;
      const bgy = B.pyr[0].gy!;
      const bw = B.w;
      const bh = B.h;
      const fb2 = this.fbMax * this.fbMax * loose * loose;

      for (let i = 0; i < n; i++) {
        if (!st[i] || !st2[i]) continue;
        const lx = fwdLocal[i * 2];
        const ly = fwdLocal[i * 2 + 1];
        fwdGlobal[i * 2] = lx + B.ox;
        fwdGlobal[i * 2 + 1] = ly + B.oy;
        status[i] = POINT_OUTLIER;

        const bdx = backLocal[i * 2] - srcLocal[i * 2];
        const bdy = backLocal[i * 2 + 1] - srcLocal[i * 2 + 1];
        if (bdx * bdx + bdy * bdy > fb2 || err[i] > 40) continue;

        // Verify the dot still sits on an edge (prevents sliding onto flat background)
        const ix = Math.round(lx);
        const iy = Math.round(ly);
        if (ix >= 1 && ix < bw - 1 && iy >= 1 && iy < bh - 1) {
          const gi = iy * bw + ix;
          const gMag = Math.hypot(bgx[gi], bgy[gi]);
          if (gMag < 1.6) continue;
        }
        use[nUse++] = i;
      }
    }

    let ok = false;
    let abs: Mat3 | null = null;
    let step: Mat3 | null = null;
    let inliers = 0;
    let geo = 0;

    if (nUse >= 1) {
      const prevInv = invert(this.M) ?? identity();
      const rx = quadCx(this.roi);
      const ry = quadCy(this.roi);
      const useSimilarity = (this.o.followSize || this.o.model !== 'translation') && nUse >= 4;

      if (useSimilarity) {
        const res = ransac('similarity', this.off, fwdGlobal, use, nUse, this.thr * 1.45 * loose, 72, this.rand);
        if (res && res.count >= Math.max(3, Math.ceil(0.32 * nUse)) && res.count >= Math.ceil(0.18 * n)) {
          const inlierSet = new Int32Array(res.count);
          let q = 0;
          for (let k = 0; k < nUse; k++) {
            if (res.inlier[use[k]]) inlierSet[q++] = use[k];
          }
          // Collective Huber IRLS refinement across all agreeing edge dots
          const refined = fitWeightedSimilarityIRLS(this.off, fwdGlobal, this.weights, inlierSet, q, this.thr * 0.9) ?? res.H;
          const stp = mul(refined, prevInv);
          const loc = localSimilarity(stp, rx, ry);
          if (loc.scale >= 0.72 && loc.scale <= 1.38 && Number.isFinite(applyX(stp, rx, ry))) {
            abs = refined;
            step = stp;
            inliers = q;
            for (let k = 0; k < q; k++) status[inlierSet[k]] = POINT_INLIER;
            geo = q / nUse;
            ok = true;
          }
        }
      }

      // Fallback or pure translation mode: collective median + MAD + Huber IRLS across all edge dots
      if (!ok && nUse >= 1) {
        const dxs = new Float64Array(nUse);
        const dys = new Float64Array(nUse);
        for (let k = 0; k < nUse; k++) {
          const i = use[k] * 2;
          dxs[k] = fwdGlobal[i] - this.off[i];
          dys[k] = fwdGlobal[i + 1] - this.off[i + 1];
        }
        const mx = median(dxs, nUse);
        const my = median(dys, nUse);
        const devs = new Float64Array(nUse);
        for (let k = 0; k < nUse; k++) devs[k] = Math.hypot(dxs[k] - mx, dys[k] - my);
        const mad = median(devs, nUse);
        const gate = Math.max(this.thr * 1.35 * loose, mad * 2.6 + 0.8);

        const inlierSet = new Int32Array(nUse);
        let q = 0;
        for (let k = 0; k < nUse; k++) {
          if (devs[k] <= gate) inlierSet[q++] = use[k];
        }
        if (q >= Math.max(nUse >= 3 ? 2 : 1, Math.ceil(0.30 * nUse)) && q >= Math.ceil(0.16 * n)) {
          const refined = fitWeightedTranslationIRLS(this.off, fwdGlobal, this.weights, inlierSet, q, this.thr * 0.85);
          if (refined) {
            // Preserve current scale in abs if we temporarily fell back from similarity
            const stp = mul(refined, prevInv);
            if (Number.isFinite(applyX(stp, rx, ry))) {
              abs = refined;
              step = stp;
              inliers = q;
              for (let k = 0; k < q; k++) status[inlierSet[k]] = POINT_INLIER;
              geo = q / nUse;
              ok = true;
            }
          }
        }
      }

      // Fast edge-guided rescue for dots that missed LK
      if (ok && abs) {
        const missIdx: number[] = [];
        for (let i = 0; i < n && missIdx.length < 24; i++) {
          if (status[i] !== POINT_INLIER) missIdx.push(i);
        }
        const cnt = missIdx.length;
        if (cnt > 0) {
          const rSrc = new Float32Array(cnt * 2);
          const rGuess = new Float32Array(cnt * 2);
          const rOut = new Float32Array(cnt * 2);
          const rSt = new Uint8Array(cnt);
          const rErr = new Float32Array(cnt);
          for (let k = 0; k < cnt; k++) {
            const i = missIdx[k];
            rSrc[k * 2] = this.pts[i * 2] - A.ox;
            rSrc[k * 2 + 1] = this.pts[i * 2 + 1] - A.oy;
            rGuess[k * 2] = applyX(abs, this.off[i * 2], this.off[i * 2 + 1]) - B.ox;
            rGuess[k * 2 + 1] = applyY(abs, this.off[i * 2], this.off[i * 2 + 1]) - B.oy;
          }
          trackLK([A.pyr[0]], [B.pyr[0]], rSrc, cnt, rOut, rSt, rErr, { ...this.lk, iterations: 6 }, rGuess);
          for (let k = 0; k < cnt; k++) {
            if (!rSt[k] || rErr[k] > 24) continue;
            if (Math.hypot(rOut[k * 2] - rGuess[k * 2], rOut[k * 2 + 1] - rGuess[k * 2 + 1]) > this.thr * 0.9) continue;
            const i = missIdx[k];
            fwdGlobal[i * 2] = rOut[k * 2] + B.ox;
            fwdGlobal[i * 2 + 1] = rOut[k * 2 + 1] + B.oy;
            status[i] = POINT_RESCUED;
          }
        }
      }
    }

    const points = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const moved = status[i] !== POINT_LOST;
      points[i * 3] = moved ? fwdGlobal[i * 2] : this.pts[i * 2];
      points[i * 3 + 1] = moved ? fwdGlobal[i * 2 + 1] : this.pts[i * 2 + 1];
      points[i * 3 + 2] = status[i];
    }

    return { ok, abs, step, fwdGlobal, status, inliers, geo, points };
  }

  private accept(currCrop: CroppedPyr, mask: MaskImage | null, m: ReturnType<EdgeTracker['solve']>) {
    const oldCx = quadCx(this.roi);
    const oldCy = quadCy(this.roi);

    if (m.abs && m.step) {
      this.M = m.abs;
      transformQuad(m.abs, this.ref, this.motion);
      const nx = applyX(m.step, oldCx, oldCy);
      const ny = applyY(m.step, oldCx, oldCy);
      const loc = localSimilarity(m.step, oldCx, oldCy);
      const s = this.o.followSize ? loc.scale : 1;
      for (let i = 0; i < 4; i++) {
        const dx = this.roi[i * 2] - oldCx;
        const dy = this.roi[i * 2 + 1] - oldCy;
        this.roi[i * 2] = nx + dx * s;
        this.roi[i * 2 + 1] = ny + dy * s;
      }
      // Exponentially smoothed velocity for next-frame warm start
      const instVx = nx - oldCx;
      const instVy = ny - oldCy;
      this.vx = this.vx * 0.45 + instVx * 0.55;
      this.vy = this.vy * 0.45 + instVy * 0.55;
    }

    // Keep surviving collective edge dots that remain inside the tracking circle
    let k = 0;
    for (let i = 0; i < this.n; i++) {
      if (m.status[i] !== POINT_INLIER && m.status[i] !== POINT_RESCUED) continue;
      const x = m.fwdGlobal[i * 2];
      const y = m.fwdGlobal[i * 2 + 1];
      if (!pointInEllipse(this.roi, x, y)) continue;
      this.pts[k * 2] = x;
      this.pts[k * 2 + 1] = y;
      this.off[k * 2] = this.off[i * 2];
      this.off[k * 2 + 1] = this.off[i * 2 + 1];
      this.normals[k * 2] = this.normals[i * 2];
      this.normals[k * 2 + 1] = this.normals[i * 2 + 1];
      this.weights[k] = this.weights[i];
      k++;
    }
    this.n = k;
    this.prevCrop = currCrop;
    this.searching = 0;

    // Replenish edge dots along uncovered edges without introducing drift
    if (k < Math.ceil(this.o.maxDots * 0.72) && (this.frameNo % 4 === 0 || k < Math.ceil(this.o.maxDots * 0.45))) {
      this.seedEdges(currCrop, mask, this.o.maxDots - k);
    }
  }

  private shiftAll(dx: number, dy: number) {
    for (let i = 0; i < 4; i++) {
      this.roi[i * 2] += dx;
      this.roi[i * 2 + 1] += dy;
      this.motion[i * 2] += dx;
      this.motion[i * 2 + 1] += dy;
    }
    this.M = mul(new Float64Array([1, 0, dx, 0, 1, dy, 0, 0, 1]), this.M);
  }

  private out(state: 0 | 1, conf: number, lost: boolean, points: Float32Array | null): StepOut {
    return { quad: Array.from(this.motion), roi: Array.from(this.roi), conf, state, lost, points };
  }

  step(f: TrackerFrame): StepOut {
    this.frameNo++;
    const cx = quadCx(this.roi);
    const cy = quadCy(this.roi);
    const qSize = quadSize(this.roi);
    const gone = cx < -0.04 * f.w || cy < -0.04 * f.h || cx > 1.04 * f.w || cy > 1.04 * f.h;

    // Crop around predicted location (widens gracefully when searching)
    const searchPad = this.searching > 0 ? Math.min(120, this.searching * 8) : 0;
    const currCrop = this.makeCrop(f, cx + this.vx, cy + this.vy, qSize, searchPad);

    let m = this.solve(currCrop, null);

    if (!m.ok) {
      this.searching++;
      this.vx *= 0.5;
      this.vy *= 0.5;

      if (this.tpl && !gone) {
        const t = this.extractCropPatch(currCrop, this.motion);
        const size = quadSize(t.sq);
        const radius = Math.min(Math.max(currCrop.pyr[t.lvl].w, currCrop.pyr[t.lvl].h) * 0.38, size * (0.75 + 0.05 * this.searching) + 6);
        const hit = searchTemplate(currCrop.pyr[t.lvl], t.sq, this.tpl, TPL_SIZE, radius, Math.max(2, size / 10));
        if (hit.score >= 0.62) {
          const sc = 1 << t.lvl;
          const again = this.solve(currCrop, [hit.dx * sc, hit.dy * sc]);
          if (again.ok) {
            m = again;
          } else {
            this.shiftAll(hit.dx * sc, hit.dy * sc);
            this.n = 0;
            this.seedEdges(currCrop, f.mask, this.o.maxDots);
            this.prevCrop = currCrop;
            this.searching = 0;
            return this.out(0, Math.min(1, hit.score * 0.8), false, this.initialPoints());
          }
        }
      }
      if (!m.ok) return this.out(1, 0, gone || this.searching > 45, m.points);
    }

    this.accept(currCrop, f.mask, m);

    let conf = Math.min(1, 0.35 + 0.65 * Math.min(1, m.geo * 1.15) * Math.min(1, m.inliers / 8));
    if (this.tpl) {
      const t = this.extractCropPatch(currCrop, this.motion);
      const ncc = t.std > 1.5 ? correlate(t.patch, this.tpl) : 0;
      conf = 0.68 * conf + 0.32 * Math.max(0, ncc);
      this.unlike = ncc < 0.05 ? this.unlike + 1 : 0;
      if (this.unlike >= 6) {
        this.unlike = 0;
        this.searching = 1;
        return this.out(1, 0, false, m.points);
      }
    }

    return this.out(0, conf, gone, m.points);
  }
}

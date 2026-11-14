// Point-based trackers.
//
//  • Flow      – pyramidal Lucas–Kanade with a forward-backward check, RANSAC motion-model fit
//                (translation / similarity / affine / perspective), patch rescue and bounded template recovery.
//  • Features  – a consensus tracker in the spirit of CMT: every keypoint remembers where it sits in the
//                object's own frame, pairwise medians give scale + rotation, and the points vote for the centre.
//                Absolute pose, so it does not accumulate drift.
//  • Frame     – dominant motion of the whole picture (camera tracking).
//  • Dual      – two independent point trackers whose baseline yields position, scale and orientation.
import type { TrackMask, TrackMotionModel, TrackRegionShape } from '../../types';
import { applyX, applyY, fitModel, identity, invert, localSimilarity, mul, median, MODEL_MIN, mulberry32, pointInEllipse, pointInQuad, quadCx, quadCy, quadSize, ransac, transformQuad, wrapAngle, type Mat3 } from './math';
import { correlate, detectCorners, extractPatch, searchTemplate, trackLK, type LKParams, type Pyramid } from './vision';

export const POINT_INLIER = 0;
export const POINT_OUTLIER = 1;
export const POINT_LOST = 2;
export const POINT_RESCUED = 3;
export const POINT_SEEDED = 4;
export const POINT_LANDMARK = 5;

export interface MaskImage {
  data: Uint8Array;
  w: number;
  h: number;
}

export interface TrackerFrame {
  w: number;
  h: number;
  gray: Float32Array;
  pyr: Pyramid;
  rgba: Uint8ClampedArray | null;
  mask: MaskImage | null;
}

export interface StepOut {
  /** recorded motion: the reference quad carried by the measured motion model (frame px) */
  quad: number[];
  /** the selector region as it follows the target (frame px) */
  roi: number[];
  conf: number;
  /** 0 tracked, 1 searching */
  state: 0 | 1;
  lost: boolean;
  /** x, y, status triples (frame px) */
  points: Float32Array | null;
}

export interface PointOptions {
  kind: 'flow' | 'features' | 'frame';
  model: TrackMotionModel;
  shape: TrackRegionShape;
  followSize: boolean;
  followRotation: boolean;
  cornerPin: boolean;
  advanced: boolean;
  mask: TrackMask;
}

interface Params {
  maxPts: number;
  levels: number;
  lk: LKParams;
  fbMax: number;
  thr: number;
  rescue: boolean;
  recover: boolean;
  searchLimit: number;
  reseedEvery: number;
  quality: number;
}

const TPL = 24;
const FALLBACK: TrackMotionModel[] = ['perspective', 'affine', 'similarity', 'translation'];
const NEED: Record<TrackMotionModel, number> = { perspective: 12, affine: 7, similarity: 3, translation: 1 };
const ITERS: Record<TrackMotionModel, number> = { perspective: 260, affine: 140, similarity: 80, translation: 30 };

export class PointTracker {
  private o: PointOptions;
  private P!: Params;
  private prev!: TrackerFrame;
  private pts = new Float32Array(0);
  /** features: each point's place in the object's own frame. flow / frame: its position in the reference frame. */
  private off = new Float32Array(0);
  /** flow / frame: absolute motion, reference frame → current frame */
  private M: Mat3 = identity();
  private n = 0;
  private roi = new Float64Array(8);
  private motion = new Float64Array(8);
  private ref = new Float64Array(8);
  private c0x = 0;
  private c0y = 0;
  private pose = { cx: 0, cy: 0, s: 1, th: 0 };
  private tpl: Float32Array | null = null;
  private searching = 0;
  private unlike = 0;
  private frameNo = 0;
  private rand = mulberry32(0x7ac4e7);

  constructor(o: PointOptions) {
    this.o = o;
  }

  private params(w: number, h: number): Params {
    const k = Math.max(w, h) / 480;
    const adv = this.o.advanced || this.o.kind === 'features';
    const frame = this.o.kind === 'frame';
    const extra = k > 1.25 ? 1 : 0;
    return adv
      ? { maxPts: frame ? 180 : 170, levels: 3 + extra, lk: { radius: frame ? 7 : 9, iterations: 20, epsilon: 0.01, minEigen: 1.2 }, fbMax: 0.9 * Math.max(1, k), thr: 1.4 * Math.max(1, k), rescue: this.o.kind !== 'features', recover: !frame, searchLimit: 60, reseedEvery: 5, quality: 0.012 }
      : { maxPts: frame ? 110 : 90, levels: 2 + extra, lk: { radius: 7, iterations: 12, epsilon: 0.03, minEigen: 1.2 }, fbMax: 1.4 * Math.max(1, k), thr: 2 * Math.max(1, k), rescue: false, recover: false, searchLimit: 15, reseedEvery: 8, quality: 0.02 };
  }

  get pyramidLevels() {
    return this.P ? this.P.levels : 3;
  }

  private maskOk(f: TrackerFrame, x: number, y: number): boolean {
    const m = f.mask;
    if (!m || this.o.mask === 'none') return true;
    const mx = Math.min(m.w - 1, Math.max(0, Math.floor((x / f.w) * m.w)));
    const my = Math.min(m.h - 1, Math.max(0, Math.floor((y / f.h) * m.h)));
    const v = m.data[my * m.w + mx];
    return this.o.mask === 'subject' ? v > 128 : v < 96;
  }

  /** `shrink` pulls the test towards the centre so seeds keep their LK window on the target, not on what is behind it. */
  private inside(x: number, y: number, shrink = 0): boolean {
    if (this.o.kind === 'frame') return true;
    if (shrink > 0) {
      const cx = quadCx(this.roi), cy = quadCy(this.roi);
      const k = 1 / (1 - shrink);
      x = cx + (x - cx) * k;
      y = cy + (y - cy) * k;
    }
    return this.o.shape === 'circle' ? pointInEllipse(this.roi, x, y) : pointInQuad(this.roi, x, y);
  }

  private seed(f: TrackerFrame, max: number): Float32Array {
    if (max <= 0) return new Float32Array(0);
    const frame = this.o.kind === 'frame';
    let x0 = 8, y0 = 8, x1 = f.w - 9, y1 = f.h - 9;
    let area = f.w * f.h;
    if (!frame) {
      const q = this.roi;
      x0 = Math.max(2, Math.min(q[0], q[2], q[4], q[6]));
      x1 = Math.min(f.w - 3, Math.max(q[0], q[2], q[4], q[6]));
      y0 = Math.max(2, Math.min(q[1], q[3], q[5], q[7]));
      y1 = Math.min(f.h - 3, Math.max(q[1], q[3], q[5], q[7]));
      const s = quadSize(q);
      area = s * s;
    }
    if (x1 - x0 < 3 || y1 - y0 < 3) return new Float32Array(0);
    const minDist = Math.min(26, Math.max(3, 0.7 * Math.sqrt(area / this.P.maxPts)));
    const shrink = frame ? 0 : Math.min(0.35, (1.6 * this.P.lk.radius) / Math.sqrt(area));
    const run = (useMask: boolean) =>
      detectCorners(f.pyr[0], {
        max,
        quality: this.P.quality,
        minDist,
        x0,
        y0,
        x1,
        y1,
        accept: (x, y) => this.inside(x, y, shrink) && (!useMask || this.maskOk(f, x, y)),
        existing: this.pts,
        nExisting: this.n,
      });
    let found = run(true);
    // a mask that rejects everything (nobody in shot) must not blind the tracker
    if (found.length < 8 && f.mask && this.o.mask !== 'none') {
      const all = run(false);
      if (all.length > found.length * 2) found = all;
    }
    return found;
  }

  private append(f: TrackerFrame, add: Float32Array) {
    const m = add.length >> 1;
    if (!m) return;
    const pts = new Float32Array((this.n + m) * 2);
    pts.set(this.pts.subarray(0, this.n * 2));
    pts.set(add, this.n * 2);
    const off = new Float32Array((this.n + m) * 2);
    off.set(this.off.subarray(0, this.n * 2));
    if (this.o.kind === 'features') {
      const { cx, cy, s, th } = this.pose;
      const c = Math.cos(-th), sn = Math.sin(-th);
      for (let i = 0; i < m; i++) {
        const dx = (add[i * 2] - cx) / s, dy = (add[i * 2 + 1] - cy) / s;
        off[(this.n + i) * 2] = dx * c - dy * sn;
        off[(this.n + i) * 2 + 1] = dx * sn + dy * c;
      }
    } else {
      // a new point is remembered where it would have been in the reference frame, so every later fit is absolute
      const inv = invert(this.M) ?? identity();
      for (let i = 0; i < m; i++) {
        off[(this.n + i) * 2] = applyX(inv, add[i * 2], add[i * 2 + 1]);
        off[(this.n + i) * 2 + 1] = applyY(inv, add[i * 2], add[i * 2 + 1]);
      }
    }
    this.off = off;
    this.pts = pts;
    this.n += m;
    void f;
  }

  /** Pick the pyramid level at which the region is ~32 px across, so the template is not aliased. */
  private patch(f: TrackerFrame, q: ArrayLike<number>, out?: Float32Array) {
    const size = quadSize(q);
    const lvl = Math.max(0, Math.min(f.pyr.length - 1, Math.round(Math.log2(Math.max(1, size / 32)))));
    const s = 1 / (1 << lvl);
    const sq = new Float64Array(8);
    for (let i = 0; i < 8; i++) sq[i] = q[i] * s;
    return { ...extractPatch(f.pyr[lvl], sq, TPL, out), lvl, sq };
  }

  /** Returns null on success or a reason the selection cannot be tracked. */
  init(f: TrackerFrame, quad: ArrayLike<number>): string | null {
    this.P = this.params(f.w, f.h);
    for (let i = 0; i < 8; i++) this.roi[i] = this.motion[i] = this.ref[i] = quad[i];
    this.c0x = quadCx(quad);
    this.c0y = quadCy(quad);
    this.pose = { cx: this.c0x, cy: this.c0y, s: 1, th: 0 };
    this.M = identity();
    this.pts = new Float32Array(0);
    this.off = new Float32Array(0);
    this.n = 0;
    this.searching = 0;
    this.frameNo = 0;
    const seeds = this.seed(f, this.P.maxPts);
    const real = seeds.length >> 1;
    if (real < 1) return 'low-detail';
    if (real < MODEL_MIN[this.o.model] && this.o.kind !== 'features') {
      // not enough structure for the requested model – it will fall back, but two points is the bare minimum
      if (real < 1) return 'low-detail';
    }
    this.append(f, seeds);
    if (real < 6 && this.o.kind !== 'frame') {
      // sparse texture: add a coarse lattice; flat patches are rejected by the eigenvalue gate later
      const extra: number[] = [];
      const q = this.roi;
      for (let j = 0; j < 3; j++) {
        for (let i = 0; i < 3; i++) {
          const u = 0.3 + 0.2 * i, v = 0.3 + 0.2 * j;
          const tx = q[0] + (q[2] - q[0]) * u, ty = q[1] + (q[3] - q[1]) * u;
          const bx = q[6] + (q[4] - q[6]) * u, by = q[7] + (q[5] - q[7]) * u;
          extra.push(tx + (bx - tx) * v, ty + (by - ty) * v);
        }
      }
      this.append(f, new Float32Array(extra));
    }
    if (this.o.kind !== 'frame') {
      const t = this.patch(f, this.roi);
      this.tpl = t.std > 1.5 ? t.patch : null;
    }
    this.prev = f;
    return null;
  }

  private out(state: 0 | 1, conf: number, lost: boolean, points: Float32Array | null): StepOut {
    return { quad: Array.from(this.motion), roi: Array.from(this.roi), conf, state, lost, points };
  }

  /** `step` is the motion since the last good frame; `abs` (when given) the motion since the reference frame. */
  private updateRegion(step: Mat3, abs: Mat3 | null) {
    const o = this.o;
    if (abs) {
      this.M = abs;
      transformQuad(abs, this.ref, this.motion);
    } else {
      this.M = mul(step, this.M);
      transformQuad(step, this.motion, this.motion);
    }
    if (o.kind === 'frame') return;
    if (o.cornerPin && o.shape === 'quad') {
      for (let i = 0; i < 8; i++) this.roi[i] = this.motion[i];
      return;
    }
    const cx = quadCx(this.roi), cy = quadCy(this.roi);
    const nx = applyX(step, cx, cy), ny = applyY(step, cx, cy);
    const loc = localSimilarity(step, cx, cy);
    const s = o.followSize ? loc.scale : 1;
    const th = o.followRotation && o.shape !== 'circle' ? loc.rotation : 0;
    const c = Math.cos(th) * s, sn = Math.sin(th) * s;
    for (let i = 0; i < 4; i++) {
      const dx = this.roi[i * 2] - cx, dy = this.roi[i * 2 + 1] - cy;
      this.roi[i * 2] = nx + dx * c - dy * sn;
      this.roi[i * 2 + 1] = ny + dx * sn + dy * c;
    }
  }

  private shift(dx: number, dy: number) {
    for (let i = 0; i < 4; i++) {
      this.roi[i * 2] += dx;
      this.roi[i * 2 + 1] += dy;
      this.motion[i * 2] += dx;
      this.motion[i * 2 + 1] += dy;
    }
    this.pose.cx += dx;
    this.pose.cy += dy;
    this.M = mul(new Float64Array([1, 0, dx, 0, 1, dy, 0, 0, 1]), this.M);
  }

  private reset(f: TrackerFrame) {
    this.pts = new Float32Array(0);
    this.off = new Float32Array(0);
    this.n = 0;
    this.append(f, this.seed(f, this.P.maxPts));
    this.prev = f;
  }

  /**
   * Measure the motion from the last good frame to `f`. With `shift`, the search starts displaced by that much
   * (used to re-lock after the template search has found where the target went).
   */
  private solve(f: TrackerFrame, shift: [number, number] | null) {
    const P = this.P;
    const o = this.o;
    const n = this.n;
    const A = this.prev.pyr;
    const B = f.pyr;
    const fwd = new Float32Array(n * 2);
    const back = new Float32Array(n * 2);
    const st = new Uint8Array(n);
    const st2 = new Uint8Array(n);
    const err = new Float32Array(n);
    const err2 = new Float32Array(n);
    const status = new Uint8Array(n).fill(POINT_LOST);
    const use = new Int32Array(n);
    let nUse = 0;
    let ok = false;
    let H: Mat3 | null = null;
    let inliers = 0;
    let geo = 0;
    let pose = this.pose;
    let abs: Mat3 | null = null;
    // after a gap the target has usually turned or changed size a little: be more forgiving when re-locking
    const loose = this.searching > 0 ? 2 : 1;
    if (n > 0) {
      let guess: Float32Array | null = null;
      if (shift) {
        guess = new Float32Array(n * 2);
        for (let i = 0; i < n; i++) {
          guess[i * 2] = this.pts[i * 2] + shift[0];
          guess[i * 2 + 1] = this.pts[i * 2 + 1] + shift[1];
        }
      }
      trackLK(A, B, this.pts, n, fwd, st, err, P.lk, guess);
      trackLK(B, A, fwd, n, back, st2, err2, P.lk, shift ? this.pts : null);
      const fb2 = P.fbMax * P.fbMax * loose * loose;
      for (let i = 0; i < n; i++) {
        if (!st[i] || !st2[i]) continue;
        status[i] = POINT_OUTLIER; // promoted to inlier by the model fit
        const dx = back[i * 2] - this.pts[i * 2], dy = back[i * 2 + 1] - this.pts[i * 2 + 1];
        if (dx * dx + dy * dy > fb2 || err[i] > 42) continue;
        use[nUse++] = i;
      }
    }

    if (o.kind === 'features') {
      // ── consensus: pairwise scale / rotation, then votes for the centre ──
      let s = this.pose.s, th = this.pose.th;
      if (nUse >= 2) {
        const ratios: number[] = [];
        const angs: number[] = [];
        const pairs = Math.min(320, (nUse * (nUse - 1)) / 2);
        for (let k = 0; k < pairs; k++) {
          const i = use[Math.floor(this.rand() * nUse)], j = use[Math.floor(this.rand() * nUse)];
          if (i === j) continue;
          const ox = this.off[j * 2] - this.off[i * 2], oy = this.off[j * 2 + 1] - this.off[i * 2 + 1];
          const od = Math.hypot(ox, oy);
          if (od < 4) continue;
          const px = fwd[j * 2] - fwd[i * 2], py = fwd[j * 2 + 1] - fwd[i * 2 + 1];
          ratios.push(Math.hypot(px, py) / od);
          angs.push(this.pose.th + wrapAngle(Math.atan2(py, px) - Math.atan2(oy, ox) - this.pose.th));
        }
        if (ratios.length >= 3) {
          s = median(ratios);
          th = median(angs);
        }
      }
      if (nUse >= 1 && s > 0.05 && s < 20) {
        const c = Math.cos(th) * s, sn = Math.sin(th) * s;
        const vx = new Float64Array(nUse), vy = new Float64Array(nUse);
        for (let k = 0; k < nUse; k++) {
          const i = use[k];
          vx[k] = fwd[i * 2] - (this.off[i * 2] * c - this.off[i * 2 + 1] * sn);
          vy[k] = fwd[i * 2 + 1] - (this.off[i * 2] * sn + this.off[i * 2 + 1] * c);
        }
        const cx = median(vx), cy = median(vy);
        const thr = Math.max(3, 0.07 * quadSize(this.ref) * s);
        let sx = 0, sy = 0;
        for (let k = 0; k < nUse; k++) {
          if (Math.hypot(vx[k] - cx, vy[k] - cy) <= thr) {
            status[use[k]] = POINT_INLIER;
            inliers++;
            sx += vx[k];
            sy += vy[k];
          }
        }
        // judged against the points we had, not just the ones that survived: an occluded target must not pass
        ok = inliers >= Math.max(nUse >= 4 ? 3 : 1, 0.3 * nUse, 0.18 * n);
        if (ok) {
          pose = { cx: sx / inliers, cy: sy / inliers, s, th };
          geo = inliers / Math.max(1, nUse);
          // polish: least-squares similarity from the constellation to where the agreeing points are now
          if (inliers >= 4) {
            const set = new Int32Array(inliers);
            let q = 0;
            for (let k = 0; k < nUse; k++) if (status[use[k]] === POINT_INLIER) set[q++] = use[k];
            const S = fitModel('similarity', this.off, fwd, set, q);
            if (S) {
              const ps = Math.hypot(S[0], S[3]);
              if (ps > 0.05 && ps < 20) pose = { cx: S[2], cy: S[5], s: ps, th: this.pose.th + wrapAngle(Math.atan2(S[3], S[0]) - this.pose.th) };
            }
          }
        }
      }
    } else if (nUse >= 1) {
      // ── frame-to-frame model fit with graceful fallback to simpler models ──
      const start = FALLBACK.indexOf(o.model);
      const prevInv = invert(this.M) ?? identity();
      const rx = quadCx(this.roi), ry = quadCy(this.roi);
      for (let m = start; m < FALLBACK.length; m++) {
        const model = FALLBACK[m];
        if (nUse < NEED[model]) continue;
        let step: Mat3;
        let res;
        if (model === 'translation') {
          // A pure translation fitted to a turning or growing target is biased by where the points happen to sit,
          // so measure a similarity between the two frames and keep only what it does to the centre of the region.
          const viaSim = nUse >= 6;
          res = ransac(viaSim ? 'similarity' : 'translation', this.pts, fwd, use, nUse, P.thr * loose, ITERS.similarity, this.rand);
          if (!res) continue;
          step = viaSim ? new Float64Array([1, 0, applyX(res.H, rx, ry) - rx, 0, 1, applyY(res.H, rx, ry) - ry, 0, 0, 1]) : res.H;
        } else {
          // Everything richer is fitted against the reference frame, not the previous one: the estimate is absolute,
          // so fitting noise cannot pile up frame after frame into skew or creeping scale.
          res = ransac(model, this.off, fwd, use, nUse, P.thr * 1.5 * loose, ITERS[model], this.rand);
          if (!res) continue;
          abs = res.H;
          step = mul(res.H, prevInv);
        }
        // judged against the points we had, not just the ones that survived: an occluded target must not pass
        const enough = res.count >= Math.min(NEED[model], nUse) && res.count >= 0.28 * nUse && (o.kind === 'frame' ? res.count >= 6 : res.count >= 0.18 * n);
        const loc = localSimilarity(step, rx, ry);
        // sanity: reject wild jumps between consecutive frames
        if (!enough || loc.scale < 0.6 || loc.scale > 1.6 || !Number.isFinite(applyX(step, rx, ry))) {
          abs = null;
          continue;
        }
        H = step;
        inliers = res.count;
        for (let k = 0; k < nUse; k++) if (res.inlier[use[k]]) status[use[k]] = POINT_INLIER;
        geo = res.count / nUse;
        ok = true;
        // patch rescue: retry points that failed, starting from where the model says they should be
        if (P.rescue && model !== 'translation') {
          const idx: number[] = [];
          for (let i = 0; i < n && idx.length < 80; i++) if (status[i] !== POINT_INLIER) idx.push(i);
          const cnt = idx.length;
          if (cnt) {
            const src = new Float32Array(cnt * 2), guess = new Float32Array(cnt * 2), out = new Float32Array(cnt * 2);
            const rst = new Uint8Array(cnt), rer = new Float32Array(cnt);
            for (let k = 0; k < cnt; k++) {
              const i = idx[k];
              src[k * 2] = this.pts[i * 2];
              src[k * 2 + 1] = this.pts[i * 2 + 1];
              guess[k * 2] = applyX(res.H, this.off[i * 2], this.off[i * 2 + 1]);
              guess[k * 2 + 1] = applyY(res.H, this.off[i * 2], this.off[i * 2 + 1]);
            }
            trackLK([A[0]], [B[0]], src, cnt, out, rst, rer, { ...P.lk, iterations: 8 }, guess);
            for (let k = 0; k < cnt; k++) {
              if (!rst[k] || rer[k] > 24) continue;
              if (Math.hypot(out[k * 2] - guess[k * 2], out[k * 2 + 1] - guess[k * 2 + 1]) > P.thr * 0.8) continue;
              const i = idx[k];
              fwd[i * 2] = out[k * 2];
              fwd[i * 2 + 1] = out[k * 2 + 1];
              status[i] = POINT_RESCUED;
            }
          }
        }
        break;
      }
    }

    // overlay points (before compaction so lost / outliers are visible for a frame)
    const points = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const moved = status[i] !== POINT_LOST;
      points[i * 3] = moved ? fwd[i * 2] : this.pts[i * 2];
      points[i * 3 + 1] = moved ? fwd[i * 2 + 1] : this.pts[i * 2 + 1];
      points[i * 3 + 2] = status[i];
    }
    return { ok, H, abs, pose, fwd, status, inliers, geo, points };
  }

  /** Commit a successful measurement: move the region, keep the surviving points, top the set back up. */
  private accept(f: TrackerFrame, m: ReturnType<PointTracker['solve']>) {
    const o = this.o;
    const P = this.P;
    if (o.kind === 'features') {
      this.pose = m.pose;
      const { cx, cy, s, th } = m.pose;
      const mc = Math.cos(th) * s, ms = Math.sin(th) * s;
      const rs = o.followSize ? s : 1;
      const rth = o.followRotation && o.shape !== 'circle' ? th : 0;
      const rc = Math.cos(rth) * rs, rsn = Math.sin(rth) * rs;
      for (let i = 0; i < 4; i++) {
        const dx = this.ref[i * 2] - this.c0x, dy = this.ref[i * 2 + 1] - this.c0y;
        this.motion[i * 2] = cx + dx * mc - dy * ms;
        this.motion[i * 2 + 1] = cy + dx * ms + dy * mc;
        this.roi[i * 2] = cx + dx * rc - dy * rsn;
        this.roi[i * 2 + 1] = cy + dx * rsn + dy * rc;
      }
    } else if (m.H) this.updateRegion(m.H, m.abs);
    let k = 0;
    for (let i = 0; i < this.n; i++) {
      if (m.status[i] !== POINT_INLIER && m.status[i] !== POINT_RESCUED) continue;
      const x = m.fwd[i * 2], y = m.fwd[i * 2 + 1];
      if (o.kind === 'flow' && !this.inside(x, y)) continue;
      this.pts[k * 2] = x;
      this.pts[k * 2 + 1] = y;
      this.off[k * 2] = this.off[i * 2];
      this.off[k * 2 + 1] = this.off[i * 2 + 1];
      k++;
    }
    this.n = k;
    this.prev = f;
    this.searching = 0;
    if (k < 0.65 * P.maxPts && (this.frameNo % P.reseedEvery === 0 || k < 0.35 * P.maxPts)) this.append(f, this.seed(f, P.maxPts - k));
  }

  step(f: TrackerFrame): StepOut {
    const P = this.P;
    const o = this.o;
    this.frameNo++;
    let m = this.solve(f, null);
    const cx = quadCx(this.roi), cy = quadCy(this.roi);
    const gone = o.kind !== 'frame' && (cx < -0.04 * f.w || cy < -0.04 * f.h || cx > 1.04 * f.w || cy > 1.04 * f.h);

    if (!m.ok) {
      this.searching++;
      if (o.kind === 'frame') {
        // a cut, a flash or a wall of motion blur: hold this frame and start again from here
        this.reset(f);
        return this.out(1, 0, this.searching > P.searchLimit, m.points);
      }
      if (P.recover && this.tpl && !gone) {
        // bounded recovery: look for the reference template around the last known position …
        const t = this.patch(f, this.motion);
        const size = quadSize(t.sq);
        const radius = Math.min(Math.max(f.pyr[t.lvl].w, f.pyr[t.lvl].h) * 0.45, size * (0.8 + 0.06 * this.searching) + 6);
        const hit = searchTemplate(f.pyr[t.lvl], t.sq, this.tpl, TPL, radius, Math.max(2, size / 12));
        if (hit.score >= 0.62) {
          const k = 1 << t.lvl;
          // … then re-lock the existing points from there, which keeps the scale and rotation the target has now
          const again = this.solve(f, [hit.dx * k, hit.dy * k]);
          if (again.ok) m = again;
          else {
            this.shift(hit.dx * k, hit.dy * k);
            this.reset(f);
            this.searching = 0;
            return this.out(0, Math.min(1, hit.score * 0.8), false, m.points);
          }
        }
      }
      // otherwise keep the last good frame as the reference: when the target reappears, LK locks straight back on
      if (!m.ok) return this.out(1, 0, gone || this.searching > P.searchLimit, m.points);
    }

    this.accept(f, m);
    let conf = Math.min(1, 0.3 + 0.7 * Math.min(1, m.geo * 1.15) * Math.min(1, m.inliers / 10));
    if (this.tpl && o.kind !== 'frame') {
      const t = this.patch(f, this.motion);
      const ncc = t.std > 1.5 ? correlate(t.patch, this.tpl) : 0;
      conf = 0.65 * conf + 0.35 * Math.max(0, ncc);
      // drift guard: the geometry agrees but what is under the region no longer resembles the target at all
      this.unlike = ncc < 0.05 ? this.unlike + 1 : 0;
      if (P.recover && this.unlike >= 6) {
        this.unlike = 0;
        this.searching = 1;
        return this.out(1, 0, false, m.points);
      }
    }
    return this.out(0, conf, gone, m.points);
  }
}

/** Two anchors tracked at once; their baseline carries scale and orientation. */
export class DualTracker {
  private a: PointTracker;
  private b: PointTracker;
  private half = 12;

  constructor(advanced: boolean) {
    const o: PointOptions = { kind: 'flow', model: 'translation', shape: 'rectangle', followSize: false, followRotation: false, cornerPin: false, advanced, mask: 'none' };
    this.a = new PointTracker(o);
    this.b = new PointTracker(o);
  }

  get pyramidLevels() {
    return 4;
  }

  private box(x: number, y: number): number[] {
    const h = this.half;
    return [x - h, y - h, x + h, y - h, x + h, y + h, x - h, y + h];
  }

  /** The quad encodes both anchors: A is the centre of its left edge, B the centre of its right edge. */
  static compose(ax: number, ay: number, bx: number, by: number): number[] {
    const cx = (ax + bx) / 2, cy = (ay + by) / 2;
    const ux = (bx - ax) / 2, uy = (by - ay) / 2;
    const vx = -uy, vy = ux;
    return [cx - ux - vx, cy - uy - vy, cx + ux - vx, cy + uy - vy, cx + ux + vx, cy + uy + vy, cx - ux + vx, cy - uy + vy];
  }

  init(f: TrackerFrame, quad: ArrayLike<number>): string | null {
    const ax = (quad[0] + quad[6]) / 2, ay = (quad[1] + quad[7]) / 2;
    const bx = (quad[2] + quad[4]) / 2, by = (quad[3] + quad[5]) / 2;
    this.half = Math.max(9, Math.min(Math.max(f.w, f.h) * 0.045, Math.hypot(bx - ax, by - ay) * 0.45));
    return this.a.init(f, this.box(ax, ay)) ?? this.b.init(f, this.box(bx, by));
  }

  step(f: TrackerFrame): StepOut {
    const ra = this.a.step(f);
    const rb = this.b.step(f);
    const ax = quadCx(ra.quad), ay = quadCy(ra.quad), bx = quadCx(rb.quad), by = quadCy(rb.quad);
    const quad = DualTracker.compose(ax, ay, bx, by);
    const na = ra.points ? ra.points.length : 0, nb = rb.points ? rb.points.length : 0;
    const points = new Float32Array(na + nb + 6);
    if (ra.points) points.set(ra.points, 0);
    if (rb.points) points.set(rb.points, na);
    points.set([ax, ay, POINT_LANDMARK, bx, by, POINT_LANDMARK], na + nb);
    return { quad, roi: quad, conf: Math.min(ra.conf, rb.conf), state: ra.state || rb.state ? 1 : 0, lost: ra.lost || rb.lost, points };
  }
}

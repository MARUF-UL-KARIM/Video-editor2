// Tracked motion as data: sampling, the "Adjust Motion" pipeline (components → filter → lock-on correction) and the
// editing operations on tracking segments (split, delete, connect, smooth, point corrections).
// Pure functions over immutable project data — safe to call from the render loop and from the exporter.
import type { ClipTracking, MotionAdjust, MotionComponent, TrackMoveMode, TrackSegment, TrackSettings } from '../../types';
import { quadAngle, quadCx, quadCy, quadSize } from './math';

export const MOTION_COMPONENTS: MotionComponent[] = ['horizontal', 'vertical', 'scale', 'rotation', 'affine', 'perspective'];

export const DEFAULT_ADJUST: MotionAdjust = {
  components: { horizontal: true, vertical: true, scale: true, rotation: true, affine: true, perspective: true },
  filter: 'off',
  strength: 60,
};

export const DEFAULT_SETTINGS: TrackSettings = {
  tracker: 'object',
  shape: 'rectangle',
  model: 'similarity',
  flowMode: 'advanced',
  followSize: true,
  followRotation: true,
  cornerPin: false,
  mask: 'none',
  resolution: 540,
  selection: 'entity',
  landmark: 'nose',
};

const uid = () => 'trk_' + Math.random().toString(36).slice(2, 10);
const r5 = (v: number) => Math.round(v * 1e5) / 1e5;

export const segStart = (s: TrackSegment) => s.t[0];
export const segEnd = (s: TrackSegment) => s.t[s.t.length - 1];
export const segCount = (s: TrackSegment) => s.t.length;

// ───────────────────────────── derived channels ─────────────────────────────

interface Derived {
  aspect: number;
  n: number;
  cx: Float64Array;
  cy: Float64Array;
  /** log of the quad's size (√area, in units of frame height) */
  ls: Float64Array;
  /** orientation, radians, unwrapped, measured in square pixels */
  rot: Float64Array;
}

const derivedCache = new WeakMap<TrackSegment, Derived>();

function derive(seg: TrackSegment, aspect: number): Derived {
  const hit = derivedCache.get(seg);
  if (hit && hit.aspect === aspect) return hit;
  const n = seg.t.length;
  const d: Derived = { aspect, n, cx: new Float64Array(n), cy: new Float64Array(n), ls: new Float64Array(n), rot: new Float64Array(n) };
  let prev = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 8;
    d.cx[i] = quadCx(seg.q, o);
    d.cy[i] = quadCy(seg.q, o);
    d.ls[i] = Math.log(quadSize(seg.q, o, aspect));
    let a = quadAngle(seg.q, o, aspect);
    if (i > 0) {
      while (a - prev > Math.PI) a -= 2 * Math.PI;
      while (a - prev < -Math.PI) a += 2 * Math.PI;
    }
    d.rot[i] = a;
    prev = a;
  }
  derivedCache.set(seg, d);
  return d;
}

function gaussian(src: Float64Array, sigma: number): Float64Array {
  const n = src.length;
  if (sigma < 0.05 || n < 3) return src.slice();
  const r = Math.min(n, Math.ceil(sigma * 3));
  const k = new Float64Array(r + 1);
  for (let i = 0; i <= r; i++) k[i] = Math.exp((-i * i) / (2 * sigma * sigma));
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0, wsum = 0;
    for (let j = -r; j <= r; j++) {
      const idx = i + j < 0 ? 0 : i + j >= n ? n - 1 : i + j;
      const w = k[j < 0 ? -j : j];
      acc += src[idx] * w;
      wsum += w;
    }
    out[i] = acc / wsum;
  }
  return out;
}

function follow(src: Float64Array, t: number[], tau: number): Float64Array {
  const out = new Float64Array(src.length);
  let v = src[0];
  for (let i = 0; i < src.length; i++) {
    const dt = i ? Math.max(1e-4, t[i] - t[i - 1]) : 0;
    v += (src[i] - v) * (1 - Math.exp(-dt / tau));
    out[i] = v;
  }
  return out;
}

interface Adjusted {
  d: Derived;
  ex: Float64Array;
  ey: Float64Array;
  els: Float64Array;
  erot: Float64Array;
  /** lock-on correction (target − effective) per channel, or null when the filter is off */
  dx: Float64Array | null;
  dy: Float64Array | null;
  dls: Float64Array | null;
  drot: Float64Array | null;
}

const adjustedCache = new WeakMap<TrackSegment, Adjusted>();

function adjusted(seg: TrackSegment, aspect: number): Adjusted {
  const hit = adjustedCache.get(seg);
  if (hit && hit.d.aspect === aspect) return hit;
  const d = derive(seg, aspect);
  const c = seg.adjust.components;
  const n = d.n;
  const hold = (src: Float64Array, keep: boolean) => (keep ? src : new Float64Array(n).fill(src[0]));
  const a: Adjusted = { d, ex: hold(d.cx, c.horizontal), ey: hold(d.cy, c.vertical), els: hold(d.ls, c.scale), erot: hold(d.rot, c.rotation), dx: null, dy: null, dls: null, drot: null };
  const f = seg.adjust.filter;
  const k = Math.min(1, Math.max(0, seg.adjust.strength / 100));
  if (f !== 'off' && n > 1) {
    const dt = Math.max(1e-3, (seg.t[n - 1] - seg.t[0]) / (n - 1));
    const target = (src: Float64Array): Float64Array => {
      if (f === 'lock') {
        // pin to where the target was when tracking started; partial strength still takes the micro-jitter out first
        const out = new Float64Array(n);
        for (let i = 0; i < n; i++) out[i] = src[i] + (src[0] - src[i]) * k;
        return out;
      }
      if (f === 'smooth') return gaussian(src, (0.06 + 1.3 * k * k) / dt);
      if (f === 'follow') return follow(src, seg.t, 0.04 + 0.9 * k * k);
      return gaussian(src, (0.03 + 0.28 * k) / dt); // stabilize: only the micro-shake is removed
    };
    const diff = (src: Float64Array) => {
      const tg = target(src);
      for (let i = 0; i < n; i++) tg[i] -= src[i];
      return tg;
    };
    a.dx = diff(a.ex);
    a.dy = diff(a.ey);
    a.dls = diff(a.els);
    a.drot = diff(a.erot);
  }
  adjustedCache.set(seg, a);
  return a;
}

// ───────────────────────────── sampling ─────────────────────────────

/** Sample position inside a segment: index of the sample at or before `ts` and the blend to the next one. */
export function locate(seg: TrackSegment, ts: number): { i: number; f: number; inside: boolean } {
  const t = seg.t;
  const n = t.length;
  if (ts <= t[0]) return { i: 0, f: 0, inside: ts >= t[0] - 1e-4 };
  if (ts >= t[n - 1]) return { i: n - 1, f: 0, inside: ts <= t[n - 1] + 1e-4 };
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= ts) lo = mid;
    else hi = mid;
  }
  const span = t[hi] - t[lo];
  return { i: lo, f: span > 1e-9 ? (ts - t[lo]) / span : 0, inside: true };
}

/** The segment covering `ts`, or the nearest one (its edge is held outside the tracked range). */
export function segmentAt(tr: ClipTracking | undefined, ts: number): { seg: TrackSegment; index: number; inside: boolean } | null {
  if (!tr || !tr.segments.length) return null;
  let best = -1;
  let bestD = Infinity;
  for (let k = 0; k < tr.segments.length; k++) {
    const s = tr.segments[k];
    if (!s.t.length) continue;
    if (ts >= s.t[0] - 1e-4 && ts <= s.t[s.t.length - 1] + 1e-4) return { seg: s, index: k, inside: true };
    const dist = ts < s.t[0] ? s.t[0] - ts : ts - s.t[s.t.length - 1];
    if (dist < bestD) {
      bestD = dist;
      best = k;
    }
  }
  return best < 0 ? null : { seg: tr.segments[best], index: best, inside: false };
}

export interface MotionState {
  seg: TrackSegment;
  segIndex: number;
  sample: number;
  inside: boolean;
  conf: number;
  searching: boolean;
  /** raw tracked centre, normalised source coordinates */
  cx: number;
  cy: number;
  /** what the adjusted motion follows (components applied) */
  ex: number;
  ey: number;
  /** absolute size (√area in frame heights) and orientation (rad) of the adjusted motion */
  size: number;
  rot: number;
  /** lock-on correction to apply to the clip in source space */
  dx: number;
  dy: number;
  ds: number;
  dr: number;
  locked: boolean;
  /** adjusted quad, normalised source coordinates */
  quad: number[];
  /** raw quad, normalised source coordinates */
  raw: number[];
}

const mix = (a: Float64Array, i: number, f: number) => (f > 0 ? a[i] + (a[i + 1] - a[i]) * f : a[i]);

export function motionAt(tr: ClipTracking | undefined, ts: number, aspect: number): MotionState | null {
  const hit = segmentAt(tr, ts);
  if (!hit) return null;
  const seg = hit.seg;
  const a = adjusted(seg, aspect);
  const d = a.d;
  const { i, f } = locate(seg, ts);
  const cx = mix(d.cx, i, f), cy = mix(d.cy, i, f), ls = mix(d.ls, i, f), rot = mix(d.rot, i, f);
  const ex = mix(a.ex, i, f), ey = mix(a.ey, i, f), els = mix(a.els, i, f), erot = mix(a.erot, i, f);
  const raw: number[] = new Array(8);
  for (let k = 0; k < 8; k++) raw[k] = f > 0 ? seg.q[i * 8 + k] + (seg.q[(i + 1) * 8 + k] - seg.q[i * 8 + k]) * f : seg.q[i * 8 + k];
  const comp = seg.adjust.components;
  const rigid = !comp.affine && !comp.perspective;
  // carry the quad from the raw pose onto the adjusted pose (or rebuild it rigidly from the first sample)
  const quad: number[] = new Array(8);
  const src = rigid ? seg.q : raw;
  const o = 0;
  const bx = rigid ? d.cx[0] : cx, by = rigid ? d.cy[0] : cy, bls = rigid ? d.ls[0] : ls, brot = rigid ? d.rot[0] : rot;
  const s = Math.exp(els - bls);
  const c = Math.cos(erot - brot) * s, sn = Math.sin(erot - brot) * s;
  for (let k = 0; k < 4; k++) {
    const px = (src[o + k * 2] - bx) * aspect, py = src[o + k * 2 + 1] - by;
    quad[k * 2] = ex + (px * c - py * sn) / aspect;
    quad[k * 2 + 1] = ey + px * sn + py * c;
  }
  if (!rigid && !comp.perspective) {
    // affine only: square the quad up into a parallelogram
    const mx = quadCx(quad), my = quadCy(quad);
    const hx = (quad[2] + quad[4] - quad[0] - quad[6]) / 4, hy = (quad[3] + quad[5] - quad[1] - quad[7]) / 4;
    const vx = (quad[6] + quad[4] - quad[0] - quad[2]) / 4, vy = (quad[7] + quad[5] - quad[1] - quad[3]) / 4;
    quad[0] = mx - hx - vx; quad[1] = my - hy - vy;
    quad[2] = mx + hx - vx; quad[3] = my + hy - vy;
    quad[4] = mx + hx + vx; quad[5] = my + hy + vy;
    quad[6] = mx - hx + vx; quad[7] = my - hy + vy;
  }
  const locked = !!a.dx;
  return {
    seg,
    segIndex: hit.index,
    sample: f > 0.5 ? i + 1 : i,
    inside: hit.inside,
    conf: seg.c[i] ?? 1,
    searching: (seg.s[i] ?? 0) === 1,
    cx,
    cy,
    ex,
    ey,
    size: Math.exp(els),
    rot: erot,
    dx: locked ? mix(a.dx!, i, f) : 0,
    dy: locked ? mix(a.dy!, i, f) : 0,
    ds: locked ? Math.exp(mix(a.dls!, i, f)) : 1,
    dr: locked ? mix(a.drot!, i, f) : 0,
    locked,
    quad,
    raw,
  };
}

/** Path of the adjusted motion (normalised source coordinates), decimated for drawing. */
export function motionPath(seg: TrackSegment, aspect: number, maxPoints = 240): number[] {
  const a = adjusted(seg, aspect);
  const n = a.d.n;
  const step = Math.max(1, Math.ceil(n / maxPoints));
  const out: number[] = [];
  for (let i = 0; i < n; i += step) out.push(a.ex[i], a.ey[i]);
  if ((n - 1) % step) out.push(a.ex[n - 1], a.ey[n - 1]);
  return out;
}

// ───────────────────────────── lock-on framing ─────────────────────────────

const zoomCache = new WeakMap<ClipTracking, { aspect: number; z: number }>();

/**
 * Smallest constant zoom that keeps the frame covered while lock-on moves the picture around.
 * Works in the clip's own space: width = aspect, height = 1, origin at the centre.
 */
export function fitZoom(tr: ClipTracking, aspect: number): number {
  if (!tr.autoFit) return 1;
  const hit = zoomCache.get(tr);
  if (hit && hit.aspect === aspect) return hit.z;
  const hw = aspect / 2, hh = 0.5;
  const corners = [-hw, -hh, hw, -hh, hw, hh, -hw, hh];
  const covers = (z: number): boolean => {
    for (const seg of tr.segments) {
      const a = adjusted(seg, aspect);
      if (!a.dx) continue;
      const n = a.d.n;
      const step = Math.max(1, Math.floor(n / 400));
      for (let i = 0; i < n; i += step) {
        const exl = (a.ex[i] - 0.5) * aspect, eyl = a.ey[i] - 0.5;
        const txl = exl + a.dx[i] * aspect, tyl = eyl + a.dy![i];
        const inv = 1 / Math.exp(a.dls![i]);
        const c = Math.cos(-a.drot![i]) * inv, s = Math.sin(-a.drot![i]) * inv;
        for (let k = 0; k < 4; k++) {
          const px = corners[k * 2] / z - txl, py = corners[k * 2 + 1] / z - tyl;
          const x = exl + px * c - py * s, y = eyl + px * s + py * c;
          if (Math.abs(x) > hw + 1e-4 || Math.abs(y) > hh + 1e-4) return false;
        }
      }
    }
    return true;
  };
  let z = 1;
  if (!covers(1)) {
    let lo = 1, hi = 3;
    if (!covers(hi)) lo = hi;
    else
      for (let it = 0; it < 16; it++) {
        const mid = (lo + hi) / 2;
        if (covers(mid)) hi = mid;
        else lo = mid;
      }
    z = covers(hi) ? hi : 3;
  }
  zoomCache.set(tr, { aspect, z });
  return z;
}

// ───────────────────────────── segment editing ─────────────────────────────

export function makeSegment(settings: TrackSettings, t: number[], q: number[], c: number[], s: number[], adjust: MotionAdjust = DEFAULT_ADJUST): TrackSegment {
  return { id: uid(), settings: { ...settings }, t: t.map((v) => Math.round(v * 1e5) / 1e5), q: q.map(r5), c: c.map((v) => Math.round(v * 100) / 100), s: s.slice(), adjust: { ...adjust, components: { ...adjust.components } } };
}

function slice(seg: TrackSegment, i0: number, i1: number, fresh: boolean): TrackSegment | null {
  if (i1 - i0 < 1) return null;
  return { ...seg, id: fresh ? uid() : seg.id, t: seg.t.slice(i0, i1 + 1), q: seg.q.slice(i0 * 8, (i1 + 1) * 8), c: seg.c.slice(i0, i1 + 1), s: seg.s.slice(i0, i1 + 1) };
}

const sorted = (segs: TrackSegment[]) => segs.filter((s) => s.t.length >= 2).sort((a, b) => a.t[0] - b.t[0]);
const emptyTracking = (): ClipTracking => ({ id: uid(), segments: [], autoFit: true });

/** Insert a freshly tracked segment; whatever it overlaps is overwritten. */
export function addSegment(tr: ClipTracking | undefined, seg: TrackSegment): ClipTracking {
  const base = tr ?? emptyTracking();
  const a = segStart(seg), b = segEnd(seg);
  const out: TrackSegment[] = [];
  for (const s of base.segments) {
    if (segEnd(s) < a - 1e-4 || segStart(s) > b + 1e-4) {
      out.push(s);
      continue;
    }
    let i0 = -1;
    for (let i = 0; i < s.t.length; i++) if (s.t[i] < a - 1e-4) i0 = i;
    let i1 = s.t.length;
    for (let i = s.t.length - 1; i >= 0; i--) if (s.t[i] > b + 1e-4) i1 = i;
    const left = i0 >= 1 ? slice(s, 0, i0, false) : null;
    const right = i1 <= s.t.length - 2 ? slice(s, i1, s.t.length - 1, !!left) : null;
    if (left) out.push(left);
    if (right) out.push(right);
  }
  out.push(seg);
  return { ...base, segments: sorted(out) };
}

export function splitAt(tr: ClipTracking, ts: number): ClipTracking | null {
  const hit = segmentAt(tr, ts);
  if (!hit || !hit.inside) return null;
  const { i, f } = locate(hit.seg, ts);
  const at = f > 0.5 ? i + 1 : i;
  if (at < 1 || at > hit.seg.t.length - 2) return null;
  const left = slice(hit.seg, 0, at, false);
  const right = slice(hit.seg, at, hit.seg.t.length - 1, true);
  if (!left || !right) return null;
  const segments = tr.segments.slice();
  segments.splice(hit.index, 1, left, right);
  return { ...tr, segments };
}

export function deleteAt(tr: ClipTracking, ts: number): ClipTracking | null {
  const hit = segmentAt(tr, ts);
  if (!hit || !hit.inside) return null;
  return { ...tr, segments: tr.segments.filter((_, k) => k !== hit.index) };
}

/**
 * Join the segment at (or just before) `ts` with the one after it into a single continuous motion.
 * Touching segments are made continuous by carrying the later one onto the end of the earlier one; a real gap is
 * bridged by interpolating the quad across it (marked as "searching" samples).
 */
export function connectAt(tr: ClipTracking, ts: number, aspect: number, fps = 30): ClipTracking | null {
  const segs = tr.segments;
  if (segs.length < 2) return null;
  let ai = -1;
  for (let k = 0; k < segs.length - 1; k++) if (ts >= segStart(segs[k]) - 1e-4 && ts < segStart(segs[k + 1])) ai = k;
  if (ai < 0) ai = ts < segStart(segs[0]) ? 0 : segs.length - 2;
  const A = segs[ai], B = segs[ai + 1];
  const na = A.t.length;
  const gap = segStart(B) - segEnd(A);
  const t = A.t.slice(), q = A.q.slice(), c = A.c.slice(), s = A.s.slice();
  const lastA = A.q.slice((na - 1) * 8, na * 8);
  const firstB = B.q.slice(0, 8);
  let bq = B.q;
  let from = 0;
  if (gap <= 1.6 / fps) {
    // adjacent: remove the jump by re-expressing B relative to where A ended
    const ax = quadCx(lastA), ay = quadCy(lastA), bx = quadCx(firstB), by = quadCy(firstB);
    const k = quadSize(lastA, 0, aspect) / quadSize(firstB, 0, aspect);
    const dth = quadAngle(lastA, 0, aspect) - quadAngle(firstB, 0, aspect);
    const cs = Math.cos(dth) * k, sn = Math.sin(dth) * k;
    bq = B.q.slice();
    for (let i = 0; i < B.t.length * 4; i++) {
      const px = (B.q[i * 2] - bx) * aspect, py = B.q[i * 2 + 1] - by;
      bq[i * 2] = r5(ax + (px * cs - py * sn) / aspect);
      bq[i * 2 + 1] = r5(ay + px * sn + py * cs);
    }
    if (gap < 1e-4) from = 1;
  } else {
    const steps = Math.min(600, Math.round(gap * fps) - 1);
    for (let k = 1; k <= steps; k++) {
      const u = k / (steps + 1);
      const e = u * u * (3 - 2 * u);
      t.push(r5(segEnd(A) + gap * u));
      for (let j = 0; j < 8; j++) q.push(r5(lastA[j] + (firstB[j] - lastA[j]) * e));
      c.push(0);
      s.push(1);
    }
  }
  for (let i = from; i < B.t.length; i++) {
    t.push(B.t[i]);
    for (let j = 0; j < 8; j++) q.push(bq[i * 8 + j]);
    c.push(B.c[i]);
    s.push(B.s[i]);
  }
  const merged: TrackSegment = { ...A, t, q, c, s };
  const segments = segs.slice();
  segments.splice(ai, 2, merged);
  return { ...tr, segments };
}

/** Destructive smoothing of the tracked path itself (the corners are filtered over time). `amount` 0–1. */
export function smoothAt(tr: ClipTracking, ts: number | null, amount: number): ClipTracking {
  const pick = ts === null ? null : segmentAt(tr, ts);
  const segments = tr.segments.map((seg, k) => {
    if (pick && pick.index !== k) return seg;
    const n = seg.t.length;
    if (n < 3) return seg;
    const dt = Math.max(1e-3, (seg.t[n - 1] - seg.t[0]) / (n - 1));
    const sigma = (0.03 + 0.25 * amount) / dt;
    const q = seg.q.slice();
    const ch = new Float64Array(n);
    for (let j = 0; j < 8; j++) {
      for (let i = 0; i < n; i++) ch[i] = seg.q[i * 8 + j];
      const sm = gaussian(ch, sigma);
      for (let i = 0; i < n; i++) q[i * 8 + j] = r5(sm[i]);
    }
    return { ...seg, q };
  });
  return { ...tr, segments };
}

/**
 * Correct a tracked point by (dx, dy) in normalised source units.
 *  current – only the sample at the playhead
 *  forward – that sample and everything after it
 *  both    – drift correction: ramp the correction in over the earlier samples, full from the playhead on
 */
export function moveSample(tr: ClipTracking, segId: string, index: number, dx: number, dy: number, mode: TrackMoveMode): ClipTracking {
  const segments = tr.segments.map((seg) => {
    if (seg.id !== segId) return seg;
    const n = seg.t.length;
    const at = Math.min(n - 1, Math.max(0, index));
    const q = seg.q.slice();
    const t0 = seg.t[0], t1 = seg.t[at];
    for (let i = 0; i < n; i++) {
      let w = 0;
      if (mode === 'current') w = i === at ? 1 : 0;
      else if (mode === 'forward') w = i >= at ? 1 : 0;
      else w = i >= at ? 1 : t1 > t0 ? (seg.t[i] - t0) / (t1 - t0) : 0;
      if (!w) continue;
      for (let k = 0; k < 4; k++) {
        q[i * 8 + k * 2] = r5(q[i * 8 + k * 2] + dx * w);
        q[i * 8 + k * 2 + 1] = r5(q[i * 8 + k * 2 + 1] + dy * w);
      }
    }
    return { ...seg, q };
  });
  return { ...tr, segments };
}

export function setAdjust(tr: ClipTracking, ts: number, patch: Partial<MotionAdjust>, all: boolean): ClipTracking {
  const pick = segmentAt(tr, ts);
  const segments = tr.segments.map((seg, k) => {
    if (!all && pick && pick.index !== k) return seg;
    return { ...seg, adjust: { ...seg.adjust, ...patch, components: { ...seg.adjust.components, ...(patch.components ?? {}) } } };
  });
  return { ...tr, segments };
}

export interface TrackingStats {
  segments: number;
  samples: number;
  start: number;
  end: number;
  searching: number;
  confidence: number;
}

export function trackingStats(tr: ClipTracking | undefined): TrackingStats | null {
  if (!tr || !tr.segments.length) return null;
  let samples = 0, searching = 0, conf = 0;
  for (const s of tr.segments) {
    samples += s.t.length;
    for (let i = 0; i < s.t.length; i++) {
      if (s.s[i]) searching++;
      conf += s.c[i] ?? 0;
    }
  }
  return { segments: tr.segments.length, samples, start: segStart(tr.segments[0]), end: segEnd(tr.segments[tr.segments.length - 1]), searching, confidence: samples ? conf / samples : 0 };
}

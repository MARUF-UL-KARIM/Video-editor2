// Where tracked motion meets the compositor. Called from collectLayers(), so the preview, on-canvas handles,
// hit-testing and the exporter all see exactly the same result.
//
//  1. Lock-on: a tracked clip is counter-moved in its own source space so its target follows the adjusted path
//     (pinned, smoothed, followed with a delay, or only de-jittered).
//  2. Attach: other layers ride on the target as it is finally displayed — position, size, rotation, or all four
//     corners (corner pin).
import type { Clip, ClipAttach, Project, TrackLock } from '../../types';
import type { Layer } from '../anim';
import { clipValue } from '../store';
import { quadCx, quadCy } from './math';
import { fitZoom, motionAt, type MotionState } from './motion';
import { trackState } from './state';

const RAD = Math.PI / 180;

export function assetAspect(project: Project, clip: Clip): number {
  const a = project.assets.find((x) => x.id === clip.assetId);
  return a && a.width && a.height ? a.width / a.height : project.width / project.height;
}

/** Unscaled size of a media clip on the canvas (same rule as the compositor's fit). */
function baseSize(project: Project, clip: Clip): { w: number; h: number } {
  const a = project.assets.find((x) => x.id === clip.assetId);
  const aw = a?.width || project.width;
  const ah = a?.height || project.height;
  const s = clip.fit === 'cover' ? Math.max(project.width / aw, project.height / ah) : Math.min(project.width / aw, project.height / ah);
  return { w: aw * s, h: ah * s };
}

export interface Geom {
  /** centre on the canvas, project px */
  cx: number;
  cy: number;
  scale: number;
  /** degrees */
  rotation: number;
  w: number;
  h: number;
  flip: boolean;
}

/** Source-normalised point → canvas px through a clip's placement. */
export function sourceToCanvas(g: Geom, sx: number, sy: number): [number, number] {
  const qx = ((g.flip ? 1 - sx : sx) - 0.5) * g.w * g.scale;
  const qy = (sy - 0.5) * g.h * g.scale;
  const c = Math.cos(g.rotation * RAD), s = Math.sin(g.rotation * RAD);
  return [g.cx + qx * c - qy * s, g.cy + qx * s + qy * c];
}

/** Canvas px → source-normalised point. */
export function canvasToSource(g: Geom, x: number, y: number): [number, number] {
  const c = Math.cos(-g.rotation * RAD), s = Math.sin(-g.rotation * RAD);
  const dx = x - g.cx, dy = y - g.cy;
  const qx = (dx * c - dy * s) / (g.w * g.scale), qy = (dx * s + dy * c) / (g.h * g.scale);
  const u = qx + 0.5;
  return [g.flip ? 1 - u : u, qy + 0.5];
}

/** Lock chosen in the tracking panel: on the clip once a track exists, otherwise the in-progress draft. */
export function pointLockOf(clip: Clip): TrackLock | null {
  if (clip.tracking?.lock) return clip.tracking.lock;
  const d = trackState().lockDraft;
  if (d && d.clipId === clip.id) return { enabled: d.enabled, x: d.x, y: d.y, placed: d.placed };
  return null;
}

/** Tracker-middle in source space. Live analysis wins inside its range so Lock can hold while tracking. */
function centreAt(clip: Clip, srcTime: number, aspect: number, useLive: boolean): { cx: number; cy: number } | null {
  if (useLive) {
    const live = trackState().live;
    const phase = trackState().phase;
    if (live && live.clipId === clip.id && live.t.length && phase !== 'idle') {
      const t = live.t;
      if (srcTime >= t[0] - 1e-3 && srcTime <= t[t.length - 1] + 0.08) {
        let lo = 0, hi = t.length - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (t[mid] <= srcTime) lo = mid;
          else hi = mid;
        }
        const i = srcTime - t[lo] <= t[hi] - srcTime ? lo : hi;
        return { cx: quadCx(live.q, i * 8), cy: quadCy(live.q, i * 8) };
      }
    }
  }
  const m = motionAt(clip.tracking, srcTime, aspect);
  return m ? { cx: m.cx, cy: m.cy } : null;
}

/** Translate `l` so the source point (sx, sy) lands on the lock point. Mutates x / y. */
function pinTo(project: Project, l: { x: number; y: number; scale: number; rotation: number }, clip: Clip, sx: number, sy: number, lockX: number, lockY: number) {
  const { w, h } = baseSize(project, clip);
  const W = project.width, H = project.height;
  const geom: Geom = { cx: W / 2 + l.x * W, cy: H / 2 + l.y * H, scale: l.scale, rotation: l.rotation, w, h, flip: clip.flipH };
  const [px, py] = sourceToCanvas(geom, sx, sy);
  l.x += (W / 2 + lockX * W - px) / W;
  l.y += (H / 2 + lockY * H - py) / H;
}

function zoomAround(l: { x: number; y: number; scale: number }, lockX: number, lockY: number, z: number) {
  if (z <= 1.001) return;
  l.x = lockX + (l.x - lockX) * z;
  l.y = lockY + (l.y - lockY) * z;
  l.scale *= z;
}

/** Smallest zoom about the lock point that keeps the canvas covered by the (already pinned) picture. */
function requiredZoom(project: Project, placed: { x: number; y: number; scale: number; rotation: number }, clip: Clip, lockX: number, lockY: number): number {
  const { w, h } = baseSize(project, clip);
  const W = project.width, H = project.height;
  const hw = (w * placed.scale) / 2, hh = (h * placed.scale) / 2;
  if (hw < 2 || hh < 2) return 1;
  const cx = W / 2 + placed.x * W, cy = H / 2 + placed.y * H;
  const lx = W / 2 + lockX * W, ly = H / 2 + lockY * H;
  const rot = (placed.rotation * Math.PI) / 180;
  const c = Math.cos(-rot), s = Math.sin(-rot);
  const corners = [0, 0, W, 0, W, H, 0, H];
  const covers = (z: number) => {
    for (let i = 0; i < 4; i++) {
      const px = lx + (corners[i * 2] - lx) / z;
      const py = ly + (corners[i * 2 + 1] - ly) / z;
      const dx = px - cx, dy = py - cy;
      const lx2 = dx * c - dy * s, ly2 = dx * s + dy * c;
      if (Math.abs(lx2) > hw + 1.25 || Math.abs(ly2) > hh + 1.25) return false;
    }
    return true;
  };
  if (covers(1)) return 1;
  let lo = 1, hi = 3.4;
  if (!covers(hi)) return hi;
  for (let it = 0; it < 14; it++) {
    const mid = (lo + hi) / 2;
    if (covers(mid)) hi = mid;
    else lo = mid;
  }
  return hi < 1.01 ? 1 : hi * 1.012;
}

function placedAt(clip: Clip, srcTime: number) {
  const pt = clip.start + (srcTime - clip.inPoint) / Math.max(1e-6, clip.speed);
  const lt = Math.min(clip.duration, Math.max(0, pt - clip.start));
  return { x: clipValue(clip, 'x', lt), y: clipValue(clip, 'y', lt), scale: clipValue(clip, 'scale', lt), rotation: clipValue(clip, 'rotation', lt) };
}

function frameCover(project: Project, clip: Clip, aspect: number, srcTime: number, sx: number, sy: number, lock: TrackLock): number {
  const placed = placedAt(clip, srcTime);
  const m = motionAt(clip.tracking, srcTime, aspect);
  if (m?.locked) lockOn(project, placed, clip, m, aspect);
  pinTo(project, placed, clip, sx, sy, lock.x, lock.y);
  return requiredZoom(project, placed, clip, lock.x, lock.y);
}

const coverCache = new WeakMap<Clip, { sig: string; z: number }>();

function committedCover(project: Project, clip: Clip, aspect: number, lock: TrackLock): number {
  const sig = `${project.width}x${project.height}|${lock.x}|${lock.y}|${clip.flipH}|${clip.fit}|${clip.assetId}|${aspect}`;
  const hit = coverCache.get(clip);
  if (hit && hit.sig === sig) return hit.z;
  const segs = clip.tracking?.segments ?? [];
  let z = 1;
  for (const seg of segs) {
    const n = seg.t.length;
    if (n < 1) continue;
    const step = Math.max(1, Math.floor(n / 160));
    for (let i = 0; i < n; i += step) z = Math.max(z, frameCover(project, clip, aspect, seg.t[i], quadCx(seg.q, i * 8), quadCy(seg.q, i * 8), lock));
    const last = n - 1;
    z = Math.max(z, frameCover(project, clip, aspect, seg.t[last], quadCx(seg.q, last * 8), quadCy(seg.q, last * 8), lock));
  }
  coverCache.set(clip, { sig, z });
  return z;
}

let liveCoverCache = { clipId: '', n: -1, x: 0, y: 0, z: 1 };

function liveCover(project: Project, clip: Clip, aspect: number, lock: TrackLock): number {
  const live = trackState().live;
  if (!live || live.clipId !== clip.id || trackState().phase === 'idle' || live.t.length < 2) return 1;
  if (liveCoverCache.clipId === clip.id && liveCoverCache.n === live.t.length && liveCoverCache.x === lock.x && liveCoverCache.y === lock.y) return liveCoverCache.z;
  let z = 1;
  const step = Math.max(1, Math.floor(live.t.length / 80));
  for (let i = 0; i < live.t.length; i += step) z = Math.max(z, frameCover(project, clip, aspect, live.t[i], quadCx(live.q, i * 8), quadCy(live.q, i * 8), lock));
  const last = live.t.length - 1;
  z = Math.max(z, frameCover(project, clip, aspect, live.t[last], quadCx(live.q, last * 8), quadCy(live.q, last * 8), lock));
  liveCoverCache = { clipId: clip.id, n: live.t.length, x: lock.x, y: lock.y, z };
  return z;
}

function coverZoom(project: Project, clip: Clip, aspect: number, lock: TrackLock): number {
  if (clip.tracking && clip.tracking.autoFit === false) return 1;
  return Math.max(committedCover(project, clip, aspect, lock), liveCover(project, clip, aspect, lock));
}

/** Hold the tracker middle on the lock point, then zoom about that point so the edges stay hidden. */
function applyPointLock(project: Project, l: { x: number; y: number; scale: number; rotation: number }, clip: Clip, srcTime: number, aspect: number) {
  const lock = pointLockOf(clip);
  if (!lock?.enabled) return;
  const c = centreAt(clip, srcTime, aspect, true);
  if (!c) return;
  pinTo(project, l, clip, c.cx, c.cy, lock.x, lock.y);
  zoomAround(l, lock.x, lock.y, coverZoom(project, clip, aspect, lock));
}

/**
 * Where the tracker middle would appear if point-lock were off, in clip.x / clip.y units.
 * Used to place the reticle and to capture the hold point when Lock is turned on.
 */
export function trackerCentreCanvas(project: Project, clip: Clip, t: number): { x: number; y: number } | null {
  const aspect = assetAspect(project, clip);
  const srcTime = clip.inPoint + Math.min(clip.duration, Math.max(0, t - clip.start)) * clip.speed;
  const c = centreAt(clip, srcTime, aspect, true);
  if (!c) return null;
  const lt = Math.min(clip.duration, Math.max(0, t - clip.start));
  const placed = { x: clipValue(clip, 'x', lt), y: clipValue(clip, 'y', lt), scale: clipValue(clip, 'scale', lt), rotation: clipValue(clip, 'rotation', lt) };
  const m = motionAt(clip.tracking, srcTime, aspect);
  if (m?.locked) lockOn(project, placed, clip, m, aspect);
  const { w, h } = baseSize(project, clip);
  const geom: Geom = { cx: project.width / 2 + placed.x * project.width, cy: project.height / 2 + placed.y * project.height, scale: placed.scale, rotation: placed.rotation, w, h, flip: clip.flipH };
  const [px, py] = sourceToCanvas(geom, c.cx, c.cy);
  return { x: (px - project.width / 2) / project.width, y: (py - project.height / 2) / project.height };
}

/** Counter-move a layer so the tracked target lands on its adjusted path. Mutates x / y / scale / rotation. */
function lockOn(project: Project, l: { x: number; y: number; scale: number; rotation: number }, clip: Clip, m: MotionState, aspect: number) {
  const { w, h } = baseSize(project, clip);
  const W = project.width, H = project.height;
  const flip = clip.flipH ? -1 : 1;
  // target point before / after correction, in the clip's unscaled local frame
  const ex = ((clip.flipH ? 1 - m.ex : m.ex) - 0.5) * w, ey = (m.ey - 0.5) * h;
  const tx = ex + flip * m.dx * w, ty = ey + m.dy * h;
  const dth = flip * m.dr;
  const c = Math.cos(dth) * m.ds, s = Math.sin(dth) * m.ds;
  // where the clip's own centre ends up: T − ds·R(dθ)·E
  let ox = tx - (ex * c - ey * s), oy = ty - (ex * s + ey * c);
  const z = fitZoom(clip.tracking!, aspect);
  ox *= z;
  oy *= z;
  const rc = Math.cos(l.rotation * RAD) * l.scale, rs = Math.sin(l.rotation * RAD) * l.scale;
  l.x += (ox * rc - oy * rs) / W;
  l.y += (ox * rs + oy * rc) / H;
  l.scale *= m.ds * z;
  l.rotation += dth / RAD;
}

export interface TargetPose {
  geom: Geom;
  motion: MotionState;
  /** canvas px of the adjusted target point */
  x: number;
  y: number;
  /** on-screen size factor and rotation (deg) of the target */
  k: number;
  r: number;
  /** adjusted quad on the canvas, project px */
  quad: number[];
}

function poseFrom(project: Project, clip: Clip, placed: { x: number; y: number; scale: number; rotation: number }, m: MotionState): TargetPose {
  const { w, h } = baseSize(project, clip);
  const geom: Geom = { cx: project.width / 2 + placed.x * project.width, cy: project.height / 2 + placed.y * project.height, scale: placed.scale, rotation: placed.rotation, w, h, flip: clip.flipH };
  const [x, y] = sourceToCanvas(geom, m.ex, m.ey);
  const quad: number[] = [];
  for (let i = 0; i < 4; i++) quad.push(...sourceToCanvas(geom, m.quad[i * 2], m.quad[i * 2 + 1]));
  return { geom, motion: m, x, y, k: placed.scale * m.size, r: placed.rotation + ((clip.flipH ? -1 : 1) * m.rot) / RAD, quad };
}

/** Final on-canvas pose of a tracked clip's target at project time t (the clip's own lock-on included). */
export function targetPose(project: Project, clip: Clip, t: number, layers?: Layer[]): TargetPose | null {
  if (!clip.tracking?.segments.length) return null;
  const aspect = assetAspect(project, clip);
  const live = layers?.find((l) => l.clip.id === clip.id && !l.tail);
  if (live) {
    const m = motionAt(clip.tracking, live.srcTime, aspect);
    return m ? poseFrom(project, clip, live, m) : null; // layers handed in are already locked on
  }
  // the target is not on screen at t (hidden track, or the attached layer outlasts it): hold its nearest frame
  const lt = Math.min(clip.duration, Math.max(0, t - clip.start));
  const m = motionAt(clip.tracking, clip.inPoint + lt * clip.speed, aspect);
  if (!m) return null;
  const placed = { x: clipValue(clip, 'x', lt), y: clipValue(clip, 'y', lt), scale: clipValue(clip, 'scale', lt), rotation: clipValue(clip, 'rotation', lt) };
  if (m.locked) lockOn(project, placed, clip, m, aspect);
  applyPointLock(project, placed, clip, clip.inPoint + lt * clip.speed, aspect);
  return poseFrom(project, clip, placed, m);
}

/** The clip an attachment follows: the original target, or whichever part of it (after a split) covers time t. */
export function attachTarget(project: Project, a: ClipAttach, t: number): Clip | null {
  let fallback: Clip | null = null;
  let best: Clip | null = null;
  let bestD = Infinity;
  for (const c of project.clips) {
    if (c.id !== a.targetId && c.tracking?.id !== a.trackingId) continue;
    if (!c.tracking?.segments.length) continue;
    if (c.id === a.targetId) fallback = c;
    const d = t < c.start ? c.start - t : t > c.start + c.duration ? t - c.start - c.duration : 0;
    if (d < bestD || (d === bestD && c.id === a.targetId)) {
      bestD = d;
      best = c;
    }
  }
  return best ?? fallback;
}

/** 2×2 matrix by which an attached layer's own offset is scaled / rotated right now. */
function carry(a: ClipAttach, p: TargetPose): { m00: number; m01: number; m10: number; m11: number; ratio: number; dth: number } {
  const ratio = a.scale && a.k0 > 1e-9 ? p.k / a.k0 : 1;
  const dth = a.rotation ? p.r - a.r0 : 0;
  const c = Math.cos(dth * RAD) * ratio, s = Math.sin(dth * RAD) * ratio;
  return { m00: c, m01: -s, m10: s, m11: c, ratio, dth };
}

/** Apply lock-on and attachments to the layers of one frame (in place). */
export function applyTracking(project: Project, layers: Layer[], t: number) {
  let any = false;
  for (const l of layers) {
    const tr = l.clip.tracking;
    if (l.clip.attach) any = true;
    if (l.clip.type !== 'video') continue;
    const aspect = assetAspect(project, l.clip);
    if (tr?.segments.length) {
      const m = motionAt(tr, l.srcTime, aspect);
      if (m && m.locked) lockOn(project, l, l.clip, m, aspect);
    }
    applyPointLock(project, l, l.clip, l.srcTime, aspect);
  }
  if (!any) return;
  const W = project.width, H = project.height;
  const poses = new Map<string, TargetPose | null>();
  for (const l of layers) {
    const a = l.clip.attach;
    if (!a) continue;
    const target = attachTarget(project, a, t);
    if (!target || target.id === l.clip.id) continue;
    let p = poses.get(target.id);
    if (p === undefined) {
      p = targetPose(project, target, t, layers);
      poses.set(target.id, p);
    }
    if (!p) continue;
    if (a.cornerPin) {
      l.quad = p.quad;
      l.x = ((p.quad[0] + p.quad[2] + p.quad[4] + p.quad[6]) / 4 - W / 2) / W;
      l.y = ((p.quad[1] + p.quad[3] + p.quad[5] + p.quad[7]) / 4 - H / 2) / H;
      continue;
    }
    const k = carry(a, p);
    const ax = W / 2 + a.ax * W, ay = H / 2 + a.ay * H;
    const ox = W / 2 + l.x * W - ax, oy = H / 2 + l.y * H - ay;
    const bx = a.position ? p.x : ax, by = a.position ? p.y : ay;
    l.x = (bx + ox * k.m00 + oy * k.m01 - W / 2) / W;
    l.y = (by + ox * k.m10 + oy * k.m11 - H / 2) / H;
    l.scale *= k.ratio;
    l.rotation += k.dth;
  }
}

/** Build the attachment that makes `clip` ride on `target` without jumping at time t. */
export function makeAttach(project: Project, target: Clip, t: number, layers?: Layer[]): ClipAttach | null {
  const p = targetPose(project, target, t, layers);
  if (!p || !target.tracking) return null;
  return { targetId: target.id, trackingId: target.tracking.id, ax: (p.x - project.width / 2) / project.width, ay: (p.y - project.height / 2) / project.height, k0: p.k, r0: p.r, position: true, scale: true, rotation: true, cornerPin: false };
}

/** Convert an on-canvas drag (project px) into the change of the attached clip's own x / y that produces it. */
export function attachDragDelta(project: Project, clip: Clip, t: number, dx: number, dy: number, layers?: Layer[]): [number, number] {
  const a = clip.attach;
  if (!a || a.cornerPin) return [dx, dy];
  const target = attachTarget(project, a, t);
  const p = target ? targetPose(project, target, t, layers) : null;
  if (!p) return [dx, dy];
  const k = carry(a, p);
  const det = k.m00 * k.m11 - k.m01 * k.m10;
  if (Math.abs(det) < 1e-9) return [dx, dy];
  return [(dx * k.m11 - dy * k.m01) / det, (dy * k.m00 - dx * k.m10) / det];
}

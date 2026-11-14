// Semantic trackers: face, hands and body. MediaPipe landmark models (the same task files the reference app bundles)
// run on the main thread with the GPU delegate; this module turns their landmarks into the common tracking result —
// a quad that follows an entity, a single landmark, or a user-drawn area riding on the entity's rigid motion.
import type { FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { TrackSelectionMode } from '../../types';
import { resolveModelUrl } from '../modelUrl';
import type { StepResult } from './client';
import { fitModel, quadCx, quadCy } from './math';
import { POINT_LANDMARK } from './pointTracker';

export type SemanticKind = 'face' | 'hands' | 'body';

export interface LandmarkDef {
  id: string;
  label: string;
  /** landmark indices that are averaged */
  idx: number[];
}

export const LANDMARKS: Record<SemanticKind, LandmarkDef[]> = {
  face: [
    { id: 'nose', label: 'Nose tip', idx: [1] },
    { id: 'brow', label: 'Between the eyes', idx: [168] },
    { id: 'eyeL', label: 'Eye (screen left)', idx: [33, 133] },
    { id: 'eyeR', label: 'Eye (screen right)', idx: [362, 263] },
    { id: 'mouth', label: 'Mouth', idx: [13, 14] },
    { id: 'chin', label: 'Chin', idx: [152] },
    { id: 'forehead', label: 'Forehead', idx: [10] },
    { id: 'cheekL', label: 'Cheek (screen left)', idx: [234] },
    { id: 'cheekR', label: 'Cheek (screen right)', idx: [454] },
  ],
  hands: [
    { id: 'palm', label: 'Palm', idx: [0, 5, 9, 13, 17] },
    { id: 'wrist', label: 'Wrist', idx: [0] },
    { id: 'thumb', label: 'Thumb tip', idx: [4] },
    { id: 'index', label: 'Index fingertip', idx: [8] },
    { id: 'middle', label: 'Middle fingertip', idx: [12] },
    { id: 'ring', label: 'Ring fingertip', idx: [16] },
    { id: 'pinky', label: 'Little fingertip', idx: [20] },
  ],
  body: [
    { id: 'nose', label: 'Head', idx: [0] },
    { id: 'chest', label: 'Chest', idx: [11, 12] },
    { id: 'torso', label: 'Torso', idx: [11, 12, 23, 24] },
    { id: 'hips', label: 'Hips', idx: [23, 24] },
    { id: 'shoulderL', label: 'Shoulder (left)', idx: [11] },
    { id: 'shoulderR', label: 'Shoulder (right)', idx: [12] },
    { id: 'elbowL', label: 'Elbow (left)', idx: [13] },
    { id: 'elbowR', label: 'Elbow (right)', idx: [14] },
    { id: 'wristL', label: 'Wrist (left)', idx: [15] },
    { id: 'wristR', label: 'Wrist (right)', idx: [16] },
    { id: 'kneeL', label: 'Knee (left)', idx: [25] },
    { id: 'kneeR', label: 'Knee (right)', idx: [26] },
    { id: 'ankleL', label: 'Ankle (left)', idx: [27] },
    { id: 'ankleR', label: 'Ankle (right)', idx: [28] },
  ],
};

const MODELS: Record<SemanticKind, { local: string; remote: string }> = {
  face: { local: '/mediapipe/models/face_landmarker.task', remote: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' },
  hands: { local: '/mediapipe/models/hand_landmarker.task', remote: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' },
  body: { local: '/mediapipe/models/pose_landmarker_lite.task', remote: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task' },
};

type AnyLandmarker = FaceLandmarker | HandLandmarker | PoseLandmarker;
const loaded: Partial<Record<SemanticKind, AnyLandmarker>> = {};
const loading: Partial<Record<SemanticKind, Promise<void>>> = {};
let clock = 0;

export const landmarkerReady = (kind: SemanticKind) => !!loaded[kind];

export function ensureLandmarker(kind: SemanticKind): Promise<void> {
  if (loaded[kind]) return Promise.resolve();
  if (!loading[kind]) {
    loading[kind] = (async () => {
      const mp = await import('@mediapipe/tasks-vision');
      const files = await mp.FilesetResolver.forVisionTasks('/mediapipe/wasm');
      const modelAssetPath = await resolveModelUrl(MODELS[kind].local, MODELS[kind].remote);
      const make = (delegate: 'GPU' | 'CPU'): Promise<AnyLandmarker> => {
        const baseOptions = { modelAssetPath, delegate };
        if (kind === 'face') return mp.FaceLandmarker.createFromOptions(files, { baseOptions, runningMode: 'VIDEO', numFaces: 4, minFaceDetectionConfidence: 0.4, minFacePresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
        if (kind === 'hands') return mp.HandLandmarker.createFromOptions(files, { baseOptions, runningMode: 'VIDEO', numHands: 4, minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
        return mp.PoseLandmarker.createFromOptions(files, { baseOptions, runningMode: 'VIDEO', numPoses: 3, minPoseDetectionConfidence: 0.4, minPosePresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
      };
      try {
        loaded[kind] = await make('GPU');
      } catch {
        loaded[kind] = await make('CPU');
      }
    })();
    loading[kind]!.catch(() => delete loading[kind]);
  }
  return loading[kind]!;
}

export interface SemanticEntity {
  /** landmarks, x,y pairs normalised to the frame */
  pts: Float32Array;
  vis: Float32Array | null;
  /** oriented box around the entity, normalised */
  quad: number[];
  cx: number;
  cy: number;
  /** orientation of the entity's own horizontal axis, radians (square pixels) */
  angle: number;
  /** √area of the box in frame heights */
  size: number;
}

function axis(kind: SemanticKind, p: Float32Array, a: number): [number, number] {
  const X = (i: number) => p[i * 2] * a;
  const Y = (i: number) => p[i * 2 + 1];
  let ex = 1, ey = 0;
  if (kind === 'face') {
    ex = X(263) - X(33);
    ey = Y(263) - Y(33);
  } else {
    // "up" runs wrist → middle knuckle (hand) or hips → shoulders (body); the horizontal axis is perpendicular to it
    const ux = kind === 'hands' ? X(9) - X(0) : (X(11) + X(12)) / 2 - (X(23) + X(24)) / 2;
    const uy = kind === 'hands' ? Y(9) - Y(0) : (Y(11) + Y(12)) / 2 - (Y(23) + Y(24)) / 2;
    ex = -uy;
    ey = ux;
  }
  const n = Math.hypot(ex, ey);
  return n > 1e-6 ? [ex / n, ey / n] : [1, 0];
}

function describe(kind: SemanticKind, pts: Float32Array, vis: Float32Array | null, a: number): SemanticEntity {
  const [ex, ey] = axis(kind, pts, a);
  const fx = -ey, fy = ex;
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  const n = Math.min(pts.length >> 1, kind === 'face' ? 468 : 999);
  let used = 0;
  for (let pass = 0; pass < 2 && used < 4; pass++) {
    u0 = v0 = Infinity;
    u1 = v1 = -Infinity;
    used = 0;
    for (let i = 0; i < n; i++) {
      if (pass === 0 && vis && vis[i] < 0.35) continue;
      const x = pts[i * 2] * a, y = pts[i * 2 + 1];
      const u = x * ex + y * ey, v = x * fx + y * fy;
      if (u < u0) u0 = u;
      if (u > u1) u1 = u;
      if (v < v0) v0 = v;
      if (v > v1) v1 = v;
      used++;
    }
  }
  const pad = 0.06;
  const w = (u1 - u0) * (1 + pad * 2), h = (v1 - v0) * (1 + pad * 2);
  const cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
  const cx = cu * ex + cv * fx, cy = cu * ey + cv * fy;
  const quad: number[] = [];
  for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) quad.push((cx + (sx * w * ex) / 2 + (sy * h * fx) / 2) / a, cy + (sx * w * ey) / 2 + (sy * h * fy) / 2);
  return { pts, vis, quad, cx: cx / a, cy, angle: Math.atan2(ey, ex), size: Math.sqrt(Math.max(1e-8, w * h)) };
}

/** Run the landmark model on a frame. Returns [] until the model has loaded. */
export function detectEntities(kind: SemanticKind, source: TexImageSource, aspect: number): SemanticEntity[] {
  const lm = loaded[kind];
  if (!lm) return [];
  clock = Math.max(clock + 34, Math.floor(performance.now()));
  let groups: { x: number; y: number; visibility?: number }[][] = [];
  try {
    if (kind === 'face') groups = (lm as FaceLandmarker).detectForVideo(source as never, clock).faceLandmarks ?? [];
    else if (kind === 'hands') groups = (lm as HandLandmarker).detectForVideo(source as never, clock).landmarks ?? [];
    else groups = (lm as PoseLandmarker).detectForVideo(source as never, clock).landmarks ?? [];
  } catch (e) {
    console.warn('landmark detection failed', e);
    return [];
  }
  const out: SemanticEntity[] = [];
  for (const g of groups) {
    if (!g || g.length < 5) continue;
    const pts = new Float32Array(g.length * 2);
    const vis = kind === 'body' ? new Float32Array(g.length) : null;
    for (let i = 0; i < g.length; i++) {
      pts[i * 2] = g[i].x;
      pts[i * 2 + 1] = g[i].y;
      if (vis) vis[i] = g[i].visibility ?? 1;
    }
    out.push(describe(kind, pts, vis, aspect));
  }
  return out;
}

export function landmarkPoint(e: SemanticEntity, def: LandmarkDef): [number, number] {
  let x = 0, y = 0, n = 0;
  for (const i of def.idx) {
    if (i * 2 + 1 >= e.pts.length) continue;
    x += e.pts[i * 2];
    y += e.pts[i * 2 + 1];
    n++;
  }
  return n ? [x / n, y / n] : [e.cx, e.cy];
}

export function nearestEntity(list: SemanticEntity[], x: number, y: number, aspect: number, gate = Infinity): number {
  let best = -1, bd = gate;
  for (let i = 0; i < list.length; i++) {
    const d = Math.hypot((list[i].cx - x) * aspect, list[i].cy - y);
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  return best;
}

const RIGID: Record<SemanticKind, (n: number) => number[]> = {
  face: (n) => Array.from({ length: Math.floor(Math.min(n, 468) / 6) }, (_, i) => i * 6),
  hands: () => [0, 1, 5, 9, 13, 17],
  body: () => [11, 12, 23, 24],
};

export class SemanticTracker {
  private kind: SemanticKind;
  private mode: TrackSelectionMode;
  private def: LandmarkDef;
  private a: number;
  private ref: Float64Array | null = null;
  private refQuad: number[] = [];
  private rigid: Int32Array = new Int32Array(0);
  private last: StepResult | null = null;
  private cx = 0;
  private cy = 0;
  private size = 0.2;
  private searching = 0;

  constructor(kind: SemanticKind, mode: TrackSelectionMode, landmark: string, aspect: number) {
    this.kind = kind;
    this.mode = mode;
    this.def = LANDMARKS[kind].find((l) => l.id === landmark) ?? LANDMARKS[kind][0];
    this.a = aspect;
  }

  private toPx(e: SemanticEntity): Float64Array {
    const out = new Float64Array(e.pts.length);
    for (let i = 0; i < e.pts.length; i += 2) {
      out[i] = e.pts[i] * this.a;
      out[i + 1] = e.pts[i + 1];
    }
    return out;
  }

  private pointQuad(e: SemanticEntity): number[] {
    const [x, y] = landmarkPoint(e, this.def);
    const half = Math.max(0.012, e.size * 0.11);
    const c = Math.cos(e.angle) * half, s = Math.sin(e.angle) * half;
    const out: number[] = [];
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) out.push(x + (sx * c - sy * s) / this.a, y + sx * s + sy * c);
    return out;
  }

  private result(e: SemanticEntity, quad: number[]): StepResult {
    const marks = LANDMARKS[this.kind];
    const points = new Float32Array(marks.length * 3);
    marks.forEach((m, i) => {
      const [x, y] = landmarkPoint(e, m);
      points.set([x, y, POINT_LANDMARK], i * 3);
    });
    this.cx = e.cx;
    this.cy = e.cy;
    this.size = e.size;
    return { quad, roi: this.mode === 'area' ? quad : e.quad, conf: 1, state: 0, lost: false, points };
  }

  /** `near` chooses which detected entity to follow; `region` is the user-drawn area (area mode). */
  init(entities: SemanticEntity[], near: [number, number], region: number[]): { error: string | null; first: StepResult | null } {
    const i = nearestEntity(entities, near[0], near[1], this.a);
    if (i < 0) return { error: 'no-entity', first: null };
    const e = entities[i];
    this.ref = this.toPx(e);
    this.rigid = new Int32Array(RIGID[this.kind](e.pts.length >> 1).filter((k) => k * 2 + 1 < e.pts.length));
    this.refQuad = region.map((v, k) => (k % 2 ? v : v * this.a));
    this.searching = 0;
    const quad = this.mode === 'entity' ? e.quad.slice() : this.mode === 'points' ? this.pointQuad(e) : region.slice();
    this.last = this.result(e, quad);
    return { error: null, first: this.last };
  }

  step(entities: SemanticEntity[]): StepResult {
    const i = nearestEntity(entities, this.cx, this.cy, this.a, Math.max(0.12, this.size * 1.4));
    if (i < 0 || !this.ref || !this.last) {
      this.searching++;
      const held = this.last ?? { quad: [0, 0, 1, 0, 1, 1, 0, 1], roi: [0, 0, 1, 0, 1, 1, 0, 1], conf: 0, state: 1 as const, lost: true, points: null };
      return { ...held, conf: 0, state: 1, lost: this.searching > 75, points: null };
    }
    this.searching = 0;
    const e = entities[i];
    let quad: number[];
    if (this.mode === 'entity') quad = e.quad.slice();
    else if (this.mode === 'points') quad = this.pointQuad(e);
    else {
      const S = fitModel('similarity', this.ref, this.toPx(e), this.rigid, this.rigid.length);
      if (S) {
        quad = [];
        for (let k = 0; k < 4; k++) {
          const x = this.refQuad[k * 2], y = this.refQuad[k * 2 + 1];
          quad.push((S[0] * x + S[1] * y + S[2]) / this.a, S[3] * x + S[4] * y + S[5]);
        }
      } else {
        const dx = e.cx - quadCx(this.last.quad), dy = e.cy - quadCy(this.last.quad);
        quad = this.last.quad.map((v, k) => v + (k % 2 ? dy : dx));
      }
    }
    this.last = this.result(e, quad);
    return this.last;
  }
}

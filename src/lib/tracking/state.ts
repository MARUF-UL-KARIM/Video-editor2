// Transient UI state of the tracking tool (selector geometry, analysis progress, live results).
// Committed tracking data lives on the clip in the project store; nothing here is persisted.
import { create } from 'zustand';
import type { ClipTracking, TrackLock, TrackMoveMode, TrackRegionShape, TrackSettings } from '../../types';
import { quadAngle, quadCx, quadCy, rectQuad } from './math';
import { DEFAULT_SETTINGS } from './motion';
import type { SemanticEntity, SemanticKind } from './semantic';

/** Mirrors the analysis life-cycle of the reference tracker. */
export type TrackPhase = 'idle' | 'preparing' | 'analyzing' | 'stopping' | 'committing';
export type TrackTab = 'track' | 'adjust' | 'edit' | 'attach';

export interface LiveTrack {
  clipId: string;
  t: number[];
  q: number[];
  c: number[];
  s: number[];
  /** selector region at the newest frame */
  roi: number[];
  /** x, y, status triples at the newest frame, normalised */
  points: Float32Array | null;
  state: 0 | 1;
}

export interface Detection {
  clipId: string;
  kind: SemanticKind;
  srcTime: number;
  entities: SemanticEntity[];
}

interface TrackUI {
  /** the tracking panel is open */
  active: boolean;
  tab: TrackTab;
  settings: TrackSettings;
  /** selector quad per clip, normalised source coordinates */
  regions: Record<string, number[]>;
  phase: TrackPhase;
  status: string;
  progress: number;
  frames: number;
  speed: number;
  live: LiveTrack | null;
  tick: number;
  moveMode: TrackMoveMode;
  applyAll: boolean;
  /** tracking as it was when Edit Motion was opened, for Reset */
  snapshot: { clipId: string; tracking: ClipTracking | undefined } | null;
  detected: Detection | null;
  detecting: boolean;
  /** show feature points / motion path on the preview */
  showPoints: boolean;
  /** lock point chosen before a track exists (committed lock lives on the clip) */
  lockDraft: (TrackLock & { clipId: string }) | null;
  patch: (p: Partial<TrackUI>) => void;
  setSettings: (p: Partial<TrackSettings>) => void;
  setRegion: (clipId: string, quad: number[]) => void;
}

export const useTrack = create<TrackUI>((set) => ({
  active: false,
  tab: 'track',
  settings: { ...DEFAULT_SETTINGS },
  regions: {},
  phase: 'idle',
  status: '',
  progress: 0,
  frames: 0,
  speed: 0,
  live: null,
  tick: 0,
  moveMode: 'current',
  applyAll: true,
  snapshot: null,
  detected: null,
  detecting: false,
  showPoints: true,
  lockDraft: null,
  patch: (p) => set(p),
  setSettings: (p) => set((s) => ({ settings: { ...s.settings, ...p } })),
  setRegion: (clipId, quad) => set((s) => ({ regions: { ...s.regions, [clipId]: quad } })),
}));

export const trackState = () => useTrack.getState();

export const isSemantic = (k: TrackSettings['tracker']): k is SemanticKind => k === 'face' || k === 'hands' || k === 'body';

export const FULL_FRAME = [0, 0, 1, 0, 1, 1, 0, 1];

/** A centred square (in pixels) covering ~28 % of the short side. */
export function defaultRegion(aspect: number): number[] {
  const side = 0.28 * Math.min(aspect, 1);
  return rectQuad(0.5, 0.5, side / aspect, side, 0, aspect);
}

/** The dual tracker's quad encodes its two anchors: A is the middle of the left edge, B the middle of the right edge. */
export function dualQuad(ax: number, ay: number, bx: number, by: number, aspect: number): number[] {
  const AX = ax * aspect, BX = bx * aspect;
  const cx = (AX + BX) / 2, cy = (ay + by) / 2;
  const ux = (BX - AX) / 2, uy = (by - ay) / 2;
  const vx = -uy, vy = ux;
  return [(cx - ux - vx) / aspect, cy - uy - vy, (cx + ux - vx) / aspect, cy + uy - vy, (cx + ux + vx) / aspect, cy + uy + vy, (cx - ux + vx) / aspect, cy - uy + vy];
}
export function dualPoints(q: ArrayLike<number>): [number, number, number, number] {
  return [(q[0] + q[6]) / 2, (q[1] + q[7]) / 2, (q[2] + q[4]) / 2, (q[3] + q[5]) / 2];
}

/** Rectangle parameters (pixel-true) of a quad: centre, edge lengths in normalised units, rotation. */
export function rectOf(q: ArrayLike<number>, aspect: number) {
  const cx = quadCx(q), cy = quadCy(q);
  const wpx = (Math.hypot((q[2] - q[0]) * aspect, q[3] - q[1]) + Math.hypot((q[4] - q[6]) * aspect, q[5] - q[7])) / 2;
  const hpx = (Math.hypot((q[6] - q[0]) * aspect, q[7] - q[1]) + Math.hypot((q[4] - q[2]) * aspect, q[5] - q[3])) / 2;
  return { cx, cy, w: wpx / aspect, h: hpx, rot: quadAngle(q, 0, aspect) };
}

/** Make a selector quad obey the constraints of a shape (circle: upright; rectangle: right angles). */
export function conformRegion(q: number[], shape: TrackRegionShape, aspect: number): number[] {
  if (shape === 'quad') return q;
  const r = rectOf(q, aspect);
  return rectQuad(r.cx, r.cy, r.w, r.h, shape === 'circle' ? 0 : r.rot, aspect);
}

/** Settings as the engine will actually use them for a given tracker (illegal combinations resolved). */
export function effectiveSettings(s: TrackSettings): TrackSettings {
  const o = { ...s };
  if (o.tracker === 'object') {
    if (o.shape === 'quad') o.shape = 'rectangle';
    o.followRotation = false;
    o.cornerPin = false;
    o.model = 'similarity';
    if (o.mask === 'background') o.mask = 'none';
  } else if (o.tracker === 'features') {
    o.model = 'similarity';
    o.cornerPin = false;
  } else if (o.tracker === 'frame') {
    o.shape = 'quad';
    o.cornerPin = true;
    if (o.model === 'translation') o.model = 'similarity';
    if (o.mask === 'subject') o.mask = 'none';
  } else if (o.tracker === 'dual') {
    o.shape = 'rectangle';
    o.model = 'similarity';
    o.mask = 'none';
    o.cornerPin = false;
  } else if (isSemantic(o.tracker)) {
    o.mask = 'none';
    o.model = 'similarity';
    o.cornerPin = false;
    if (o.shape === 'quad') o.shape = 'rectangle';
  } else {
    // flow
    if (o.shape !== 'quad') o.cornerPin = false;
    if (o.cornerPin) {
      o.followSize = true;
      o.followRotation = true;
    }
    if (o.shape === 'circle') o.followRotation = false;
  }
  return o;
}

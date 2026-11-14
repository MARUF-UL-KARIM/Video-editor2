// Point-lock stabilizer. One reticle chooses where the tracker middle should sit; turning Lock on
// freezes that point and the picture is counter-moved so the middle never leaves it.
import type { Clip, ClipTracking, TrackLock } from '../../types';
import { engine } from '../engine';
import { getState } from '../store';
import { pointLockOf, trackerCentreCanvas } from './resolve';
import { trackState, useTrack } from './state';

export type LockDraft = TrackLock & { clipId: string };

const clampP = (v: number) => Math.min(0.48, Math.max(-0.48, v));

export function lockOf(clip: Clip): TrackLock | null {
  return pointLockOf(clip);
}

function write(clipId: string, next: TrackLock, history?: boolean) {
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip) return;
  if (clip.tracking?.segments.length) {
    st.updateClip(clipId, { tracking: { ...clip.tracking, lock: next } }, history === false ? { history: false } : undefined);
    if (trackState().lockDraft?.clipId === clipId) trackState().patch({ lockDraft: null });
    return;
  }
  trackState().patch({ lockDraft: { clipId, ...next } });
  // draft is not project data, so nudge the preview ourselves
  if (!st.playing) engine.seek(st.time);
}

/** Move the lock point. Marks it as placed so it no longer follows the tracker. */
export function placeLock(clipId: string, x: number, y: number, opts?: { history?: boolean; snap?: boolean }) {
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip) return;
  let nx = x;
  let ny = y;
  if (opts?.snap) {
    if (Math.abs(nx) < 0.012) nx = 0;
    if (Math.abs(ny) < 0.012) ny = 0;
  }
  const prev = lockOf(clip);
  write(clipId, { enabled: prev?.enabled ?? false, x: clampP(nx), y: clampP(ny), placed: true }, opts?.history);
}

/**
 * Turn Lock on or off. Enabling sets the main point: a point you already placed is kept,
 * otherwise the tracker middle (where it sits right now) is captured so the shot does not jump.
 */
export function toggleLock(clipId: string, on: boolean) {
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip) return;
  const prev = lockOf(clip);
  let x = prev?.x ?? 0;
  let y = prev?.y ?? 0;
  let placed = !!prev?.placed;
  if (on && !placed) {
    const at = trackerCentreCanvas(st.project, clip, st.time);
    if (at) {
      x = at.x;
      y = at.y;
    }
    placed = true;
  }
  write(clipId, { enabled: on, x: clampP(x), y: clampP(y), placed });
  if (on) {
    const tracked = !!clip.tracking?.segments.length || trackState().live?.clipId === clipId;
    st.toast(tracked ? 'Lock on — tracker centre held' : 'Lock point set. Track a target to stabilize.', 'ok');
  }
}

/** Put the lock point on the tracker middle of the current frame (stabilize in place from here). */
export function anchorToTracker(clipId: string) {
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip) return;
  const at = trackerCentreCanvas(st.project, clip, st.time);
  if (!at) {
    st.toast('Nothing tracked at the playhead yet');
    return;
  }
  const prev = lockOf(clip);
  write(clipId, { enabled: prev?.enabled ?? false, x: clampP(at.x), y: clampP(at.y), placed: true });
}

/** Put the lock point at the frame centre. */
export function anchorToCentre(clipId: string) {
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip) return;
  const prev = lockOf(clip);
  write(clipId, { enabled: prev?.enabled ?? false, x: 0, y: 0, placed: true });
}

/** Fold a pre-track lock point into a tracking object that is about to be saved. */
export function absorbLockDraft(clipId: string, tracking: ClipTracking): ClipTracking {
  const draft = trackState().lockDraft;
  if (!draft || draft.clipId !== clipId) return tracking;
  useTrack.getState().patch({ lockDraft: null });
  if (tracking.lock) return tracking;
  return { ...tracking, lock: { enabled: draft.enabled, x: draft.x, y: draft.y, placed: draft.placed } };
}

/** Keep the lock point when tracking data itself is cleared. */
export function stashLock(clip: Clip) {
  const lock = clip.tracking?.lock;
  if (!lock) return;
  trackState().patch({ lockDraft: { clipId: clip.id, ...lock } });
}

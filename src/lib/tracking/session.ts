// Analysis sessions: decode the clip frame by frame from the playhead, run the chosen tracker, show live results and
// commit a tracking segment to the clip. Mirrors the reference life-cycle:
//   preparing → analyzing → (stopping) → committing → idle
import type { Clip, ClipTracking, TrackSegment, TrackSettings } from '../../types';
import { collectLayers } from '../anim';
import { engine } from '../engine';
import { runtime } from '../runtime';
import { ensureSegmenter, segmentStill, segmenterState } from '../segmenter';
import { getState } from '../store';
import { disposeTracker, initTracker, stepTracker, warmObjectTracker, type MaskPayload, type StepResult } from './client';
import { sampleEdgePreview } from './edgePreview';
import { openFrameSource, type FrameSource } from './frames';
import { quadCx, quadCy } from './math';
import { absorbLockDraft } from './lock';
import { addSegment, DEFAULT_ADJUST, makeSegment, segEnd, segmentAt, segStart } from './motion';
import { assetAspect, makeAttach } from './resolve';
import { detectEntities, ensureLandmarker, landmarkerReady, SemanticTracker } from './semantic';
import { conformRegion, defaultRegion, dualQuad, effectiveSettings, FULL_FRAME, isSemantic, trackState, useTrack, type LiveTrack } from './state';

const MESSAGES: Record<string, string> = {
  'low-detail': 'Low detail. Try a different area or adjust settings',
  'no-entity': 'The selected subject or points could not be tracked. Choose a clearer frame and select them again.',
  'no-colour': 'Tracker failed. Try a different area or adjust settings',
};

let cancel = false;

const frameTick = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

/** Source time (seconds into the media file) under the playhead for a clip. */
export function sourceTimeOf(clip: Clip, t: number): number {
  return clip.inPoint + Math.min(clip.duration, Math.max(0, t - clip.start)) * clip.speed;
}
export const projectTimeOf = (clip: Clip, src: number) => clip.start + (src - clip.inPoint) / clip.speed;

/** The selector for a clip as the chosen tracker wants it. */
export function regionFor(clip: Clip, aspect: number, settings: TrackSettings = trackState().settings): number[] {
  const s = effectiveSettings(settings);
  if (s.tracker === 'frame') return FULL_FRAME.slice();
  const saved = trackState().regions[clip.id];
  if (s.tracker === 'dual') {
    if (saved && saved.length === 8) {
      // keep whatever the user placed: its left / right edge midpoints are the two anchors
      return dualQuad((saved[0] + saved[6]) / 2, (saved[1] + saved[7]) / 2, (saved[2] + saved[4]) / 2, (saved[3] + saved[5]) / 2, aspect);
    }
    return dualQuad(0.4, 0.5, 0.6, 0.5, aspect);
  }
  return conformRegion(saved ?? defaultRegion(aspect), s.shape, aspect);
}

export function stopTracking() {
  const ui = trackState();
  if (ui.phase === 'analyzing' || ui.phase === 'preparing') {
    cancel = true;
    ui.patch({ phase: 'stopping', status: 'Stopping…' });
  }
}

export interface TrackRun {
  /** stop at this source time instead of the end of the clip */
  until?: number;
  /** start here instead of at the playhead (source time) */
  from?: number;
  region?: number[];
}

/** Track the selected region forward from the playhead. Resolves when the segment has been committed. */
export async function startTracking(clipId: string, run: TrackRun = {}): Promise<boolean> {
  const ui = trackState();
  if (ui.phase !== 'idle') return false;
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip || clip.type !== 'video' || !clip.assetId) {
    st.toast('Select a video clip to track', 'error');
    return false;
  }
  const asset = st.project.assets.find((a) => a.id === clip.assetId);
  if (!asset || !runtime.has(asset.id)) {
    st.toast('This clip is offline — re-import its media to track it', 'error');
    return false;
  }
  engine.pause();
  const settings = effectiveSettings(ui.settings);
  const aspect = assetAspect(st.project, clip);
  const srcEnd = Math.min(clip.inPoint + clip.duration * clip.speed, asset.duration || Infinity);
  const srcStart = run.from ?? sourceTimeOf(clip, st.time);
  const limit = Math.min(srcEnd, run.until ?? srcEnd);
  if (limit - srcStart < 2 / 30) {
    st.toast('Not enough video here. Move the playhead earlier.', 'error');
    return false;
  }
  const region = run.region ?? regionFor(clip, aspect, settings);
  const semantic = isSemantic(settings.tracker) ? settings.tracker : null;

  cancel = false;
  ui.patch({ phase: 'preparing', status: 'Preparing…', progress: 0, frames: 0, speed: 0, live: null, detected: null });
  let source: FrameSource | null = null;
  let live: LiveTrack | null = null;
  let lost = false;
  let failure: string | null = null;

  try {
    if (semantic) {
      ui.patch({ status: 'Loading the landmark model…' });
      await ensureLandmarker(semantic);
    } else if (settings.tracker === 'object' && !settings.edgeTrack) {
      ui.patch({ status: 'Loading the object tracker…' });
      await warmObjectTracker();
    }
    let useMask = settings.mask !== 'none' && !semantic;
    if (useMask) {
      ui.patch({ status: 'Preparing the tracking mask…' });
      await ensureSegmenter();
      if (segmenterState() !== 'ready') {
        useMask = false;
        st.toast('Tracking masks are unavailable. Continuing without a mask.', 'error');
      }
    }
    ui.patch({ status: 'Opening the clip…' });
    source = await openFrameSource(asset, srcStart, settings.resolution);
    const t0 = await source.next();
    if (t0 === null) throw new Error('No frames could be read from this clip');
    const mask = (): MaskPayload | null => (useMask ? segmentStill(source!.canvas) : null);

    let tracker: SemanticTracker | null = null;
    let first: StepResult | null = null;
    let initPoints: Float32Array | null = null;
    if (semantic) {
      tracker = new SemanticTracker(semantic, settings.selection, settings.landmark, aspect);
      // the landmark graph needs a frame or two to warm up on a new shot
      let entities = detectEntities(semantic, source.canvas, aspect);
      if (!entities.length) entities = detectEntities(semantic, source.canvas, aspect);
      const r = tracker.init(entities, [quadCx(region), quadCy(region)], region);
      failure = r.error;
      first = r.first;
    } else {
      const initRes = await initTracker(settings, region, source.read(), mask());
      failure = initRes.error;
      initPoints = initRes.points ?? null;
    }
    if (failure) throw new Error(MESSAGES[failure] ?? failure);

    const q0 = first ? first.quad : region;
    live = { clipId, t: [t0], q: q0.slice(), c: [1], s: [0], roi: first ? first.roi : region, points: first?.points ?? initPoints, state: 0 };
    ui.patch({ phase: 'analyzing', status: settings.edgeTrack ? 'Edge tracking' : 'Analyzing', live, tick: 1 });

    const began = performance.now();
    let lastUi = 0;
    let n = 0;
    while (!cancel) {
      const t = await source.next();
      if (t === null || t > limit + 1e-4) break;
      n++;
      let r: StepResult;
      if (tracker) r = tracker.step(detectEntities(semantic!, source.canvas, aspect));
      else r = await stepTracker(source.read(), useMask && n % 6 === 0 ? mask() : undefined);
      live.t.push(t);
      for (let k = 0; k < 8; k++) live.q.push(r.quad[k]);
      live.c.push(r.conf);
      live.s.push(r.state);
      live.roi = r.roi;
      live.points = r.points;
      live.state = r.state;
      if (r.lost) {
        lost = true;
        break;
      }
      const now = performance.now();
      if (now - lastUi > 90) {
        lastUi = now;
        useTrack.setState((s) => ({ tick: s.tick + 1, frames: n, progress: Math.min(1, (t - srcStart) / (limit - srcStart)), speed: (n * 1000) / Math.max(1, now - began), status: r.state ? 'Searching' : settings.edgeTrack ? 'Edge tracking' : 'Analyzing' }));
        engine.seek(projectTimeOf(clip, t));
        await frameTick();
      } else if (tracker && n % 2 === 0) await frameTick();
    }
  } catch (e) {
    failure = e instanceof Error ? e.message : String(e);
  } finally {
    source?.dispose();
    disposeTracker();
  }

  // ── commit ──
  trackState().patch({ phase: 'committing', status: 'Saving the track…' });
  let committed = false;
  if (live) {
    // a track never ends on frames where the target was not actually seen
    let end = live.t.length;
    while (end > 1 && live.s[end - 1] === 1) end--;
    if (end >= 2) {
      const now = getState();
      const current = now.project.clips.find((c) => c.id === clipId);
      if (current) {
        const prev = segmentAt(current.tracking, live.t[0]);
        const adjust = prev?.seg.adjust ?? current.tracking?.segments[0]?.adjust ?? DEFAULT_ADJUST;
        const seg = makeSegment(settings, live.t.slice(0, end), live.q.slice(0, end * 8), live.c.slice(0, end), live.s.slice(0, end), adjust);
        now.updateClip(clipId, { tracking: absorbLockDraft(clipId, addSegment(current.tracking, seg)) });
        committed = true;
        const lastT = live.t[end - 1];
        // leave the selector where the target was last seen, ready to adjust and continue
        if (settings.tracker !== 'frame') trackState().setRegion(clipId, live.roi.slice());
        engine.seek(projectTimeOf(current, lastT));
        if (lost) now.toast('Object lost. Adjust the tracker and continue', 'error');
        else if (cancel) now.toast(`Tracking stopped — ${end} frames kept`, 'info');
        else now.toast(`Tracked ${end} frames`, 'ok');
      }
    } else if (lost || !failure) {
      failure = failure ?? MESSAGES['low-detail'];
    }
  }
  if (failure && !committed) getState().toast(failure.startsWith('Low') || failure.startsWith('The selected') ? failure : `Tracker failed. ${failure}`, 'error');
  trackState().patch({ phase: 'idle', status: '', live: null, progress: 0 });
  cancel = false;
  return committed;
}

/** Track the segment under the playhead again, from its first frame, with the current settings. */
export async function retrack(clipId: string): Promise<boolean> {
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip?.tracking) return false;
  const hit = segmentAt(clip.tracking, sourceTimeOf(clip, st.time));
  if (!hit) return false;
  const seg: TrackSegment = hit.seg;
  const next = clip.tracking.segments[hit.index + 1];
  const aspect = assetAspect(st.project, clip);
  const settings = effectiveSettings(trackState().settings);
  const first = seg.q.slice(0, 8);
  const region = settings.tracker === 'frame' ? FULL_FRAME.slice() : settings.tracker === 'dual' ? first : conformRegion(first, settings.shape, aspect);
  engine.seek(projectTimeOf(clip, segStart(seg)));
  return startTracking(clipId, { from: segStart(seg), until: next ? segStart(next) : Math.max(segEnd(seg), clip.inPoint + clip.duration * clip.speed), region });
}

let detectSeq = 0;

/** Face / hands / body: find entities in the frame under the playhead so the user can see and pick a target. */
export async function detectPreview(clipId: string): Promise<void> {
  const ui = trackState();
  const kind = ui.settings.tracker;
  if (!isSemantic(kind) || ui.phase !== 'idle') return;
  const seq = ++detectSeq;
  ui.patch({ detecting: true });
  try {
    if (!landmarkerReady(kind)) await ensureLandmarker(kind);
    for (let attempt = 0; attempt < 12; attempt++) {
      if (seq !== detectSeq) return;
      const st = getState();
      const clip = st.project.clips.find((c) => c.id === clipId);
      const el = engine.videoFor(clipId);
      if (!clip || trackState().phase !== 'idle') return;
      if (el && el.readyState >= 2 && !el.seeking && el.videoWidth > 0) {
        const aspect = assetAspect(st.project, clip);
        let entities = detectEntities(kind, el, aspect);
        if (!entities.length) entities = detectEntities(kind, el, aspect);
        if (seq === detectSeq) trackState().patch({ detected: { clipId, kind, srcTime: sourceTimeOf(clip, st.time), entities } });
        return;
      }
      await new Promise((r) => setTimeout(r, 120));
    }
  } catch (e) {
    console.warn('landmark preview failed', e);
    getState().toast('The landmark model could not be loaded', 'error');
  } finally {
    if (seq === detectSeq) trackState().patch({ detecting: false });
  }
}

let edgePreviewSeq = 0;

/** Edge Track: run the edge-detection filter inside the tracking circle on the current frame for live dot preview. */
export async function refreshEdgePreview(clipId: string): Promise<void> {
  const ui = trackState();
  const s = effectiveSettings(ui.settings);
  if (!s.edgeTrack || ui.phase !== 'idle') {
    if (ui.edgePreview) ui.patch({ edgePreview: null });
    return;
  }
  const seq = ++edgePreviewSeq;
  for (let attempt = 0; attempt < 10; attempt++) {
    if (seq !== edgePreviewSeq) return;
    const st = getState();
    const clip = st.project.clips.find((c) => c.id === clipId);
    if (!clip || trackState().phase !== 'idle') return;
    const latest = effectiveSettings(trackState().settings);
    if (!latest.edgeTrack) return;
    const el = engine.videoFor(clipId);
    const targetSrc = sourceTimeOf(clip, st.time);
    const aligned = el ? Math.abs(el.currentTime - targetSrc) < 0.08 || attempt >= 8 : false;
    if (el && el.readyState >= 2 && !el.seeking && el.videoWidth > 0 && aligned) {
      const aspect = assetAspect(st.project, clip);
      const region = regionFor(clip, aspect, latest);
      const preview = sampleEdgePreview(el, clipId, targetSrc, region, {
        resolution: latest.resolution,
        sensitivity: latest.edgeSensitivity ?? 55,
        maxDots: latest.edgeDots ?? 32,
      });
      if (seq === edgePreviewSeq) trackState().patch({ edgePreview: preview });
      return;
    }
    await new Promise((r) => setTimeout(r, 80));
  }
}

// ───────────────────────────── attachments ─────────────────────────────

/** Make `clipId` ride on the tracked motion of `targetId`, keeping it exactly where it is right now. */
export function attachClip(clipId: string, targetId: string): boolean {
  const st = getState();
  const target = st.project.clips.find((c) => c.id === targetId);
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!target || !clip || clip.id === target.id || clip.type === 'audio') return false;
  const attach = makeAttach(st.project, target, st.time, collectLayers(st.project, st.time));
  if (!attach) {
    st.toast('Track this clip first, then attach layers to it');
    return false;
  }
  st.updateClip(clipId, { attach });
  return true;
}

/** Release a layer, leaving it where it currently appears. */
export function detachClip(clipId: string) {
  const st = getState();
  const clip = st.project.clips.find((c) => c.id === clipId);
  if (!clip?.attach) return;
  const layer = collectLayers(st.project, st.time).find((l) => l.clip.id === clipId && !l.tail);
  const patch: Partial<Clip> = { attach: undefined };
  if (layer && !clip.attach.cornerPin) {
    if (!clip.keyframes.x?.length) patch.x = layer.x;
    if (!clip.keyframes.y?.length) patch.y = layer.y;
    if (!clip.keyframes.scale?.length) patch.scale = layer.scale;
    if (!clip.keyframes.rotation?.length) patch.rotation = layer.rotation;
  }
  st.updateClip(clipId, patch);
}

/** Replace a clip's tracking (history-aware helper for the segment tools). */
export function commitTracking(clipId: string, tracking: ClipTracking | null | undefined, coalesce?: string) {
  if (tracking === null) return;
  getState().updateClip(clipId, { tracking: tracking && tracking.segments.length ? tracking : undefined }, coalesce ? { coalesce } : undefined);
}

import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Box,
  Circle,
  Crosshair,
  Eraser,
  Eye,
  EyeOff,
  Film,
  Frame,
  Hand,
  Image as ImageIcon,
  Link2,
  Lock,
  LocateFixed,
  PenLine,
  Pentagon,
  PersonStanding,
  Pin,
  Play,
  Plus,
  RectangleHorizontal,
  RefreshCw,
  RotateCcw,
  ScanFace,
  ScanSearch,
  Scissors,
  Sparkles,
  Square,
  Trash2,
  Type,
  Unlink2,
  Waypoints,
  Wind,
} from 'lucide-react';
import { useEffect, useMemo, useState, type ComponentType, type ReactNode } from 'react';
import { engine } from '../../lib/engine';
import { clipEnd, getState, useStore } from '../../lib/store';
import { connectAt, deleteAt, MOTION_COMPONENTS, motionAt, moveSample, segEnd, segmentAt, segStart, setAdjust, smoothAt, splitAt, trackingStats } from '../../lib/tracking/motion';
import { anchorToCentre, anchorToTracker, lockOf, placeLock, stashLock, toggleLock } from '../../lib/tracking/lock';
import { assetAspect, trackerCentreCanvas } from '../../lib/tracking/resolve';
import { LANDMARKS } from '../../lib/tracking/semantic';
import { attachClip, commitTracking, detachClip, detectPreview, projectTimeOf, retrack, sourceTimeOf, startTracking, stopTracking } from '../../lib/tracking/session';
import { effectiveSettings, isSemantic, useTrack, type TrackTab } from '../../lib/tracking/state';
import type { Clip, ClipAttach, MotionComponent, MotionFilter, TrackerKind, TrackMask, TrackMotionModel, TrackMoveMode, TrackRegionShape, TrackSelectionMode } from '../../types';
import { Btn, cx, Field, Note, PanelHeader, Progress, Section, Segmented, Select, Slider, Toggle } from '../ui';

type Icon = ComponentType<{ size?: number; strokeWidth?: number; className?: string }>;

const TRACKERS: { id: TrackerKind; title: string; icon: Icon; desc: string }[] = [
  { id: 'object', title: 'Object', icon: Box, desc: 'Reliable with fast motion or lower-quality footage, prioritizing stability over precision.' },
  { id: 'flow', title: 'Flow', icon: Wind, desc: 'Tracks the selected region with optical flow and bounded recovery.' },
  { id: 'features', title: 'Features', icon: ScanSearch, desc: 'Precise on clear, steady videos. Less reliable with fast motion or blurry footage.' },
  { id: 'frame', title: 'Frame', icon: Frame, desc: 'Tracks dominant motion across the full frame, including affine and perspective changes.' },
  { id: 'face', title: 'Face', icon: ScanFace, desc: 'Detects facial landmarks and tracks a face, selected points, or a face area.' },
  { id: 'hands', title: 'Hands', icon: Hand, desc: 'Detects hand landmarks and tracks a hand, selected joints, or a hand area.' },
  { id: 'body', title: 'Body', icon: PersonStanding, desc: 'Detects body landmarks and tracks a person, selected joints, or a body area.' },
  { id: 'dual', title: 'Dual', icon: Waypoints, desc: 'Tracks two points at once, keeping scale and orientation.' },
];

const MODELS: { id: TrackMotionModel; title: string; desc: string }[] = [
  { id: 'translation', title: 'Translation', desc: 'Tracks horizontal and vertical position only. Size, rotation, shape, and perspective stay unchanged.' },
  { id: 'similarity', title: 'Similarity', desc: 'Tracks position, uniform scale, and rotation. Shape and perspective stay unchanged.' },
  { id: 'affine', title: 'Affine', desc: 'Tracks position, rotation, scale, stretch, and skew. Parallel lines stay parallel.' },
  { id: 'perspective', title: 'Perspective', desc: 'Tracks the complete planar transformation, including perspective changes. Best for flat surfaces turning toward or away from the camera.' },
];

const COMPONENT_LABEL: Record<MotionComponent, string> = { horizontal: 'Horizontal', vertical: 'Vertical', scale: 'Scale', rotation: 'Rotation', affine: 'Affine', perspective: 'Perspective' };

const FILTERS: { id: MotionFilter; title: string; desc: string }[] = [
  { id: 'off', title: 'Off', desc: 'The clip plays as shot. Tracked motion is still available to attached layers.' },
  { id: 'lock', title: 'Lock', desc: 'Tries to keep the tracked target as stable as possible. Higher Strength removes micro jitter and tightly locks motion to the chosen point.' },
  { id: 'smooth', title: 'Smooth', desc: 'Non-delayed smoothing. Keeps the original path but makes movement softer and cleaner, removing small shakes without visible lag.' },
  { id: 'follow', title: 'Follow', desc: 'Smooth follow with a gentle delay. The target is followed more softly instead of snapping. Higher Strength means more smoothing and slightly more lagged motion.' },
  { id: 'stabilize', title: 'Stabilize', desc: 'Uses only the tiny vibrations of the tracked object to stabilize the frame around it. It does not follow the object itself.' },
];

function Chip({ on, onClick, children, disabled, title }: { on: boolean; onClick: () => void; children: ReactNode; disabled?: boolean; title?: string }) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      aria-pressed={on}
      className={cx('flex h-7 items-center justify-center gap-1 rounded-md px-2 text-[11.5px] transition-colors disabled:opacity-40', on ? 'bg-accent/15 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]' : 'bg-ink text-muted shadow-[inset_0_0_0_1px_var(--color-line)] hover:text-fg')}
    >
      {children}
    </button>
  );
}

function Tool({ icon: I, label, onClick, disabled, danger, title }: { icon: Icon; label: string; onClick: () => void; disabled?: boolean; danger?: boolean; title?: string }) {
  return (
    <button
      type="button"
      title={title ?? label}
      disabled={disabled}
      onClick={onClick}
      className={cx('flex h-[46px] flex-col items-center justify-center gap-1 rounded-lg bg-raised text-[10.5px] shadow-[inset_0_0_0_1px_var(--color-line)] transition-colors hover:bg-hover disabled:opacity-35 disabled:hover:bg-raised', danger ? 'text-danger' : 'text-fg')}
    >
      <I size={14} />
      {label}
    </button>
  );
}

/** The clip's tracked ranges along its (trimmed) source timeline. Click to move the playhead. */
function SegmentStrip({ clip }: { clip: Clip }) {
  const time = useStore((s) => s.time);
  const live = useTrack((s) => s.live);
  useTrack((s) => s.tick);
  const a = clip.inPoint;
  const span = Math.max(1e-3, clip.duration * clip.speed);
  const pct = (src: number) => `${Math.min(100, Math.max(0, ((src - a) / span) * 100))}%`;
  const here = sourceTimeOf(clip, time);
  const inClip = time >= clip.start - 1e-3 && time <= clipEnd(clip) + 1e-3;
  const segs = clip.tracking?.segments ?? [];
  const current = segmentAt(clip.tracking, here);
  return (
    <div
      className="relative h-8 cursor-pointer overflow-hidden rounded-md bg-ink shadow-[inset_0_0_0_1px_var(--color-line)]"
      onPointerDown={(e) => {
        const seek = (ev: { clientX: number }) => {
          const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
          engine.pause();
          engine.seek(projectTimeOf(clip, a + Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)) * span));
        };
        seek(e);
        const el = e.currentTarget as HTMLElement;
        el.setPointerCapture(e.pointerId);
        const move = (ev: PointerEvent) => seek(ev);
        const up = () => {
          el.removeEventListener('pointermove', move);
          el.removeEventListener('pointerup', up);
          el.removeEventListener('pointercancel', up);
        };
        el.addEventListener('pointermove', move);
        el.addEventListener('pointerup', up);
        el.addEventListener('pointercancel', up);
      }}
      style={{ touchAction: 'none' }}
      role="slider"
      aria-label="Tracked ranges"
      aria-valuenow={Math.round(here * 100) / 100}
    >
      {segs.map((s, i) => {
        const left = ((segStart(s) - a) / span) * 100;
        const width = ((segEnd(s) - segStart(s)) / span) * 100;
        const on = current?.inside && current.index === i;
        // searching (held) stretches inside the segment
        const gaps: [number, number][] = [];
        let g0 = -1;
        for (let k = 0; k < s.t.length; k++) {
          if (s.s[k] && g0 < 0) g0 = k;
          if ((!s.s[k] || k === s.t.length - 1) && g0 >= 0) {
            gaps.push([s.t[g0], s.t[k]]);
            g0 = -1;
          }
        }
        const len = Math.max(1e-6, segEnd(s) - segStart(s));
        return (
          <div key={s.id} className={cx('absolute inset-y-[5px] rounded-[3px]', on ? 'bg-ok' : 'bg-ok/55')} style={{ left: `${left}%`, width: `max(3px, ${width}%)` }}>
            {gaps.map(([x0, x1], j) => (
              <span key={j} className="absolute inset-y-0 bg-warn" style={{ left: `${((x0 - segStart(s)) / len) * 100}%`, width: `max(2px, ${((x1 - x0) / len) * 100}%)` }} />
            ))}
          </div>
        );
      })}
      {live && live.clipId === clip.id && live.t.length > 1 && <div className="absolute inset-y-[5px] rounded-[3px] bg-accent" style={{ left: pct(live.t[0]), width: `max(3px, ${((live.t[live.t.length - 1] - live.t[0]) / span) * 100}%)` }} />}
      {inClip && <div className="pointer-events-none absolute inset-y-0 w-[2px] -translate-x-px bg-fg" style={{ left: pct(here) }} />}
      {!segs.length && !live && <span className="pointer-events-none absolute inset-0 flex items-center justify-center text-[10.5px] text-faint">No motion tracked yet</span>}
    </div>
  );
}

// ───────────────────────────── Lock (stabilizer) ─────────────────────────────

function LockSection({ clip }: { clip: Clip }) {
  const time = useStore((s) => s.time);
  const project = useStore((s) => s.project);
  const draft = useTrack((s) => s.lockDraft);
  const live = useTrack((s) => s.live);
  const phase = useTrack((s) => s.phase);
  const lock = lockOf(clip) ?? (draft?.clipId === clip.id ? draft : null);
  const enabled = !!lock?.enabled;
  const placed = !!lock?.placed;
  const centre = trackerCentreCanvas(project, clip, time);
  const x = placed ? lock!.x : centre?.x ?? 0;
  const y = placed ? lock!.y : centre?.y ?? 0;
  const hasMotion = !!clip.tracking?.segments.length || (live?.clipId === clip.id && phase !== 'idle');
  const shift = enabled && centre ? Math.hypot((centre.x - (lock?.x ?? 0)) * project.width, (centre.y - (lock?.y ?? 0)) * project.height) : null;
  const nudge = (dx: number, dy: number) => placeLock(clip.id, x + dx, y + dy);
  return (
    <Section
      title={
        <span className="inline-flex items-center gap-1.5">
          <Lock size={12} /> Lock
        </span>
      }
      right={enabled ? <span className="mr-3 rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-accent">Holding</span> : undefined}
    >
      <div className={cx('rounded-xl border p-2.5 transition-colors', enabled ? 'border-accent/45 bg-accent/[0.07]' : 'border-line bg-ink/40')}>
        <Toggle
          label="Lock"
          hint="Sets the lock point, then holds the tracker’s middle there. The video moves underneath, so the shot stays stable."
          value={enabled}
          onChange={(v) => toggleLock(clip.id, v)}
        />
        <div className="mt-2 flex items-center justify-between gap-2 border-t border-line/80 pt-2">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-[0.12em] text-faint">Lock point</div>
            <div className="font-mono text-[12px] tabular-nums text-fg">
              {x >= 0 ? '+' : ''}
              {(x * 100).toFixed(1)} , {y >= 0 ? '+' : ''}
              {(y * 100).toFixed(1)}
            </div>
          </div>
          <span className={cx('shrink-0 rounded-md px-1.5 py-0.5 text-[10.5px]', placed ? 'bg-raised text-fg' : 'text-faint')}>{placed ? 'Set' : 'On tracker'}</span>
        </div>
      </div>
      <p className="text-[11.5px] leading-snug text-muted">Drag the reticle on the preview — that one point is where the tracker middle will stay. Turning Lock on sets it there.</p>
      <div className="grid grid-cols-2 gap-1.5">
        <Btn size="sm" onClick={() => anchorToTracker(clip.id)} title="Put the lock point on the tracker middle of this frame">
          <Crosshair size={12} /> Tracker centre
        </Btn>
        <Btn size="sm" onClick={() => anchorToCentre(clip.id)} title="Hold the tracker middle at the centre of the frame">
          <LocateFixed size={12} /> Frame centre
        </Btn>
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="grid grid-cols-3 gap-1">
          <span />
          <NudgeBtn label="Lock point up" onClick={(e) => nudge(0, e.shiftKey ? -0.02 : -0.005)}>
            <ArrowUp size={12} />
          </NudgeBtn>
          <span />
          <NudgeBtn label="Lock point left" onClick={(e) => nudge(e.shiftKey ? -0.02 : -0.005, 0)}>
            <ArrowLeft size={12} />
          </NudgeBtn>
          <span className="flex h-7 w-7 items-center justify-center text-accent">
            <Lock size={12} />
          </span>
          <NudgeBtn label="Lock point right" onClick={(e) => nudge(e.shiftKey ? 0.02 : 0.005, 0)}>
            <ArrowRight size={12} />
          </NudgeBtn>
          <span />
          <NudgeBtn label="Lock point down" onClick={(e) => nudge(0, e.shiftKey ? 0.02 : 0.005)}>
            <ArrowDown size={12} />
          </NudgeBtn>
          <span />
        </div>
        <p className="min-w-0 flex-1 text-[10.5px] leading-snug text-faint">Nudge the lock point. Hold Shift for a larger step.</p>
      </div>
      {enabled && shift != null && (
        <p className="font-mono text-[10.5px] tabular-nums text-muted">
          {hasMotion ? `Holding · ${shift < 0.6 ? 'steady' : `${Math.round(shift)} px correction`}` : 'Lock point set — track a target to stabilize'}
        </p>
      )}
      {enabled && !hasMotion && <p className="text-[11px] leading-snug text-warn">Track a target. Lock will hold its middle point on the reticle as soon as motion exists.</p>}
      {clip.tracking && (
        <Toggle
          label="Auto-fit frame"
          hint="Zoom in just enough that the stabilized picture never shows its edges."
          value={clip.tracking.autoFit}
          disabled={!enabled}
          onChange={(v) => getState().updateClip(clip.id, { tracking: { ...clip.tracking!, autoFit: v } })}
        />
      )}
    </Section>
  );
}

// ───────────────────────────── Track ─────────────────────────────

function TrackTab({ clip }: { clip: Clip }) {
  const raw = useTrack((s) => s.settings);
  const set = useTrack((s) => s.setSettings);
  const phase = useTrack((s) => s.phase);
  const status = useTrack((s) => s.status);
  const progress = useTrack((s) => s.progress);
  const frames = useTrack((s) => s.frames);
  const speed = useTrack((s) => s.speed);
  const live = useTrack((s) => s.live);
  const detected = useTrack((s) => s.detected);
  const detecting = useTrack((s) => s.detecting);
  const showPoints = useTrack((s) => s.showPoints);
  const lockDraft = useTrack((s) => s.lockDraft);
  const time = useStore((s) => s.time);
  const playing = useStore((s) => s.playing);
  const project = useStore((s) => s.project);
  const s = effectiveSettings(raw);
  const busy = phase !== 'idle';
  const info = TRACKERS.find((t) => t.id === raw.tracker)!;
  const semantic = isSemantic(s.tracker);
  const inClip = time >= clip.start - 1e-3 && time < clipEnd(clip) - 1e-3;
  const here = sourceTimeOf(clip, time);
  const hit = segmentAt(clip.tracking, here);
  const onSeg = !!hit?.inside;
  const stats = trackingStats(clip.tracking);
  const aspect = assetAspect(project, clip);
  const lockHolding = !!(clip.tracking?.lock?.enabled || (lockDraft?.clipId === clip.id && lockDraft.enabled));

  // face / hands / body: show what the model sees in the current frame so a target can be picked
  useEffect(() => {
    if (!semantic || busy || playing || !inClip) return;
    const id = setTimeout(() => void detectPreview(clip.id), 220);
    return () => clearTimeout(id);
  }, [semantic, s.tracker, busy, playing, inClip, clip.id, time]);

  const found = semantic && detected && detected.clipId === clip.id && detected.kind === s.tracker ? detected.entities.length : null;
  const edit = (fn: (tr: NonNullable<Clip['tracking']>) => ReturnType<typeof splitAt>, fail: string) => {
    if (!clip.tracking) return;
    const next = fn(clip.tracking);
    if (!next) getState().toast(fail);
    else commitTracking(clip.id, next);
  };

  return (
    <>
      <Section title="Tracker type">
        <div className="grid grid-cols-4 gap-1.5">
          {TRACKERS.map((t) => (
            <button
              key={t.id}
              type="button"
              disabled={busy}
              onClick={() => set({ tracker: t.id })}
              aria-pressed={raw.tracker === t.id}
              className={cx('flex h-[52px] flex-col items-center justify-center gap-1 rounded-lg text-[10.5px] transition-colors disabled:opacity-50', raw.tracker === t.id ? 'bg-accent/15 text-accent shadow-[inset_0_0_0_1.5px_var(--color-accent)]' : 'bg-raised text-muted shadow-[inset_0_0_0_1px_var(--color-line)] hover:bg-hover hover:text-fg')}
            >
              <t.icon size={17} strokeWidth={1.8} />
              {t.title}
            </button>
          ))}
        </div>
        <p className="text-[11.5px] leading-snug text-muted">{info.desc}</p>
      </Section>

      <Section title="Tracker settings">
        {s.tracker === 'flow' && (
          <Field label="Tracking mode">
            <Segmented value={raw.flowMode} onChange={(v) => set({ flowMode: v })} options={[{ value: 'fast', label: 'Fast' }, { value: 'advanced', label: 'Advanced' }]} className="w-40" />
          </Field>
        )}
        {(s.tracker === 'flow' || s.tracker === 'frame') && (
          <>
            <Field label="Motion model">
              <Select<TrackMotionModel> value={s.model} onChange={(v) => set({ model: v })} className="w-36" label="Motion model">
                {MODELS.filter((m) => s.tracker !== 'frame' || m.id !== 'translation').map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.title}
                  </option>
                ))}
              </Select>
            </Field>
            <p className="text-[11px] leading-snug text-faint">{MODELS.find((m) => m.id === s.model)!.desc}</p>
          </>
        )}
        {(s.tracker === 'flow' || s.tracker === 'features' || s.tracker === 'object') && (
          <>
            <Field label="Selector shape">
              <Segmented<TrackRegionShape>
                value={s.shape}
                onChange={(v) => set({ shape: v })}
                options={[
                  { value: 'circle', label: <Circle size={13} />, title: 'Circle: follows size, without rotation or perspective changes' },
                  { value: 'rectangle', label: <RectangleHorizontal size={14} />, title: 'Rectangle: follows size and rotation independently' },
                  ...(s.tracker === 'flow' ? [{ value: 'quad' as const, label: <Pentagon size={13} />, title: 'Quad: also supports perspective changes with Corner Pin' }] : []),
                ]}
                className="w-32"
              />
            </Field>
            <div>
              <div className="mb-1.5 text-[12px] text-muted">ROI transforms</div>
              <div className="grid grid-cols-3 gap-1.5">
                <Chip on={s.followSize} disabled={s.cornerPin} onClick={() => set({ followSize: !raw.followSize })} title="Resizes the region as the tracked surface changes size">
                  Follow size
                </Chip>
                <Chip on={s.followRotation} disabled={s.cornerPin || s.shape === 'circle' || s.tracker === 'object'} onClick={() => set({ followRotation: !raw.followRotation })} title="Rotates Rectangle or Quad with the tracked surface">
                  Follow rotation
                </Chip>
                <Chip on={s.cornerPin} disabled={s.tracker !== 'flow' || s.shape !== 'quad'} onClick={() => set({ cornerPin: !raw.cornerPin, ...(raw.cornerPin ? {} : raw.model === 'translation' || raw.model === 'similarity' ? { model: 'perspective' as const } : {}) })} title="Lets all four Quad corners follow perspective">
                  Corner Pin
                </Chip>
              </div>
              {s.cornerPin && <p className="mt-1.5 text-[11px] leading-snug text-faint">Corner Pin is enabled. Size and rotation are applied automatically.</p>}
            </div>
          </>
        )}
        {semantic && (
          <>
            <Field label="Target selection">
              <Segmented<TrackSelectionMode> value={s.selection} onChange={(v) => set({ selection: v })} options={[{ value: 'entity', label: 'Entity' }, { value: 'points', label: 'Point' }, { value: 'area', label: 'Area' }]} className="w-44" />
            </Field>
            {s.selection === 'points' && (
              <Field label="Point">
                <Select value={LANDMARKS[s.tracker as 'face'].some((l) => l.id === s.landmark) ? s.landmark : LANDMARKS[s.tracker as 'face'][0].id} onChange={(v) => set({ landmark: v })} className="w-44" label="Tracked point">
                  {LANDMARKS[s.tracker as 'face'].map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.label}
                    </option>
                  ))}
                </Select>
              </Field>
            )}
            <p className="text-[11px] leading-snug text-faint">
              {!inClip ? 'Move the playhead onto the clip to detect targets.' : detecting && found === null ? 'Wait for detection to finish before starting tracking.' : found === 0 ? 'Nothing detected in this frame. Choose a frame where the subject is visible.' : found ? `${found} detected — tap one on the preview to choose it${s.selection === 'points' ? ', then tap a point' : s.selection === 'area' ? ', then place the area' : ''}.` : 'Detecting…'}
            </p>
          </>
        )}
        {s.tracker === 'dual' && <p className="text-[11px] leading-snug text-faint">Drag anchors A and B onto two distinct details of the same object. The distance between them drives scale; the angle drives rotation.</p>}
        {(s.tracker === 'flow' || s.tracker === 'features' || s.tracker === 'object' || s.tracker === 'frame') && (
          <>
            <Field label="Tracking mask">
              <Segmented<TrackMask>
                value={s.mask}
                onChange={(v) => set({ mask: v })}
                options={[{ value: 'none', label: 'None' }, ...(s.tracker !== 'frame' ? [{ value: 'subject' as const, label: 'Subject' }] : []), ...(s.tracker !== 'object' ? [{ value: 'background' as const, label: 'Background' }] : [])]}
                className="w-48"
              />
            </Field>
            {s.mask !== 'none' && <p className="text-[11px] leading-snug text-faint">{s.mask === 'subject' ? 'Uses the detected person inside the region.' : 'Excludes detected people inside the region.'} Masks guide tracking, do not move the region, and can be inaccurate.{s.tracker === 'object' ? ' Object tracking uses the mask at initialization.' : ''}</p>}
          </>
        )}
        <Field label="Tracking resolution">
          <Segmented value={raw.resolution} onChange={(v) => set({ resolution: v })} options={[{ value: 360, label: 'Fast' }, { value: 540, label: 'Balanced' }, { value: 720, label: 'High' }]} className="w-48" />
        </Field>
      </Section>

      <LockSection clip={clip} />

      <Section
        title="Track"
        right={
          <button type="button" title={showPoints ? 'Hide tracking points' : 'Show tracking points'} onClick={() => useTrack.getState().patch({ showPoints: !showPoints })} className="text-faint hover:text-fg">
            {showPoints ? <Eye size={14} /> : <EyeOff size={14} />}
          </button>
        }
      >
        <SegmentStrip clip={clip} />
        {busy ? (
          <div className="space-y-2 rounded-lg border border-line bg-ink/60 p-2.5">
            <div className="flex items-center justify-between text-[12px]">
              <span className="flex items-center gap-1.5 text-fg">
                <span className={cx('h-2 w-2 rounded-full oc-pulse', live?.state ? 'bg-warn' : 'bg-ok')} />
                {status || 'Working…'}
              </span>
              <span className="font-mono text-[11px] tabular-nums text-muted">{phase === 'analyzing' ? `${frames} frames · ${speed.toFixed(0)} fps` : ''}</span>
            </div>
            <Progress value={phase === 'analyzing' ? progress : null} />
            <Btn variant="soft" className="w-full" onClick={stopTracking} disabled={phase === 'stopping' || phase === 'committing'}>
              <Square size={12} fill="currentColor" /> Stop
            </Btn>
            {lockHolding && <p className="text-[11px] leading-snug text-accent">Lock is on — the tracker centre is held on the reticle while this runs.</p>}
          </div>
        ) : (
          <>
            <Btn variant="primary" size="lg" className="w-full" disabled={!inClip || (semantic && found === 0)} onClick={() => void startTracking(clip.id)}>
              <Play size={14} fill="currentColor" /> Track from playhead
            </Btn>
            {!inClip && <p className="text-[11px] leading-snug text-warn">Move the playhead onto this clip to choose where tracking starts.</p>}
          </>
        )}
        <div className="grid grid-cols-5 gap-1.5">
          <Tool icon={RefreshCw} label="Retrack" disabled={busy || !onSeg} onClick={() => void retrack(clip.id)} title="Track the segment at the playhead again with the current settings" />
          <Tool icon={Scissors} label="Split" disabled={busy || !onSeg} onClick={() => edit((tr) => splitAt(tr, here), 'Move the playhead inside a tracking segment to split it')} title="Divide the tracking segment at the playhead" />
          <Tool icon={Link2} label="Connect" disabled={busy || (clip.tracking?.segments.length ?? 0) < 2} onClick={() => edit((tr) => connectAt(tr, here, aspect, project.fps), 'There is no following segment to connect to')} title="Join this segment with the next one into a single smooth motion" />
          <Tool icon={Trash2} label="Delete" disabled={busy || !onSeg} onClick={() => edit((tr) => deleteAt(tr, here), 'There is no motion at this position')} title="Remove the tracking segment at the playhead" />
          <Tool icon={Eraser} label="Clear" danger disabled={busy || !clip.tracking} onClick={() => { stashLock(clip); getState().updateClip(clip.id, { tracking: undefined }); }} title="Remove all tracking from this clip" />
        </div>
        {stats && (
          <div className="grid grid-cols-3 gap-1.5 text-center">
            {[
              ['Segments', String(stats.segments)],
              ['Frames', String(stats.samples)],
              ['Confidence', `${Math.round(stats.confidence * 100)}%`],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-line px-1 py-1.5">
                <div className="font-mono text-[13px] tabular-nums text-fg">{v}</div>
                <div className="text-[9.5px] uppercase tracking-wider text-faint">{k}</div>
              </div>
            ))}
          </div>
        )}
      </Section>
    </>
  );
}

// ───────────────────────────── Adjust Motion ─────────────────────────────

function AdjustTab({ clip }: { clip: Clip }) {
  const time = useStore((s) => s.time);
  const applyAll = useTrack((s) => s.applyAll);
  const tr = clip.tracking;
  if (!tr?.segments.length) return <Empty>Track this clip first. Adjust Motion then decides which motions apply and tunes the look from rigid to smooth while removing jitter. To hold the tracker middle on a point you choose, use Lock in the Track tab.</Empty>;
  const here = sourceTimeOf(clip, time);
  const hit = segmentAt(tr, here)!;
  const adj = hit.seg.adjust;
  const change = (patch: Parameters<typeof setAdjust>[2], key: string) => commitTracking(clip.id, setAdjust(tr, here, patch, applyAll), 'trk-' + key + clip.id);
  const f = FILTERS.find((x) => x.id === adj.filter)!;
  return (
    <>
      <Section title="Motion">
        <p className="text-[11.5px] leading-snug text-muted">Choose which tracked motion types to keep. Turn off a motion type to freeze it.</p>
        <div className="grid grid-cols-3 gap-1.5">
          {MOTION_COMPONENTS.map((c) => (
            <Chip key={c} on={adj.components[c]} onClick={() => change({ components: { ...adj.components, [c]: !adj.components[c] } }, 'comp')}>
              {COMPONENT_LABEL[c]}
            </Chip>
          ))}
        </div>
      </Section>
      <Section title="Filter">
        <div className="grid grid-cols-5 gap-1">
          {FILTERS.map((x) => (
            <button
              key={x.id}
              type="button"
              onClick={() => change({ filter: x.id }, 'filter')}
              aria-pressed={adj.filter === x.id}
              className={cx('flex h-9 items-center justify-center rounded-md text-[11px] transition-colors', adj.filter === x.id ? 'bg-accent/15 text-accent shadow-[inset_0_0_0_1.5px_var(--color-accent)]' : 'bg-ink text-muted shadow-[inset_0_0_0_1px_var(--color-line)] hover:text-fg')}
            >
              {x.title}
            </button>
          ))}
        </div>
        <p className="text-[11.5px] leading-snug text-muted">{f.desc}</p>
        {adj.filter === 'lock' && <p className="text-[11px] leading-snug text-faint">This filter pins motion to where tracking started. For a point you place yourself, use Lock in the Track tab — the tracker middle stays on that reticle.</p>}
        <Slider label="Strength" value={adj.strength} min={0} max={100} def={60} disabled={adj.filter === 'off'} format={(v) => `${Math.round(v)}%`} onChange={(v) => change({ strength: v }, 'strength')} />
        <Toggle label="Auto-fit frame" hint="Zoom in just enough that the moving picture never shows its edges." value={tr.autoFit} disabled={adj.filter === 'off'} onChange={(v) => commitTracking(clip.id, { ...tr, autoFit: v })} />
        <Toggle label="Apply changes to all segments" hint={tr.segments.length > 1 ? `Off: only segment ${hit.index + 1} of ${tr.segments.length} (at the playhead) changes.` : undefined} value={applyAll} onChange={(v) => useTrack.getState().patch({ applyAll: v })} />
      </Section>
      <div className="p-3">
        <Note>Combine motion types with different filters and strength levels to get the motion look you want. Layers attached to this clip follow the adjusted motion.</Note>
      </div>
    </>
  );
}

// ───────────────────────────── Edit Motion ─────────────────────────────

function EditTab({ clip }: { clip: Clip }) {
  const time = useStore((s) => s.time);
  const project = useStore((s) => s.project);
  const mode = useTrack((s) => s.moveMode);
  const snapshot = useTrack((s) => s.snapshot);
  const [smooth, setSmooth] = useState(40);
  const tr = clip.tracking;

  // remember the track as it was when Edit Motion was opened, for Reset
  useEffect(() => {
    useTrack.getState().patch({ snapshot: { clipId: clip.id, tracking: getState().project.clips.find((c) => c.id === clip.id)?.tracking } });
    return () => useTrack.getState().patch({ snapshot: null });
  }, [clip.id]);

  if (!tr?.segments.length) return <Empty>Track this clip first. Edit Motion lets you recover from occlusions, repair drift, split and reconnect segments for one smooth motion.</Empty>;
  const aspect = assetAspect(project, clip);
  const here = sourceTimeOf(clip, time);
  const m = motionAt(tr, here, aspect);
  const on = !!m?.inside;
  const canReset = !!snapshot && snapshot.clipId === clip.id && snapshot.tracking !== tr;
  const asset = project.assets.find((a) => a.id === clip.assetId);
  const nudge = (dx: number, dy: number, big: boolean) => {
    if (!m || !on) return;
    const k = big ? 10 : 1;
    commitTracking(clip.id, moveSample(tr, m.seg.id, m.sample, (dx * k) / (asset?.width || 1920), (dy * k) / (asset?.height || 1080), mode), 'trk-nudge' + clip.id);
  };
  const step = (dir: number) => {
    if (!m) return;
    const i = Math.min(m.seg.t.length - 1, Math.max(0, m.sample + dir));
    engine.pause();
    engine.seek(projectTimeOf(clip, m.seg.t[i]));
  };
  const apply = (next: ReturnType<typeof splitAt>, fail: string) => (next ? commitTracking(clip.id, next) : getState().toast(fail));
  return (
    <>
      <Section title="Move a tracked point">
        <SegmentStrip clip={clip} />
        <p className="text-[11.5px] leading-snug text-muted">{on ? 'Drag the marker on the preview, or nudge it here, then choose how that correction changes the motion.' : 'There is no motion at this position.'}</p>
        <Segmented<TrackMoveMode>
          value={mode}
          onChange={(v) => useTrack.getState().patch({ moveMode: v })}
          options={[
            { value: 'current', label: 'Current', title: 'Current point — Move only the tracked point at the playhead.' },
            { value: 'forward', label: 'From here', title: 'From here — Move the point at the playhead and every later point in the segment.' },
            { value: 'both', label: 'Drift', title: 'Drift correction — Blend the correction into earlier points, then apply the full correction from the playhead onward.' },
          ]}
        />
        <p className="text-[11px] leading-snug text-faint">
          {mode === 'current' ? 'Current point — Move only the tracked point at the playhead.' : mode === 'forward' ? 'From here — Move the point at the playhead and every later point in the segment.' : 'Drift correction — Blend the correction into earlier points, then apply the full correction from the playhead onward.'}
        </p>
        <div className="flex items-center justify-between gap-3">
          <div className="grid grid-cols-3 gap-1">
            <span />
            <NudgeBtn label="Nudge up" disabled={!on} onClick={(e) => nudge(0, -1, e.shiftKey)}>
              <ArrowUp size={13} />
            </NudgeBtn>
            <span />
            <NudgeBtn label="Nudge left" disabled={!on} onClick={(e) => nudge(-1, 0, e.shiftKey)}>
              <ArrowLeft size={13} />
            </NudgeBtn>
            <span className="flex h-7 w-7 items-center justify-center text-faint">
              <Crosshair size={13} />
            </span>
            <NudgeBtn label="Nudge right" disabled={!on} onClick={(e) => nudge(1, 0, e.shiftKey)}>
              <ArrowRight size={13} />
            </NudgeBtn>
            <span />
            <NudgeBtn label="Nudge down" disabled={!on} onClick={(e) => nudge(0, 1, e.shiftKey)}>
              <ArrowDown size={13} />
            </NudgeBtn>
            <span />
          </div>
          <div className="min-w-0 flex-1 space-y-1.5 text-[11px] text-muted">
            <div className="flex justify-between">
              <span>Sample</span>
              <span className="font-mono tabular-nums text-fg">{m ? `${m.sample + 1} / ${m.seg.t.length}` : '—'}</span>
            </div>
            <div className="flex justify-between">
              <span>Position</span>
              <span className="font-mono tabular-nums text-fg">{m ? `${(m.cx * 100).toFixed(1)}, ${(m.cy * 100).toFixed(1)}` : '—'}</span>
            </div>
            <div className="flex justify-between">
              <span>Confidence</span>
              <span className={cx('font-mono tabular-nums', m?.searching ? 'text-warn' : 'text-fg')}>{m ? (m.searching ? 'searching' : `${Math.round(m.conf * 100)}%`) : '—'}</span>
            </div>
            <div className="flex gap-1">
              <Btn size="sm" className="flex-1" disabled={!m} onClick={() => step(-1)}>
                Prev
              </Btn>
              <Btn size="sm" className="flex-1" disabled={!m} onClick={() => step(1)}>
                Next
              </Btn>
            </div>
          </div>
        </div>
        <p className="text-[10.5px] leading-snug text-faint">Hold Shift while nudging to move ten pixels at a time. Changes are saved when you release.</p>
      </Section>
      <Section title="Segment">
        <div className="grid grid-cols-4 gap-1.5">
          <Tool icon={Scissors} label="Split" disabled={!on} onClick={() => apply(splitAt(tr, here), 'Move the playhead inside a tracking segment to split it')} title="Split — Divide the tracking segment at the playhead." />
          <Tool icon={Link2} label="Connect" disabled={tr.segments.length < 2} onClick={() => apply(connectAt(tr, here, aspect, project.fps), 'There is no following segment to connect to')} title="Connect — Join this segment with the next one." />
          <Tool icon={Trash2} label="Delete" disabled={!on} onClick={() => apply(deleteAt(tr, here), 'There is no motion at this position')} title="Delete — Remove the tracking segment at the playhead." />
          <Tool icon={RotateCcw} label="Reset" disabled={!canReset} onClick={() => snapshot && getState().updateClip(clip.id, { tracking: snapshot.tracking })} title="Reset — Restore the tracking snapshot captured when Edit Motion was opened." />
        </div>
      </Section>
      <Section title="Smooth the track">
        <Slider label="Amount" value={smooth} min={5} max={100} def={40} format={(v) => `${Math.round(v)}%`} onChange={setSmooth} />
        <div className="grid grid-cols-2 gap-1.5">
          <Btn disabled={!on} onClick={() => commitTracking(clip.id, smoothAt(tr, here, smooth / 100))}>
            <Sparkles size={13} /> This segment
          </Btn>
          <Btn onClick={() => commitTracking(clip.id, smoothAt(tr, null, smooth / 100))}>
            <Sparkles size={13} /> All segments
          </Btn>
        </div>
        <p className="text-[11px] leading-snug text-faint">Rewrites the tracked path itself to take out jitter. For a look you can change later, use a filter in Adjust instead.</p>
      </Section>
    </>
  );
}

function NudgeBtn({ children, label, disabled, onClick }: { children: ReactNode; label: string; disabled?: boolean; onClick: (e: React.MouseEvent) => void }) {
  return (
    <button type="button" aria-label={label} title={label} disabled={disabled} onClick={onClick} className="flex h-7 w-7 items-center justify-center rounded-md bg-raised text-fg shadow-[inset_0_0_0_1px_var(--color-line)] hover:bg-hover disabled:opacity-35">
      {children}
    </button>
  );
}

// ───────────────────────────── Attach ─────────────────────────────

function ClipGlyph({ clip }: { clip: Clip }) {
  const cls = 'shrink-0 text-muted';
  return clip.type === 'text' ? <Type size={14} className={cls} /> : clip.type === 'image' ? <ImageIcon size={14} className={cls} /> : <Film size={14} className={cls} />;
}
const clipLabel = (c: Clip) => (c.type === 'text' ? c.text.replace(/\n/g, ' ').slice(0, 40) || 'Text' : c.name || c.type);

function AttachControls({ clip }: { clip: Clip }) {
  const a = clip.attach!;
  const patch = (p: Partial<ClipAttach>) => getState().updateClip(clip.id, { attach: { ...a, ...p } });
  const off = Math.hypot(clip.x - a.ax, clip.y - a.ay) > 1e-4 || !!clip.keyframes.x?.length || !!clip.keyframes.y?.length;
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-1">
        <Chip on={a.position && !a.cornerPin} disabled={a.cornerPin} onClick={() => patch({ position: !a.position })}>
          Position
        </Chip>
        <Chip on={a.scale && !a.cornerPin} disabled={a.cornerPin} onClick={() => patch({ scale: !a.scale })}>
          Size
        </Chip>
        <Chip on={a.rotation && !a.cornerPin} disabled={a.cornerPin} onClick={() => patch({ rotation: !a.rotation })}>
          Rotation
        </Chip>
        <Chip on={a.cornerPin} onClick={() => patch({ cornerPin: !a.cornerPin })} title="Map the layer's four corners onto the tracked quad">
          Corner pin
        </Chip>
      </div>
      {a.cornerPin ? (
        <p className="text-[11px] leading-snug text-faint">The layer fills the tracked quad. Track with Flow · Quad · Corner Pin (Perspective) for a true screen replacement.</p>
      ) : (
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-[10.5px] tabular-nums text-muted">
            Offset dx {((clip.x - a.ax) * 100).toFixed(1)} · dy {((clip.y - a.ay) * 100).toFixed(1)}
          </span>
          <button type="button" disabled={!off} onClick={() => {
              const keyframes = { ...clip.keyframes };
              delete keyframes.x;
              delete keyframes.y;
              getState().updateClip(clip.id, { x: a.ax, y: a.ay, keyframes });
            }} className="flex items-center gap-1 text-[11.5px] text-accent hover:underline disabled:text-faint disabled:no-underline">
            <LocateFixed size={12} /> Centre on target
          </button>
        </div>
      )}
    </div>
  );
}

function AttachRow({ clip, target }: { clip: Clip; target: Clip }) {
  const mine = clip.attach && (clip.attach.targetId === target.id || clip.attach.trackingId === target.tracking?.id);
  const elsewhere = !!clip.attach && !mine;
  return (
    <div className={cx('rounded-lg border p-2', mine ? 'border-accent/50 bg-accent/[0.06]' : 'border-line')}>
      <div className="flex items-center gap-2">
        <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => getState().select([clip.id])} title="Select this layer">
          <ClipGlyph clip={clip} />
          <span className="truncate text-[12px] text-fg">{clipLabel(clip)}</span>
        </button>
        {mine ? (
          <Btn size="sm" onClick={() => detachClip(clip.id)}>
            <Unlink2 size={12} /> Detach
          </Btn>
        ) : (
          <Btn size="sm" variant="primary" onClick={() => attachClip(clip.id, target.id)} title={elsewhere ? 'Attached to another clip — attach here instead' : undefined}>
            <Pin size={12} /> Attach
          </Btn>
        )}
      </div>
      {mine && (
        <div className="mt-2">
          <AttachControls clip={clip} />
        </div>
      )}
    </div>
  );
}

function AttachTab({ clip }: { clip: Clip }) {
  const project = useStore((s) => s.project);
  const time = useStore((s) => s.time);
  const layers = useMemo(
    () =>
      project.clips
        .filter((c) => c.id !== clip.id && c.type !== 'audio' && !c.caption && c.start < clipEnd(clip) && clipEnd(c) > clip.start)
        .sort((a, b) => Number(!!b.attach) - Number(!!a.attach) || a.start - b.start),
    [project.clips, clip],
  );
  if (!clip.tracking?.segments.length) return <Empty>Track this clip first. Text, stickers, images and other videos can then be attached so they follow the tracked target.</Empty>;
  const addText = () => {
    const st = getState();
    const start = Math.min(Math.max(st.time, clip.start), clipEnd(clip) - 0.2);
    const id = st.addTextClip({ start, duration: Math.max(0.5, clipEnd(clip) - start), text: 'Tracked label' });
    if (attachClip(id, clip.id)) {
      const made = getState().project.clips.find((c) => c.id === id);
      if (made?.attach) getState().updateClip(id, { x: made.attach.ax, y: made.attach.ay - 0.08 });
    }
    getState().select([clip.id]);
  };
  return (
    <>
      <Section title="Attach layers">
        <p className="text-[11.5px] leading-snug text-muted">Attached layers keep their offset from the target and follow its position, size and rotation. Drag them on the preview to set the offset.</p>
        <Btn className="w-full" onClick={addText}>
          <Plus size={13} /> Add a text label on the target
        </Btn>
        {layers.length ? (
          <div className="space-y-1.5">
            {layers.map((c) => (
              <AttachRow key={c.id} clip={c} target={clip} />
            ))}
          </div>
        ) : (
          <Note>No other layers overlap this clip yet. Add text, an image or an overlay video above it on the timeline, then attach it here.</Note>
        )}
        {time < clip.start || time > clipEnd(clip) ? <p className="text-[11px] leading-snug text-warn">Move the playhead onto the clip so new attachments are placed where you see them.</p> : null}
      </Section>
    </>
  );
}

/** Shown when the selected clip is a layer (text / image / overlay) rather than a tracked video. */
function LayerAttach({ clip }: { clip: Clip }) {
  const project = useStore((s) => s.project);
  const targets = project.clips.filter((c) => c.id !== clip.id && c.type === 'video' && c.tracking?.segments.length && c.start < clipEnd(clip) && clipEnd(c) > clip.start);
  const current = clip.attach ? project.clips.find((c) => c.id === clip.attach!.targetId) ?? project.clips.find((c) => c.tracking?.id === clip.attach!.trackingId) : undefined;
  return (
    <Section title="Follow tracked motion">
      {clip.attach ? (
        <>
          <div className="flex items-center justify-between gap-2 rounded-lg border border-accent/50 bg-accent/[0.06] p-2">
            <button type="button" className="flex min-w-0 items-center gap-2 text-left" onClick={() => current && getState().select([current.id])} title="Select the tracked clip">
              <Pin size={13} className="shrink-0 text-accent" />
              <span className="truncate text-[12px] text-fg">{current ? clipLabel(current) : 'Tracked clip removed'}</span>
            </button>
            <Btn size="sm" onClick={() => detachClip(clip.id)}>
              <Unlink2 size={12} /> Detach
            </Btn>
          </div>
          <AttachControls clip={clip} />
        </>
      ) : targets.length ? (
        <>
          <p className="text-[11.5px] leading-snug text-muted">Attach this layer to a tracked clip. It stays where it is now and follows the target from there.</p>
          {targets.map((t) => (
            <div key={t.id} className="flex items-center gap-2 rounded-lg border border-line p-2">
              <Film size={14} className="shrink-0 text-muted" />
              <span className="min-w-0 flex-1 truncate text-[12px] text-fg">{clipLabel(t)}</span>
              <Btn size="sm" variant="primary" onClick={() => attachClip(clip.id, t.id)}>
                <Pin size={12} /> Attach
              </Btn>
            </div>
          ))}
        </>
      ) : (
        <Note>Select a video clip and track something in it first. This layer can then be attached so it follows the tracked target.</Note>
      )}
    </Section>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="p-3">
      <Note>{children}</Note>
    </div>
  );
}

const TABS: { value: TrackTab; label: ReactNode; title: string }[] = [
  { value: 'track', label: <><Crosshair size={12} /> Track</>, title: 'Tracking' },
  { value: 'adjust', label: <><Activity size={12} /> Adjust</>, title: 'Adjust Motion' },
  { value: 'edit', label: <><PenLine size={12} /> Edit</>, title: 'Edit Motion' },
  { value: 'attach', label: <><Pin size={12} /> Attach</>, title: 'Attach layers' },
];

/** Motion tracking: pick a tracker and a target, analyse, then adjust, repair and use the motion. */
export function TrackingPanel() {
  const project = useStore((s) => s.project);
  const selection = useStore((s) => s.selection);
  const tab = useTrack((s) => s.tab);
  const phase = useTrack((s) => s.phase);
  const clip = useMemo(() => (selection.length === 1 ? project.clips.find((c) => c.id === selection[0]) : undefined), [project.clips, selection]);

  useEffect(() => {
    useTrack.getState().patch({ active: true });
    return () => useTrack.getState().patch({ active: false });
  }, []);

  const videos = project.clips.filter((c) => c.type === 'video');
  return (
    <div className="flex h-full flex-col">
      <PanelHeader
        title="Motion tracking"
        sub={clip?.type === 'video' ? <span className="block max-w-[200px] truncate">{clip.name}</span> : 'On-device · nothing is uploaded'}
        right={clip?.type === 'video' && clip.tracking?.lock?.enabled ? <span className="mt-0.5 rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-accent">Locked</span> : undefined}
      />
      {!clip ? (
        <div className="space-y-2 p-3">
          <Note>Select a video clip on the timeline to track something in it — an object, a surface, a face, hands, a body or the camera itself.</Note>
          {videos.slice(0, 4).map((v) => (
            <button key={v.id} type="button" onClick={() => getState().select([v.id])} className="flex w-full items-center gap-2 rounded-lg border border-line p-2 text-left hover:border-line-strong hover:bg-raised">
              <Film size={14} className="shrink-0 text-muted" />
              <span className="min-w-0 flex-1 truncate text-[12px] text-fg">{v.name}</span>
              {v.tracking?.segments.length ? <span className="rounded bg-ok/20 px-1.5 text-[10px] text-ok">tracked</span> : null}
            </button>
          ))}
        </div>
      ) : clip.type !== 'video' ? (
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          {clip.type === 'audio' ? <Empty>Audio clips have no picture to track. Select a video clip.</Empty> : <LayerAttach clip={clip} />}
        </div>
      ) : (
        <>
          <div className="px-3 pb-2">
            <Segmented<TrackTab> value={tab} onChange={(v) => phase === 'idle' && useTrack.getState().patch({ tab: v })} options={TABS} />
          </div>
          <div className="scroll-thin min-h-0 flex-1 overflow-y-auto border-t border-line">
            {tab === 'track' && <TrackTab clip={clip} />}
            {tab === 'adjust' && <AdjustTab clip={clip} />}
            {tab === 'edit' && <EditTab clip={clip} />}
            {tab === 'attach' && <AttachTab clip={clip} />}
          </div>
        </>
      )}
    </div>
  );
}

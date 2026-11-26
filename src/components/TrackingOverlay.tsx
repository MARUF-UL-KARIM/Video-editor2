import { useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import { engine } from '../lib/engine';
import { getState, useStore } from '../lib/store';
import { placeLock } from '../lib/tracking/lock';
import { quadCx, quadCy, rectQuad } from '../lib/tracking/math';
import { motionAt, motionPath, moveSample } from '../lib/tracking/motion';
import { POINT_INLIER, POINT_LANDMARK, POINT_OUTLIER, POINT_RESCUED } from '../lib/tracking/pointTracker';
import { assetAspect, canvasToSource, sourceToCanvas, trackerCentreCanvas, type Geom } from '../lib/tracking/resolve';
import { LANDMARKS, landmarkPoint, nearestEntity } from '../lib/tracking/semantic';
import { regionFor, sourceTimeOf } from '../lib/tracking/session';
import { dualPoints, dualQuad, effectiveSettings, isSemantic, rectOf, useTrack } from '../lib/tracking/state';
import type { ClipTracking } from '../types';

const C = {
  select: '#ff6a3d',
  tracked: '#7fd08a',
  edge: '#38e8d0',
  searching: '#f2c14e',
  lost: '#ff5d5d',
  rescued: '#6fd3e8',
  ink: '#0b0c0b',
  fg: '#eceae2',
};

/** Start a drag that keeps working when the pointer leaves the handle. */
function drag(e: RPointerEvent, onMove: (ev: PointerEvent) => void, onEnd?: () => void) {
  e.stopPropagation();
  e.preventDefault();
  const move = (ev: PointerEvent) => onMove(ev);
  const up = () => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    window.removeEventListener('pointercancel', up);
    onEnd?.();
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
  window.addEventListener('pointercancel', up);
}

const hot = { pointerEvents: 'auto' as const, touchAction: 'none' as const };

/**
 * Everything the tracker draws over the preview: the selector (circle / rectangle / quad, dual anchors, detected
 * faces · hands · bodies), live analysis feedback (region, feature points by status, path) and the committed track.
 */
export function TrackingOverlay({ k }: { k: number }) {
  const active = useTrack((s) => s.active);
  const tab = useTrack((s) => s.tab);
  const phase = useTrack((s) => s.phase);
  const settingsRaw = useTrack((s) => s.settings);
  const live = useTrack((s) => s.live);
  const detected = useTrack((s) => s.detected);
  const edgePreview = useTrack((s) => s.edgePreview);
  const showPoints = useTrack((s) => s.showPoints);
  const moveMode = useTrack((s) => s.moveMode);
  useTrack((s) => s.tick);
  useTrack((s) => s.regions);
  const project = useStore((s) => s.project);
  const time = useStore((s) => s.time);
  const selection = useStore((s) => s.selection);
  useStore((s) => s.mediaTick);
  const draft = useTrack((s) => s.lockDraft);
  const svg = useRef<SVGSVGElement>(null);
  const [lockDrag, setLockDrag] = useState<{ x: boolean; y: boolean } | null>(null);

  if (!active) return null;
  const clip = selection.length === 1 ? project.clips.find((c) => c.id === selection[0]) : undefined;
  if (!clip || clip.type !== 'video') return null;
  const box = engine.boxOf(clip.id);
  if (!box) return null;
  const g: Geom = { cx: box.cx, cy: box.cy, scale: 1, rotation: box.rotation, w: box.w, h: box.h, flip: clip.flipH };
  const aspect = assetAspect(project, clip);
  const settings = effectiveSettings(settingsRaw);
  const srcTime = sourceTimeOf(clip, time);
  const idle = phase === 'idle';

  const P = (sx: number, sy: number): [number, number] => {
    const [x, y] = sourceToCanvas(g, sx, sy);
    return [x * k, y * k];
  };
  const poly = (q: ArrayLike<number>, o = 0) => [0, 1, 2, 3].map((i) => P(q[o + i * 2], q[o + i * 2 + 1]).join(',')).join(' ');
  const toSrc = (ev: { clientX: number; clientY: number }): [number, number] => {
    const r = svg.current!.getBoundingClientRect();
    return canvasToSource(g, (ev.clientX - r.left) / k, (ev.clientY - r.top) / k);
  };
  const setRegion = (q: number[]) => useTrack.getState().setRegion(clip.id, q);

  const nodes: React.ReactNode[] = [];

  // ── committed motion at the playhead ──
  const lock = clip.tracking?.lock ?? (draft?.clipId === clip.id ? draft : null);
  const lockOn = !!lock?.enabled;
  const centre = trackerCentreCanvas(project, clip, time);
  const follow = !lock?.placed;
  const lockX = follow ? centre?.x ?? 0 : lock!.x;
  const lockY = follow ? centre?.y ?? 0 : lock!.y;
  const toNorm = (ev: { clientX: number; clientY: number }): [number, number] => {
    const r = svg.current!.getBoundingClientRect();
    const px = (ev.clientX - r.left) / k;
    const py = (ev.clientY - r.top) / k;
    return [(px - project.width / 2) / project.width, (py - project.height / 2) / project.height];
  };
  const lockPx = (project.width / 2 + lockX * project.width) * k;
  const lockPy = (project.height / 2 + lockY * project.height) * k;

  const m = motionAt(clip.tracking, srcTime, aspect);
  if (m && !live) {
    const col = m.searching ? C.searching : C.tracked;
    if (showPoints && !m.locked && !lockOn) {
      const path = motionPath(m.seg, aspect);
      const pts: string[] = [];
      for (let i = 0; i < path.length; i += 2) pts.push(P(path[i], path[i + 1]).join(','));
      nodes.push(<polyline key="path" points={pts.join(' ')} fill="none" stroke={C.tracked} strokeOpacity={0.55} strokeWidth={1.25} strokeLinejoin="round" />);
    }
    if (m.inside) {
      const dimmed = tab === 'track' && idle;
      if (m.seg.settings.shape === 'circle' && m.seg.settings.tracker !== 'frame') nodes.push(<Ellipse key="mq" q={m.raw} P={P} stroke={col} dash={m.searching} opacity={dimmed ? 0.5 : 1} />);
      else nodes.push(<polygon key="mq" points={poly(m.raw)} fill="none" stroke={col} strokeWidth={1.5} strokeDasharray={m.searching ? '5 4' : undefined} opacity={dimmed ? 0.5 : 1} />);
      const [cx, cy] = P(m.cx, m.cy);
      nodes.push(<Cross key="mc" x={cx} y={cy} color={col} />);
    }
  }

  // ── live analysis ──
  if (live && live.clipId === clip.id) {
    const isEdge = !!settings.edgeTrack;
    const col = live.state ? C.searching : isEdge ? C.edge : C.tracked;
    const n = live.t.length;
    if (n > 1) {
      const step = Math.max(1, Math.ceil(n / 240));
      const pts: string[] = [];
      for (let i = 0; i < n; i += step) pts.push(P(quadCx(live.q, i * 8), quadCy(live.q, i * 8)).join(','));
      nodes.push(<polyline key="lp" points={pts.join(' ')} fill="none" stroke={isEdge ? C.edge : C.tracked} strokeOpacity={0.7} strokeWidth={1.25} />);
    }
    if (settings.shape === 'circle' && settings.tracker !== 'frame') nodes.push(<Ellipse key="lr" q={live.roi} P={P} stroke={col} dash={!!live.state} opacity={1} />);
    else nodes.push(<polygon key="lr" points={poly(live.roi)} fill={col} fillOpacity={0.06} stroke={col} strokeWidth={1.75} strokeDasharray={live.state ? '5 4' : undefined} />);
    const last = (live.t.length - 1) * 8;
    const [lcx, lcy] = P(quadCx(live.q, last), quadCy(live.q, last));
    if (showPoints && live.points) {
      const p = live.points;
      const dots: React.ReactNode[] = [];
      const spokes: React.ReactNode[] = [];
      for (let i = 0; i < p.length; i += 3) {
        const st = p[i + 2];
        const [x, y] = P(p[i], p[i + 1]);
        if (isEdge) {
          const fill = st === POINT_INLIER ? C.edge : st === POINT_RESCUED ? C.rescued : st === POINT_OUTLIER ? C.lost : '#777';
          if (st === POINT_INLIER || st === POINT_RESCUED) {
            spokes.push(<line key={'s' + i} x1={lcx} y1={lcy} x2={x} y2={y} stroke={C.edge} strokeOpacity={0.14} strokeWidth={0.8} strokeDasharray="2 3" />);
          }
          dots.push(
            <circle
              key={i}
              cx={x}
              cy={y}
              r={st === POINT_INLIER ? 2.6 : st === POINT_RESCUED ? 2.3 : 1.8}
              fill={fill}
              fillOpacity={st === POINT_INLIER ? 0.98 : 0.78}
              stroke={C.ink}
              strokeWidth={st === POINT_INLIER || st === POINT_RESCUED ? 1.05 : 0.6}
            />,
          );
        } else {
          const fill = st === POINT_INLIER ? C.tracked : st === POINT_RESCUED ? C.rescued : st === POINT_LANDMARK ? C.fg : st === POINT_OUTLIER ? C.lost : '#777';
          dots.push(<circle key={i} cx={x} cy={y} r={st === POINT_LANDMARK ? 2.6 : 1.7} fill={fill} fillOpacity={st === POINT_INLIER || st === POINT_LANDMARK ? 0.95 : 0.7} />);
        }
      }
      if (spokes.length) nodes.push(<g key="espk" style={{ pointerEvents: 'none' }}>{spokes}</g>);
      nodes.push(<g key="pts" style={{ pointerEvents: 'none' }}>{dots}</g>);
    }
    nodes.push(<Cross key="lc" x={lcx} y={lcy} color={col} />);
  }

  // ── selector ──
  if (idle && tab === 'track') {
    const region = regionFor(clip, aspect, settingsRaw);
    if (settings.tracker === 'frame') {
      nodes.push(<polygon key="ff" points={poly(region)} fill="none" stroke={C.select} strokeWidth={2} strokeDasharray="8 6" />);
    } else if (settings.tracker === 'dual') {
      const [ax, ay, bx, by] = dualPoints(region);
      const A = P(ax, ay), B = P(bx, by);
      nodes.push(<line key="dl" x1={A[0]} y1={A[1]} x2={B[0]} y2={B[1]} stroke={C.select} strokeWidth={1.5} strokeDasharray="4 4" />);
      [A, B].forEach((pt, i) =>
        nodes.push(
          <g
            key={'d' + i}
            style={{ ...hot, cursor: 'move' }}
            onPointerDown={(e) =>
              drag(e, (ev) => {
                const [x, y] = toSrc(ev);
                const cur = dualPoints(regionFor(getState().project.clips.find((c) => c.id === clip.id) ?? clip, aspect));
                setRegion(i === 0 ? dualQuad(x, y, cur[2], cur[3], aspect) : dualQuad(cur[0], cur[1], x, y, aspect));
              })
            }
          >
            <circle cx={pt[0]} cy={pt[1]} r={16} fill="transparent" />
            <circle cx={pt[0]} cy={pt[1]} r={11} fill={C.select} fillOpacity={0.12} stroke={C.select} strokeWidth={1.75} />
            <Cross x={pt[0]} y={pt[1]} color={C.select} />
            <text x={pt[0] + 14} y={pt[1] - 12} fill={C.select} fontSize={11} fontWeight={700} style={{ pointerEvents: 'none' }}>
              {i === 0 ? 'A' : 'B'}
            </text>
          </g>,
        ),
      );
    } else {
      const semantic = isSemantic(settings.tracker);
      // detected faces / hands / bodies for this frame
      if (semantic && detected && detected.clipId === clip.id && detected.kind === settings.tracker && Math.abs(detected.srcTime - srcTime) < 0.2) {
        const picked = nearestEntity(detected.entities, quadCx(region), quadCy(region), aspect);
        detected.entities.forEach((ent, i) => {
          const on = i === picked;
          nodes.push(
            <polygon
              key={'e' + i}
              points={poly(ent.quad)}
              fill={on && settings.selection === 'entity' ? C.select : 'transparent'}
              fillOpacity={0.1}
              stroke={on ? C.select : C.fg}
              strokeOpacity={on ? 1 : 0.55}
              strokeWidth={on ? 2 : 1.25}
              strokeDasharray={on ? undefined : '4 4'}
              style={{ ...hot, cursor: 'pointer' }}
              onPointerDown={(e) => {
                e.stopPropagation();
                if (settings.selection !== 'area') setRegion(ent.quad.slice());
                else {
                  const r = rectOf(region, aspect);
                  setRegion(rectQuad(ent.cx, ent.cy, r.w, r.h, r.rot, aspect));
                }
              }}
            />,
          );
          if (on) {
            for (const def of LANDMARKS[settings.tracker as 'face']) {
              const [lx, ly] = landmarkPoint(ent, def);
              const [x, y] = P(lx, ly);
              const sel = settings.selection === 'points' && def.id === settings.landmark;
              nodes.push(
                <g key={'l' + i + def.id} style={{ ...hot, cursor: settings.selection === 'points' ? 'pointer' : 'default' }} onPointerDown={(e) => { e.stopPropagation(); if (settings.selection === 'points') useTrack.getState().setSettings({ landmark: def.id }); }}>
                  <circle cx={x} cy={y} r={9} fill="transparent" />
                  <circle cx={x} cy={y} r={sel ? 5 : 2.6} fill={sel ? C.select : C.fg} stroke={C.ink} strokeWidth={1} />
                  {sel && <circle cx={x} cy={y} r={10} fill="none" stroke={C.select} strokeWidth={1.5} />}
                </g>,
              );
            }
          }
        });
      }
      if (!semantic || settings.selection === 'area') {
        // Edge Track live preview: detected edges + placed edge tracking dots inside the tracking circle
        if (settings.edgeTrack) {
          const ep = edgePreview && edgePreview.clipId === clip.id && Math.abs(edgePreview.srcTime - srcTime) < 0.15 ? edgePreview : null;
          const [ccx, ccy] = P(quadCx(region), quadCy(region));
          const bot = P((region[4] + region[6]) / 2, (region[5] + region[7]) / 2);
          if (ep && settings.edgePreview !== false && ep.edges.length > 0) {
            const eNodes: React.ReactNode[] = [];
            for (let i = 0; i < ep.edges.length; i += 2) {
              const [ex, ey] = P(ep.edges[i], ep.edges[i + 1]);
              eNodes.push(<circle key={i} cx={ex} cy={ey} r={1.15} fill={C.edge} fillOpacity={0.5} />);
            }
            nodes.push(<g key="ep-edges" style={{ pointerEvents: 'none' }}>{eNodes}</g>);
          }
          nodes.push(<Selector key="sel" region={region} shape={settings.shape} aspect={aspect} P={P} toSrc={toSrc} onChange={setRegion} poly={poly} accent={C.edge} />);
          if (ep && showPoints && ep.dots.length > 0) {
            const spokes: React.ReactNode[] = [];
            const dNodes: React.ReactNode[] = [];
            for (let i = 0; i < ep.dots.length; i += 2) {
              const [dx, dy] = P(ep.dots[i], ep.dots[i + 1]);
              spokes.push(<line key={'s' + i} x1={ccx} y1={ccy} x2={dx} y2={dy} stroke={C.edge} strokeOpacity={0.16} strokeWidth={0.85} strokeDasharray="2 3" />);
              dNodes.push(<circle key={i} cx={dx} cy={dy} r={2.8} fill={C.edge} stroke={C.ink} strokeWidth={1.25} />);
            }
            nodes.push(<g key="ep-spokes" style={{ pointerEvents: 'none' }}>{spokes}</g>);
            nodes.push(<g key="ep-dots" style={{ pointerEvents: 'none' }}>{dNodes}</g>);
          }
          const labelText = ep ? `EDGE TRACK · ${ep.dotCount} DOTS` : 'EDGE TRACK';
          const pillW = Math.max(76, labelText.length * 5.6 + 14);
          nodes.push(
            <g key="ep-badge" style={{ pointerEvents: 'none' }}>
              <rect x={bot[0] - pillW / 2} y={bot[1] + 10} width={pillW} height={15} rx={7.5} fill={C.ink} fillOpacity={0.86} stroke={C.edge} strokeOpacity={0.75} />
              <text x={bot[0]} y={bot[1] + 20.5} textAnchor="middle" fill={C.edge} fontSize={8.5} fontWeight={700} letterSpacing={0.65}>
                {labelText}
              </text>
            </g>,
          );
        } else {
          nodes.push(<Selector key="sel" region={region} shape={settings.shape} aspect={aspect} P={P} toSrc={toSrc} onChange={setRegion} poly={poly} />);
        }
      }
    }
  }

  // ── edit motion: drag the tracked point ──
  if (idle && tab === 'edit' && m && m.inside) {
    const [x, y] = P(m.cx, m.cy);
    nodes.push(
      <g
        key="edit"
        style={{ ...hot, cursor: 'grab' }}
        onPointerDown={(e) => {
          const st = getState();
          const baseTracking: ClipTracking = clip.tracking!;
          const [x0, y0] = toSrc(e);
          let moved = false;
          engine.pause();
          drag(e, (ev) => {
            const [x1, y1] = toSrc(ev);
            if (!moved) {
              moved = true;
              st.beginGesture();
            }
            getState().updateClip(clip.id, { tracking: moveSample(baseTracking, m.seg.id, m.sample, x1 - x0, y1 - y0, moveMode) }, { history: false });
          });
        }}
      >
        <circle cx={x} cy={y} r={22} fill="transparent" />
        <circle cx={x} cy={y} r={13} fill={C.select} fillOpacity={0.18} stroke={C.select} strokeWidth={2} />
        <Cross x={x} y={y} color={C.fg} />
      </g>,
    );
  }

  const midX = (project.width / 2) * k;
  const midY = (project.height / 2) * k;
  return (
    <svg ref={svg} className="absolute inset-0 h-full w-full" style={{ pointerEvents: 'none', overflow: 'visible' }}>
      {nodes}
      {lockDrag?.x && <line x1={midX} y1={0} x2={midX} y2={project.height * k} stroke={C.select} strokeOpacity={0.55} strokeWidth={1} />}
      {lockDrag?.y && <line x1={0} y1={midY} x2={project.width * k} y2={midY} stroke={C.select} strokeOpacity={0.55} strokeWidth={1} />}
      <g
        style={{ ...hot, cursor: tab === 'edit' ? 'default' : 'move', pointerEvents: tab === 'edit' ? 'none' : 'auto' }}
        onPointerDown={(e) => {
          if (tab === 'edit') return;
          const [x0, y0] = toNorm(e);
          const baseX = lockX;
          const baseY = lockY;
          let moved = false;
          drag(
            e,
            (ev) => {
              const [x1, y1] = toNorm(ev);
              if (!moved) {
                if (Math.hypot((x1 - x0) * project.width, (y1 - y0) * project.height) * k < 3) return;
                moved = true;
                if (clip.tracking?.segments.length) getState().beginGesture();
              }
              const nx = baseX + (x1 - x0);
              const ny = baseY + (y1 - y0);
              const snap = !ev.shiftKey;
              placeLock(clip.id, nx, ny, { history: false, snap });
              setLockDrag(snap ? { x: Math.abs(nx) < 0.012, y: Math.abs(ny) < 0.012 } : { x: false, y: false });
            },
            () => setLockDrag(null),
          );
        }}
      >
        <circle cx={lockPx} cy={lockPy} r={28} fill="transparent" />
        {lockOn && <circle cx={lockPx} cy={lockPy} r={22} fill="none" stroke={C.select} strokeOpacity={0.35} strokeWidth={6} />}
        <circle cx={lockPx} cy={lockPy} r={15} fill={lockOn ? C.select : C.ink} fillOpacity={lockOn ? 0.16 : 0.55} stroke={C.select} strokeWidth={1.75} strokeDasharray={lockOn ? undefined : '3 2.5'} />
        <path d={`M${lockPx - 22} ${lockPy}H${lockPx - 11}M${lockPx + 11} ${lockPy}H${lockPx + 22}M${lockPx} ${lockPy - 22}V${lockPy - 11}M${lockPx} ${lockPy + 11}V${lockPy + 22}`} stroke={C.select} strokeWidth={1.5} strokeLinecap="round" />
        <path d={`M${lockPx - 5} ${lockPy}H${lockPx + 5}M${lockPx} ${lockPy - 5}V${lockPy + 5}`} stroke={C.fg} strokeWidth={1.5} strokeLinecap="round" />
        <rect x={lockPx - 3.2} y={lockPy - 3.2} width={6.4} height={6.4} transform={`rotate(45 ${lockPx} ${lockPy})`} fill={C.select} />
        <g style={{ pointerEvents: 'none' }}>
          <rect x={lockPx - 18} y={lockPy - 36} width={36} height={14} rx={7} fill={C.ink} fillOpacity={0.82} stroke={C.select} strokeOpacity={lockOn ? 1 : 0.45} />
          <text x={lockPx} y={lockPy - 26} textAnchor="middle" fill={lockOn ? C.select : C.fg} fontSize={8.5} fontWeight={700} letterSpacing={0.8}>
            {lockOn ? 'LOCKED' : 'LOCK'}
          </text>
        </g>
        <title>{lockOn ? 'Lock point — the tracker middle is held here. Drag to move it.' : 'Lock point — drag to choose where the tracker middle will stay, then turn Lock on.'}</title>
      </g>
    </svg>
  );
}

function Cross({ x, y, color }: { x: number; y: number; color: string }) {
  return (
    <g style={{ pointerEvents: 'none' }}>
      <path d={`M${x - 6} ${y}H${x + 6}M${x} ${y - 6}V${y + 6}`} stroke={C.ink} strokeWidth={3.5} strokeLinecap="round" />
      <path d={`M${x - 6} ${y}H${x + 6}M${x} ${y - 6}V${y + 6}`} stroke={color} strokeWidth={1.5} strokeLinecap="round" />
    </g>
  );
}

function Ellipse({ q, P, stroke, dash, opacity, fill }: { q: ArrayLike<number>; P: (x: number, y: number) => [number, number]; stroke: string; dash: boolean; opacity: number; fill?: boolean }) {
  const tl = P(q[0], q[1]), tr = P(q[2], q[3]), br = P(q[4], q[5]), bl = P(q[6], q[7]);
  const cx = (tl[0] + tr[0] + br[0] + bl[0]) / 4, cy = (tl[1] + tr[1] + br[1] + bl[1]) / 4;
  const rx = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]) / 2, ry = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]) / 2;
  const ang = (Math.atan2(tr[1] - tl[1], tr[0] - tl[0]) * 180) / Math.PI;
  return <ellipse cx={cx} cy={cy} rx={rx} ry={ry} transform={`rotate(${ang} ${cx} ${cy})`} fill={fill ? stroke : 'none'} fillOpacity={0.08} stroke={stroke} strokeWidth={1.75} strokeDasharray={dash ? '5 4' : undefined} opacity={opacity} />;
}

/** The editable region: drag the body to move, corners to resize (or reshape a quad), the stalk to rotate. */
function Selector({ region, shape, aspect, P, toSrc, onChange, poly, accent = C.select }: { region: number[]; shape: 'circle' | 'rectangle' | 'quad'; aspect: number; P: (x: number, y: number) => [number, number]; toSrc: (e: { clientX: number; clientY: number }) => [number, number]; onChange: (q: number[]) => void; poly: (q: ArrayLike<number>) => string; accent?: string }) {
  const corners = [0, 1, 2, 3].map((i) => P(region[i * 2], region[i * 2 + 1]));
  const move = (e: RPointerEvent) => {
    const [x0, y0] = toSrc(e);
    const base = region.slice();
    drag(e, (ev) => {
      const [x1, y1] = toSrc(ev);
      onChange(base.map((v, i) => v + (i % 2 ? y1 - y0 : x1 - x0)));
    });
  };
  const corner = (i: number) => (e: RPointerEvent) => {
    const r0 = rectOf(region, aspect);
    drag(e, (ev) => {
      const [x, y] = toSrc(ev);
      if (shape === 'quad') {
        const q = region.slice();
        q[i * 2] = x;
        q[i * 2 + 1] = y;
        onChange(q);
        return;
      }
      // resize about the centre, in the rectangle's own (pixel-true) frame
      const rot = shape === 'circle' ? 0 : r0.rot;
      const dx = (x - r0.cx) * aspect, dy = y - r0.cy;
      const lx = dx * Math.cos(-rot) - dy * Math.sin(-rot), ly = dx * Math.sin(-rot) + dy * Math.cos(-rot);
      onChange(rectQuad(r0.cx, r0.cy, Math.max(0.02, (Math.abs(lx) * 2) / aspect), Math.max(0.02, Math.abs(ly) * 2), rot, aspect));
    });
  };
  const rotate = (e: RPointerEvent) => {
    const r0 = rectOf(region, aspect);
    drag(e, (ev) => {
      const [x, y] = toSrc(ev);
      let rot = Math.atan2(y - r0.cy, (x - r0.cx) * aspect) + Math.PI / 2;
      for (const snap of [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, Math.PI]) if (Math.abs(rot - snap) < 0.05) rot = snap;
      onChange(rectQuad(r0.cx, r0.cy, r0.w, r0.h, rot, aspect));
    });
  };
  const top: [number, number] = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
  const ctr: [number, number] = [(corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2];
  const ux = top[0] - ctr[0], uy = top[1] - ctr[1];
  const ul = Math.hypot(ux, uy) || 1;
  const stalk: [number, number] = [top[0] + (ux / ul) * 26, top[1] + (uy / ul) * 26];
  return (
    <g>
      {shape === 'circle' ? (
        <>
          <g style={{ ...hot, cursor: 'move' }} onPointerDown={move}>
            <Ellipse q={region} P={P} stroke={accent} dash={false} opacity={1} fill />
          </g>
          <polygon points={poly(region)} fill="none" stroke={accent} strokeOpacity={0.35} strokeWidth={1} strokeDasharray="3 4" />
        </>
      ) : (
        <polygon points={poly(region)} fill={accent} fillOpacity={0.08} stroke={accent} strokeWidth={1.75} style={{ ...hot, cursor: 'move' }} onPointerDown={move} />
      )}
      <Cross x={ctr[0]} y={ctr[1]} color={accent} />
      {shape === 'rectangle' && (
        <g style={{ ...hot, cursor: 'grab' }} onPointerDown={rotate}>
          <line x1={top[0]} y1={top[1]} x2={stalk[0]} y2={stalk[1]} stroke={accent} strokeWidth={1.5} />
          <circle cx={stalk[0]} cy={stalk[1]} r={14} fill="transparent" />
          <circle cx={stalk[0]} cy={stalk[1]} r={5.5} fill={C.ink} stroke={accent} strokeWidth={2} />
        </g>
      )}
      {corners.map((c, i) => (
        <g key={i} style={{ ...hot, cursor: shape === 'quad' ? 'crosshair' : i % 2 ? 'nesw-resize' : 'nwse-resize' }} onPointerDown={corner(i)}>
          <circle cx={c[0]} cy={c[1]} r={15} fill="transparent" />
          {shape === 'quad' ? <circle cx={c[0]} cy={c[1]} r={6} fill={accent} stroke={C.ink} strokeWidth={2} /> : <rect x={c[0] - 5} y={c[1] - 5} width={10} height={10} rx={2} fill={C.fg} stroke={C.ink} strokeWidth={2} />}
        </g>
      ))}
    </g>
  );
}

/// <reference lib="webworker" />
// Tracking worker. Frames arrive as RGBA at analysis resolution; every tracker runs here so the editor stays
// responsive while a clip is analysed. Geometry crosses the boundary normalised to the frame (0–1).
import type { TrackSettings } from '../types';
import { NanoTracker, type OrtLike, type OrtSession } from '../lib/tracking/nano';
import { DualTracker, PointTracker, type MaskImage, type StepOut, type TrackerFrame } from '../lib/tracking/pointTracker';
import { buildPyramid, grayFromRGBA } from '../lib/tracking/vision';

interface FrameMsg {
  data: ArrayBuffer;
  w: number;
  h: number;
}
interface InMsg {
  id: number;
  type: 'init' | 'step' | 'dispose' | 'warm';
  settings?: TrackSettings;
  quad?: number[];
  frame?: FrameMsg;
  mask?: MaskImage | null;
  modelBase?: string;
}

type AnyTracker = { pyramidLevels: number; init(f: TrackerFrame, q: ArrayLike<number>): string | null | Promise<string | null>; step(f: TrackerFrame): StepOut | Promise<StepOut> };

let tracker: AnyTracker | null = null;
let needsColour = false;
let lastMask: MaskImage | null = null;

let nano: Promise<{ ort: OrtLike; backbone: OrtSession; head: OrtSession }> | null = null;

function loadNano(base: string) {
  if (!nano) {
    nano = (async () => {
      const ort = await import('onnxruntime-web');
      ort.env.wasm.wasmPaths = self.location.origin + '/ort/';
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) >> 1)) : 1;
      const fetchModel = async (name: string) => {
        const res = await fetch(base + name);
        if (!res.ok) throw new Error(`Could not load the tracking model (${name}: ${res.status})`);
        return new Uint8Array(await res.arrayBuffer());
      };
      const [bb, hd] = await Promise.all([fetchModel('nanotrack_backbone_dyn.onnx'), fetchModel('nanotrack_head_sim.onnx')]);
      const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' as const };
      const backbone = (await ort.InferenceSession.create(bb, opts)) as unknown as OrtSession;
      const head = (await ort.InferenceSession.create(hd, opts)) as unknown as OrtSession;
      return { ort: ort as unknown as OrtLike, backbone, head };
    })();
    nano.catch(() => (nano = null));
  }
  return nano;
}

function makeFrame(m: FrameMsg, levels: number): TrackerFrame {
  const rgba = new Uint8ClampedArray(m.data);
  const gray = levels > 0 ? grayFromRGBA(rgba, m.w, m.h) : new Float32Array(0);
  return { w: m.w, h: m.h, gray, pyr: levels > 0 ? buildPyramid(gray, m.w, m.h, levels) : [], rgba: needsColour ? rgba : null, mask: lastMask };
}

function normalise(r: StepOut, w: number, h: number) {
  const n = (q: number[]) => q.map((v, i) => (i % 2 ? v / h : v / w));
  let points: Float32Array | null = null;
  if (r.points) {
    points = r.points;
    for (let i = 0; i < points.length; i += 3) {
      points[i] /= w;
      points[i + 1] /= h;
    }
  }
  return { quad: n(r.quad), roi: n(r.roi), conf: r.conf, state: r.state, lost: r.lost, points };
}

const post = (msg: Record<string, unknown>, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

self.onmessage = async (e: MessageEvent<InMsg>) => {
  const m = e.data;
  try {
    if (m.type === 'warm') {
      await loadNano(m.modelBase!);
      post({ id: m.id, ok: true });
      return;
    }
    if (m.type === 'dispose') {
      tracker = null;
      lastMask = null;
      post({ id: m.id, ok: true });
      return;
    }
    if (m.mask !== undefined) lastMask = m.mask;
    if (m.type === 'init') {
      const s = m.settings!;
      const f = m.frame!;
      needsColour = s.tracker === 'object';
      if (s.tracker === 'object') {
        const n = await loadNano(m.modelBase!);
        tracker = new NanoTracker(n.ort, n.backbone, n.head, { followSize: s.followSize, mask: s.mask });
      } else if (s.tracker === 'dual') {
        tracker = new DualTracker(s.flowMode === 'advanced');
      } else {
        const kind = s.tracker === 'features' ? 'features' : s.tracker === 'frame' ? 'frame' : 'flow';
        tracker = new PointTracker({
          kind,
          model: s.model,
          shape: s.shape,
          followSize: s.followSize || s.cornerPin,
          followRotation: s.followRotation || s.cornerPin,
          cornerPin: s.cornerPin && s.shape === 'quad',
          advanced: s.flowMode === 'advanced',
          mask: s.mask,
        });
      }
      // enough pyramid for the largest tracker configuration at this resolution
      const levels = s.tracker === 'object' ? 0 : 4;
      const frame = makeFrame(f, levels);
      const quad = m.quad!.map((v, i) => (i % 2 ? v * f.h : v * f.w));
      const err = await tracker.init(frame, quad);
      if (err) {
        tracker = null;
        post({ id: m.id, ok: false, error: err });
        return;
      }
      post({ id: m.id, ok: true, levels });
      return;
    }
    if (m.type === 'step') {
      if (!tracker) throw new Error('tracker not initialised');
      const f = m.frame!;
      const frame = makeFrame(f, needsColour ? 0 : 4);
      const r = normalise(await tracker.step(frame), f.w, f.h);
      post({ id: m.id, ok: true, result: r }, r.points ? [r.points.buffer] : []);
    }
  } catch (err) {
    post({ id: m.id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};

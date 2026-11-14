// Main-thread handle on the tracking worker.
import type { TrackSettings } from '../../types';

export interface StepResult {
  /** recorded motion quad, normalised source coordinates */
  quad: number[];
  /** selector region as it follows the target */
  roi: number[];
  conf: number;
  state: 0 | 1;
  lost: boolean;
  /** x, y, status triples, normalised */
  points: Float32Array | null;
}

export interface MaskPayload {
  data: Uint8Array;
  w: number;
  h: number;
}

export const MODEL_BASE = '/models/tracking/';

let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, { resolve: (v: Record<string, unknown>) => void; reject: (e: Error) => void }>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('../../workers/tracking.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent) => {
      const p = pending.get(e.data.id);
      if (!p) return;
      pending.delete(e.data.id);
      p.resolve(e.data);
    };
    worker.onerror = (e) => {
      for (const p of pending.values()) p.reject(new Error(e.message || 'The tracking worker stopped unexpectedly'));
      pending.clear();
      worker = null;
    };
  }
  return worker;
}

function call(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, ...msg }, transfer);
  });
}

const base = () => location.origin + MODEL_BASE;

/** Pre-load the object-tracking network so the first frame is not held up by it. */
export async function warmObjectTracker(): Promise<void> {
  const r = await call({ type: 'warm', modelBase: base() });
  if (!r.ok) throw new Error(String(r.error));
}

/** Returns null on success, or the reason the selection cannot be tracked ('low-detail', …). */
export async function initTracker(settings: TrackSettings, quad: number[], frame: ImageData, mask?: MaskPayload | null): Promise<string | null> {
  const data = frame.data.buffer.slice(0);
  const r = await call({ type: 'init', settings, quad, frame: { data, w: frame.width, h: frame.height }, mask: mask ?? null, modelBase: base() }, [data]);
  return r.ok ? null : String(r.error);
}

export async function stepTracker(frame: ImageData, mask?: MaskPayload | null): Promise<StepResult> {
  const data = frame.data.buffer.slice(0);
  const msg: Record<string, unknown> = { type: 'step', frame: { data, w: frame.width, h: frame.height } };
  if (mask !== undefined) msg.mask = mask;
  const r = await call(msg, [data]);
  if (!r.ok) throw new Error(String(r.error));
  return r.result as StepResult;
}

export function disposeTracker() {
  if (worker) void call({ type: 'dispose' }).catch(() => undefined);
}

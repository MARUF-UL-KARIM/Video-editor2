// Sequential, frame-accurate access to a clip's source frames for analysis.
// Prefers WebCodecs (via mediabunny) and falls back to stepping a <video> element.
import { ALL_FORMATS, BlobSource, CanvasSink, Input } from 'mediabunny';
import type { Asset } from '../../types';
import { runtime } from '../runtime';
import { once } from '../util';

export interface FrameSource {
  w: number;
  h: number;
  /** the current frame, drawn at analysis resolution */
  canvas: HTMLCanvasElement;
  /** advance to the next frame; resolves to its source time in seconds, or null at the end */
  next(): Promise<number | null>;
  read(): ImageData;
  /** approximate frame interval of the source, seconds */
  frameDuration: number;
  dispose(): void;
}

export function analysisSize(asset: Pick<Asset, 'width' | 'height'>, longSide: number): { w: number; h: number } {
  const aw = asset.width || 1280, ah = asset.height || 720;
  const k = Math.min(1, longSide / Math.max(aw, ah));
  const even = (v: number) => Math.max(16, Math.round((v * k) / 2) * 2);
  return { w: even(aw), h: even(ah) };
}

export async function openFrameSource(asset: Asset, startSrc: number, longSide: number): Promise<FrameSource> {
  const rt = runtime.get(asset.id);
  if (!rt) throw new Error('This clip is offline — re-import its media to track it');
  const { w, h } = analysisSize(asset, longSide);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const read = () => ctx.getImageData(0, 0, w, h);

  // ── WebCodecs path ──
  let input: Input | null = null;
  try {
    input = new Input({ formats: ALL_FORMATS, source: new BlobSource(rt.file) });
    const track = await input.getPrimaryVideoTrack();
    if (!track || !(await track.canDecode())) throw new Error('cannot decode with WebCodecs');
    const first = await track.getFirstTimestamp();
    const sink = new CanvasSink(track, { width: w, height: h, fit: 'fill', poolSize: 2 });
    const it = sink.canvases(Math.max(first, first + startSrc - 1e-3));
    let done = false;
    let last = -1;
    let frameDuration = 1 / 30;
    const held = input;
    return {
      w,
      h,
      canvas,
      get frameDuration() {
        return frameDuration;
      },
      async next() {
        if (done) return null;
        const r = await it.next();
        if (r.done) {
          done = true;
          return null;
        }
        ctx.drawImage(r.value.canvas, 0, 0, w, h);
        const t = r.value.timestamp - first;
        if (last >= 0 && t > last) frameDuration = frameDuration * 0.8 + (t - last) * 0.2;
        last = t;
        return t;
      },
      read,
      dispose() {
        done = true;
        void it.return();
        held.dispose();
      },
    };
  } catch {
    input?.dispose();
  }

  // ── <video> fallback: slower, but works wherever the browser can play the file ──
  const el = document.createElement('video');
  el.muted = true;
  el.playsInline = true;
  el.preload = 'auto';
  el.src = rt.url;
  await once(el, ['loadeddata', 'error'], 15000);
  if (el.readyState < 2) throw new Error('This video could not be decoded for tracking');
  const step = 1 / 30;
  let t = Math.max(0, startSrc) - step;
  const end = (el.duration || asset.duration || 0) - 0.02;
  return {
    w,
    h,
    canvas,
    frameDuration: step,
    async next() {
      t += step;
      if (t > end) return null;
      el.currentTime = t;
      await once(el, ['seeked', 'error'], 4000);
      ctx.drawImage(el, 0, 0, w, h);
      return t;
    },
    read,
    dispose() {
      el.removeAttribute('src');
      el.load();
    },
  };
}

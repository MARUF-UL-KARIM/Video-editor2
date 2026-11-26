// Live Edge Track preview: detects edges inside the tracking circle on the current video frame
// so the user immediately sees the edge-detection filter output and placed tracking dots while
// positioning or resizing the tracking circle.
import { detectEdgesInCircle } from './edgeTracker';
import { grayFromRGBA } from './vision';

export interface EdgePreviewData {
  clipId: string;
  srcTime: number;
  /** normalised source coordinates [x0, y0, x1, y1, ...] of placed tracking dots */
  dots: Float32Array;
  /** normalised source coordinates [x0, y0, ...] of detected edge pixels for contour preview */
  edges: Float32Array;
  /** number of placed tracking dots */
  dotCount: number;
  /** total detected edge pixels inside the circle */
  edgeCount: number;
  /** edge contrast quality, 0–1 */
  quality: number;
}

let scratchCanvas: HTMLCanvasElement | null = null;
let scratchCtx: CanvasRenderingContext2D | null = null;

function ensureScratch(w: number, h: number): CanvasRenderingContext2D | null {
  if (!scratchCanvas) {
    scratchCanvas = document.createElement('canvas');
    scratchCtx = scratchCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (!scratchCtx) return null;
  if (scratchCanvas.width !== w) scratchCanvas.width = w;
  if (scratchCanvas.height !== h) scratchCanvas.height = h;
  return scratchCtx;
}

/**
 * Sample just the bounding box of the tracking circle from the live <video> element,
 * run the edge-detection filter, and return normalised source coordinates for edges + dots.
 */
export function sampleEdgePreview(
  video: HTMLVideoElement,
  clipId: string,
  srcTime: number,
  region: ArrayLike<number>,
  opts: { resolution: number; sensitivity: number; maxDots: number },
): EdgePreviewData | null {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh || video.readyState < 2) return null;

  const minNx = Math.max(0, Math.min(region[0], region[2], region[4], region[6]));
  const maxNx = Math.min(1, Math.max(region[0], region[2], region[4], region[6]));
  const minNy = Math.max(0, Math.min(region[1], region[3], region[5], region[7]));
  const maxNy = Math.min(1, Math.max(region[1], region[3], region[5], region[7]));

  const spanX = maxNx - minNx;
  const spanY = maxNy - minNy;
  if (spanX < 0.01 || spanY < 0.01) {
    return { clipId, srcTime, dots: new Float32Array(0), edges: new Float32Array(0), dotCount: 0, edgeCount: 0, quality: 0 };
  }

  // Analysis-equivalent scale with a small pixel padding around the circle for clean 3×3 borders
  const scale = Math.min(1, opts.resolution / Math.max(vw, vh));
  const padPx = Math.max(6, Math.round(6 / Math.max(0.1, scale)));
  const sx = Math.max(0, Math.floor(minNx * vw) - padPx);
  const sy = Math.max(0, Math.floor(minNy * vh) - padPx);
  const sx1 = Math.min(vw, Math.ceil(maxNx * vw) + padPx);
  const sy1 = Math.min(vh, Math.ceil(maxNy * vh) + padPx);
  const sw = sx1 - sx;
  const sh = sy1 - sy;
  if (sw < 8 || sh < 8) return null;

  // Keep the crop canvas compact for instant 60fps feedback while dragging
  const cap = Math.min(scale, 320 / Math.max(sw, sh));
  const cw = Math.max(16, Math.round(sw * cap));
  const ch = Math.max(16, Math.round(sh * cap));

  const ctx = ensureScratch(cw, ch);
  if (!ctx) return null;

  try {
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, cw, ch);
    const img = ctx.getImageData(0, 0, cw, ch);
    const gray = grayFromRGBA(img.data, cw, ch);

    // Map normalised source quad into crop canvas coordinates
    const kx = cw / sw;
    const ky = ch / sh;
    const localQuad = new Float64Array(8);
    for (let i = 0; i < 4; i++) {
      localQuad[i * 2] = (region[i * 2] * vw - sx) * kx;
      localQuad[i * 2 + 1] = (region[i * 2 + 1] * vh - sy) * ky;
    }

    const res = detectEdgesInCircle(gray, cw, ch, localQuad, {
      maxDots: opts.maxDots,
      sensitivity: opts.sensitivity,
      maxPreviewEdges: 320,
      shrink: 0.05,
    });

    const dots = new Float32Array(res.dots.length);
    for (let i = 0; i < res.dots.length; i += 2) {
      dots[i] = (sx + res.dots[i] / kx) / vw;
      dots[i + 1] = (sy + res.dots[i + 1] / ky) / vh;
    }

    const edges = new Float32Array(res.edgePixels.length);
    for (let i = 0; i < res.edgePixels.length; i += 2) {
      edges[i] = (sx + res.edgePixels[i] / kx) / vw;
      edges[i + 1] = (sy + res.edgePixels[i + 1] / ky) / vh;
    }

    return {
      clipId,
      srcTime,
      dots,
      edges,
      dotCount: dots.length >> 1,
      edgeCount: res.edgeCount,
      quality: res.quality,
    };
  } catch {
    return null;
  }
}

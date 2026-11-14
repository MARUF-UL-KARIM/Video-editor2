// Object tracker: NanoTrack v2 (Siamese network, ONNX). The same two models the reference app ships
// (backbone + correlation head), run fully on-device through onnxruntime-web.
//
// Pipeline (identical to the classic SiamFC/SiamRPN family, as used by OpenCV's TrackerNano):
//   init   – crop a 127×127 exemplar around the target (with context), embed it with the backbone.
//   update – crop a 255×255 search window around the last position, embed, correlate with the exemplar in the head;
//            decode the 16×16 score / box maps, apply scale-change and cosine-window penalties, pick the best cell,
//            and smooth the box size with a confidence-weighted learning rate.
// On top of that: a searching state with a bounded ring scan around the last known position, so a short occlusion
// does not end the track.
import type { MaskImage, StepOut, TrackerFrame } from './pointTracker';

export interface OrtTensor {
  data: Float32Array;
  dims: readonly number[];
}
export interface OrtSession {
  inputNames: readonly string[];
  outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
}
export interface OrtLike {
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
}

const EXEMPLAR = 127;
const INSTANCE = 255;
const STRIDE = 16;
const SCORE = (INSTANCE - EXEMPLAR) / STRIDE + 8; // 16
const CONTEXT = 0.5;
const WINDOW_INFLUENCE = 0.455;
const PENALTY_K = 0.055;
const LR = 0.37;

const hann = (() => {
  const w = new Float32Array(SCORE * SCORE);
  const h1 = new Float32Array(SCORE);
  for (let i = 0; i < SCORE; i++) h1[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (SCORE - 1));
  for (let y = 0; y < SCORE; y++) for (let x = 0; x < SCORE; x++) w[y * SCORE + x] = h1[y] * h1[x];
  return w;
})();

const sizeCal = (w: number, h: number) => {
  const pad = (w + h) * 0.5;
  return Math.sqrt((w + pad) * (h + pad));
};

export class NanoTracker {
  private ort: OrtLike;
  private backbone: OrtSession;
  private head: OrtSession;
  private zf: unknown = null;
  private px = 0;
  private py = 0;
  private tw = 0;
  private th = 0;
  private w0 = 0;
  private h0 = 0;
  private avg: [number, number, number] = [0, 0, 0];
  private searching = 0;
  private followSize: boolean;
  private maskMode: 'none' | 'subject' | 'background';
  /** last raw network score, 0–1 */
  score = 0;

  constructor(ort: OrtLike, backbone: OrtSession, head: OrtSession, opts: { followSize: boolean; mask: 'none' | 'subject' | 'background' }) {
    this.ort = ort;
    this.backbone = backbone;
    this.head = head;
    this.followSize = opts.followSize;
    this.maskMode = opts.mask;
  }

  get pyramidLevels() {
    return 0;
  }

  /** Square crop of side `size` centred on (cx, cy), resampled to out×out, CHW float RGB. Outside pixels take the frame's mean colour. */
  private crop(f: TrackerFrame, cx: number, cy: number, size: number, out: number, mask: MaskImage | null): Float32Array {
    const rgba = f.rgba!;
    const { w, h } = f;
    const plane = out * out;
    const data = new Float32Array(3 * plane);
    const c = (size + 1) / 2;
    const x0 = Math.floor(cx) - c;
    const y0 = Math.floor(cy) - c;
    const k = size / out;
    const [ar, ag, ab] = this.avg;
    for (let j = 0; j < out; j++) {
      const sy = y0 + (j + 0.5) * k - 0.5;
      const yi = Math.floor(sy);
      const fy = sy - yi;
      for (let i = 0; i < out; i++) {
        const sx = x0 + (i + 0.5) * k - 0.5;
        const xi = Math.floor(sx);
        const fx = sx - xi;
        let r = 0, g = 0, b = 0;
        for (let t = 0; t < 4; t++) {
          const xx = xi + (t & 1), yy = yi + (t >> 1);
          const wgt = (t & 1 ? fx : 1 - fx) * (t >> 1 ? fy : 1 - fy);
          if (wgt === 0) continue;
          let pr = ar, pg = ag, pb = ab;
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) {
            let keep = true;
            if (mask) {
              const m = mask.data[Math.min(mask.h - 1, Math.floor((yy / h) * mask.h)) * mask.w + Math.min(mask.w - 1, Math.floor((xx / w) * mask.w))];
              keep = this.maskMode === 'subject' ? m > 96 : m < 128;
            }
            if (keep) {
              const o = (yy * w + xx) * 4;
              pr = rgba[o];
              pg = rgba[o + 1];
              pb = rgba[o + 2];
            }
          }
          r += pr * wgt;
          g += pg * wgt;
          b += pb * wgt;
        }
        const o = j * out + i;
        data[o] = r;
        data[plane + o] = g;
        data[2 * plane + o] = b;
      }
    }
    return data;
  }

  private mean(f: TrackerFrame) {
    const rgba = f.rgba!;
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < rgba.length; i += 4 * 7) {
      r += rgba[i];
      g += rgba[i + 1];
      b += rgba[i + 2];
      n++;
    }
    this.avg = [r / n, g / n, b / n];
  }

  private async embed(data: Float32Array, size: number): Promise<unknown> {
    const t = new this.ort.Tensor('float32', data, [1, 3, size, size]);
    const out = await this.backbone.run({ [this.backbone.inputNames[0]]: t });
    return out[this.backbone.outputNames[0]];
  }

  async init(f: TrackerFrame, quad: ArrayLike<number>): Promise<string | null> {
    if (!f.rgba) return 'no-colour';
    const xs = [quad[0], quad[2], quad[4], quad[6]], ys = [quad[1], quad[3], quad[5], quad[7]];
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    this.tw = this.w0 = Math.max(4, x1 - x0);
    this.th = this.h0 = Math.max(4, y1 - y0);
    this.px = (x0 + x1) / 2;
    this.py = (y0 + y1) / 2;
    this.searching = 0;
    this.mean(f);
    const sum = this.tw + this.th;
    const sz = Math.sqrt((this.tw + CONTEXT * sum) * (this.th + CONTEXT * sum));
    // "Object tracking uses the mask at initialization": background (or subject) pixels are neutralised in the exemplar only.
    const mask = this.maskMode !== 'none' ? f.mask : null;
    this.zf = await this.embed(this.crop(f, this.px, this.py, Math.round(sz), EXEMPLAR, mask), EXEMPLAR);
    this.score = 1;
    return null;
  }

  private async evaluate(f: TrackerFrame, cx: number, cy: number) {
    const sum = this.tw + this.th;
    const sz = Math.sqrt((this.tw + CONTEXT * sum) * (this.th + CONTEXT * sum));
    const scaleZ = EXEMPLAR / sz;
    const sx = sz * (INSTANCE / EXEMPLAR);
    const xf = await this.embed(this.crop(f, cx, cy, Math.round(sx), INSTANCE, null), INSTANCE);
    const out = await this.head.run({ [this.head.inputNames[0]]: this.zf, [this.head.inputNames[1]]: xf });
    const cls = out[this.head.outputNames[0]].data;
    const box = out[this.head.outputNames[1]].data;
    const N = SCORE * SCORE;
    const tw = this.tw * scaleZ, th = this.th * scaleZ;
    const ref = sizeCal(tw, th);
    const ratio = tw / th;
    let best = -1, bestP = -1, bestScore = 0, bestPen = 0;
    let bx1 = 0, by1 = 0, bx2 = 0, by2 = 0;
    for (let i = 0; i < N; i++) {
      const a = cls[i], b = cls[N + i];
      const m = Math.max(a, b);
      const score = Math.exp(b - m) / (Math.exp(a - m) + Math.exp(b - m));
      const gx = ((i % SCORE) - SCORE / 2) * STRIDE + INSTANCE / 2;
      const gy = (Math.floor(i / SCORE) - SCORE / 2) * STRIDE + INSTANCE / 2;
      const x1 = gx - box[i], y1 = gy - box[N + i], x2 = gx + box[2 * N + i], y2 = gy + box[3 * N + i];
      const pw = Math.max(1e-3, x2 - x1), ph = Math.max(1e-3, y2 - y1);
      let sc = sizeCal(pw, ph) / ref;
      sc = Math.max(sc, 1 / sc);
      let rc = ratio / (pw / ph);
      rc = Math.max(rc, 1 / rc);
      const penalty = Math.exp(-(rc * sc - 1) * PENALTY_K);
      const p = penalty * score * (1 - WINDOW_INFLUENCE) + hann[i] * WINDOW_INFLUENCE;
      if (p > bestP) {
        bestP = p;
        best = i;
        bestScore = score;
        bestPen = penalty;
        bx1 = x1; by1 = y1; bx2 = x2; by2 = y2;
      }
    }
    void best;
    const nx = cx + ((bx1 + bx2) / 2 - INSTANCE / 2) / scaleZ;
    const ny = cy + ((by1 + by2) / 2 - INSTANCE / 2) / scaleZ;
    const lr = bestPen * bestScore * LR;
    const nw = ((bx2 - bx1) / scaleZ) * lr + (1 - lr) * this.tw;
    const nh = ((by2 - by1) / scaleZ) * lr + (1 - lr) * this.th;
    return { score: bestScore, x: nx, y: ny, w: nw, h: nh };
  }

  private result(state: 0 | 1, conf: number, lost: boolean): StepOut {
    const w = this.followSize ? this.tw : this.w0, h = this.followSize ? this.th : this.h0;
    const x = this.px, y = this.py;
    const quad = [x - w / 2, y - h / 2, x + w / 2, y - h / 2, x + w / 2, y + h / 2, x - w / 2, y + h / 2];
    return { quad, roi: quad, conf, state, lost, points: null };
  }

  async step(f: TrackerFrame): Promise<StepOut> {
    if (!f.rgba || !this.zf) return this.result(1, 0, true);
    this.mean(f);
    let cx = this.px, cy = this.py;
    if (this.searching > 0) {
      // ring scan: look one half search-window away, rotating around the last known position
      const reach = Math.max(this.tw, this.th) * (0.9 + 0.12 * Math.min(8, this.searching >> 3));
      const ang = (this.searching * 2 * Math.PI) / 8;
      if (this.searching % 3 !== 0) {
        cx += Math.cos(ang) * reach;
        cy += Math.sin(ang) * reach;
      }
    }
    const r = await this.evaluate(f, cx, cy);
    this.score = r.score;
    const accept = this.searching > 0 ? r.score >= 0.5 : r.score >= 0.22;
    if (!accept) {
      this.searching++;
      return this.result(1, 0, this.searching > 60);
    }
    this.searching = 0;
    this.px = Math.max(0, Math.min(f.w, r.x));
    this.py = Math.max(0, Math.min(f.h, r.y));
    this.tw = Math.max(6, Math.min(f.w, r.w));
    this.th = Math.max(6, Math.min(f.h, r.h));
    return this.result(0, Math.min(1, r.score), false);
  }
}

// Puts the on-device model runtimes and weights where the app expects them (public/).
// Large binaries do not always survive a git push, so this runs before `dev` and `build` and quietly restores
// anything that is missing: WebAssembly runtimes are copied from node_modules, model weights are fetched once.
// It never fails the build — at runtime the app falls back to the publishers' CDNs for models it cannot find.
import { copyFileSync, existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pub = (...p) => join(root, 'public', ...p);
const mod = (...p) => join(root, 'node_modules', ...p);
const present = (file, min) => existsSync(file) && statSync(file).size >= min;

const copies = [
  [mod('onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.jsep.wasm'), pub('ort', 'ort-wasm-simd-threaded.jsep.wasm')],
  [mod('onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.jsep.mjs'), pub('ort', 'ort-wasm-simd-threaded.jsep.mjs')],
  [mod('@mediapipe', 'tasks-vision', 'wasm', 'vision_wasm_internal.wasm'), pub('mediapipe', 'wasm', 'vision_wasm_internal.wasm')],
  [mod('@mediapipe', 'tasks-vision', 'wasm', 'vision_wasm_internal.js'), pub('mediapipe', 'wasm', 'vision_wasm_internal.js')],
];

const GCS = 'https://storage.googleapis.com/mediapipe-models';
const downloads = [
  [`${GCS}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`, pub('mediapipe', 'selfie_multiclass_256x256.tflite'), 5e6],
  [`${GCS}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`, pub('mediapipe', 'models', 'face_landmarker.task'), 1e6],
  [`${GCS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`, pub('mediapipe', 'models', 'hand_landmarker.task'), 1e6],
  [`${GCS}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`, pub('mediapipe', 'models', 'pose_landmarker_lite.task'), 1e6],
];

let restored = 0;
for (const [from, to] of copies) {
  if (existsSync(to) && existsSync(from) && statSync(to).size === statSync(from).size) continue;
  if (!existsSync(from)) {
    if (!existsSync(to)) console.warn(`[assets] missing ${from} — run npm install`);
    continue;
  }
  mkdirSync(dirname(to), { recursive: true });
  copyFileSync(from, to);
  restored++;
}

// Hosted builds have a size budget for uploaded files; there the app streams these weights from the CDN instead.
const skipDownloads = !!(process.env.VERCEL || process.env.CI || process.env.OFFCUT_SKIP_MODELS);
for (const [url, to, min] of downloads) {
  if (skipDownloads || present(to, min)) continue;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length < min) throw new Error('unexpectedly small file');
    mkdirSync(dirname(to), { recursive: true });
    writeFileSync(to, buf);
    restored++;
  } catch (e) {
    console.warn(`[assets] could not fetch ${url.split('/').pop()} (${e.message}) — the app will load it from the CDN at runtime`);
  }
}

for (const f of ['nanotrack_backbone_dyn.onnx', 'nanotrack_head_sim.onnx']) {
  if (!present(pub('models', 'tracking', f), 5e5)) console.warn(`[assets] public/models/tracking/${f} is missing — the Object tracker will not load`);
}
if (restored) console.log(`[assets] restored ${restored} model/runtime file${restored > 1 ? 's' : ''} into public/`);

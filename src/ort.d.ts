// onnxruntime-web ships its typings outside its package "exports" map; this is the small surface the tracker uses.
declare module 'onnxruntime-web' {
  export const env: { wasm: { wasmPaths?: string | Record<string, string>; numThreads?: number } };
  export class Tensor {
    constructor(type: 'float32', data: Float32Array, dims: readonly number[]);
    data: Float32Array;
    dims: readonly number[];
  }
  export const InferenceSession: {
    create(model: Uint8Array | ArrayBuffer, options?: Record<string, unknown>): Promise<unknown>;
  };
}

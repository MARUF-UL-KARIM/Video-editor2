// Model files are bundled with the app (public/), but large binaries are sometimes missing from a fresh checkout.
// Resolve to the local copy when it is really there, otherwise to the publisher's CDN.
const cache = new Map<string, Promise<string>>();

export function resolveModelUrl(local: string, remote: string): Promise<string> {
  let p = cache.get(local);
  if (!p) {
    p = (async () => {
      try {
        const res = await fetch(local, { method: 'HEAD' });
        const type = res.headers.get('content-type') ?? '';
        const size = Number(res.headers.get('content-length') ?? '0');
        // a dev server / SPA fallback answers 200 with index.html for files that do not exist
        if (res.ok && !type.includes('text/html') && (size === 0 || size > 50_000)) return local;
      } catch {
        /* offline or blocked: fall through */
      }
      return remote;
    })();
    cache.set(local, p);
  }
  return p;
}

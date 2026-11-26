// Generates a crisp in-browser sample MP4 when no bundled /sample/sample-clip.mp4 is present,
// so users can immediately test editing and Edge Track motion tracking without uploading footage.
import { BufferTarget, CanvasSource, getFirstEncodableVideoCodec, Mp4OutputFormat, Output, WebMOutputFormat } from 'mediabunny';

export async function generateSampleClip(): Promise<File> {
  const width = 960;
  const height = 540;
  const fps = 30;
  const duration = 4;
  const totalFrames = fps * duration;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;

  const mp4Format = new Mp4OutputFormat();
  let codec = await getFirstEncodableVideoCodec(mp4Format.getSupportedVideoCodecs(), { width, height });
  let format: Mp4OutputFormat | WebMOutputFormat = mp4Format;
  let ext = 'mp4';
  let mime = 'video/mp4';

  if (!codec) {
    const webmFormat = new WebMOutputFormat();
    codec = await getFirstEncodableVideoCodec(webmFormat.getSupportedVideoCodecs(), { width, height });
    format = webmFormat;
    ext = 'webm';
    mime = 'video/webm';
  }
  if (!codec) throw new Error('No supported video encoder available');

  const target = new BufferTarget();
  const output = new Output({ format, target });
  const source = new CanvasSource(canvas, { codec, bitrate: 3_500_000 });
  output.addVideoTrack(source, { frameRate: fps });
  await output.start();

  for (let f = 0; f < totalFrames; f++) {
    const t = f / fps;
    const u = t / duration;

    // Deep studio slate gradient background with subtle camera pan
    const panX = Math.sin(u * Math.PI * 2) * 24;
    const panY = Math.cos(u * Math.PI * 2) * 12;
    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, '#12161d');
    bg.addColorStop(0.55, '#1b222c');
    bg.addColorStop(1, '#10141a');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    // Subtle architectural grid lines in background
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    const step = 60;
    for (let x = ((panX % step) + step) % step; x < width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = ((panY % step) + step) % step; y < height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    ctx.restore();

    // Moving high-contrast primary target (smooth figure-8 / arc with subtle scale breathing)
    const cx = width * 0.5 + Math.sin(u * Math.PI * 2) * 210 + Math.sin(u * Math.PI * 4) * 28;
    const cy = height * 0.5 + Math.sin(u * Math.PI * 2 * 1.5) * 95;
    const sc = 1 + 0.14 * Math.sin(u * Math.PI * 2);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(sc, sc);

    // Drop shadow
    ctx.fillStyle = 'rgba(0,0,0,0.42)';
    ctx.beginPath();
    ctx.arc(6, 12, 64, 0, Math.PI * 2);
    ctx.fill();

    // Outer metallic ring
    ctx.fillStyle = '#eceae2';
    ctx.beginPath();
    ctx.arc(0, 0, 62, 0, Math.PI * 2);
    ctx.fill();

    // Inner dark bezel
    ctx.fillStyle = '#181c22';
    ctx.beginPath();
    ctx.arc(0, 0, 48, 0, Math.PI * 2);
    ctx.fill();

    // Crisp notched aperture blades / geometric markings (high-contrast edges at diverse angles)
    ctx.fillStyle = '#ff6a3d';
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI) / 3;
      ctx.save();
      ctx.rotate(a);
      ctx.beginPath();
      ctx.moveTo(16, -6);
      ctx.lineTo(39, -10);
      ctx.lineTo(39, 10);
      ctx.lineTo(16, 6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // Center lens core with crisp specular ring
    ctx.fillStyle = '#f8f7f2';
    ctx.beginPath();
    ctx.arc(0, 0, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#141920';
    ctx.beginPath();
    ctx.arc(0, 0, 7, 0, Math.PI * 2);
    ctx.fill();

    // Outer registration tabs (strong corners along the silhouette)
    ctx.fillStyle = '#38e8d0';
    for (let i = 0; i < 4; i++) {
      const a = (i * Math.PI) / 2 + Math.PI / 4;
      ctx.save();
      ctx.rotate(a);
      ctx.fillRect(52, -5, 14, 10);
      ctx.restore();
    }

    ctx.restore();

    // Secondary background pill to test isolation
    ctx.save();
    ctx.translate(width * 0.16 - panX * 0.5, height * 0.22 - panY * 0.5);
    ctx.strokeStyle = 'rgba(236, 234, 226, 0.28)';
    ctx.lineWidth = 2;
    ctx.strokeRect(-42, -18, 84, 36);
    ctx.fillStyle = 'rgba(236, 234, 226, 0.45)';
    ctx.font = '600 11px monospace';
    ctx.fillText(`F ${String(f + 1).padStart(3, '0')}`, -24, 4);
    ctx.restore();

    await source.add(t, 1 / fps);
  }

  source.close();
  await output.finalize();
  if (!target.buffer) throw new Error('Sample clip encoding failed');
  return new File([target.buffer], `Sample motion clip.${ext}`, { type: mime });
}

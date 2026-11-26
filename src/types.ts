export type TrackKind = 'video' | 'text' | 'audio';
export interface Track {
  id: string;
  kind: TrackKind;
  name: string;
  muted: boolean;
  hidden: boolean;
  locked: boolean;
}

export type ClipType = 'video' | 'image' | 'audio' | 'text';
export type AnimProp = 'x' | 'y' | 'scale' | 'rotation' | 'opacity';
export type Ease = 'linear' | 'easeIn' | 'easeOut' | 'easeInOut' | 'hold';
export interface Keyframe {
  t: number;
  v: number;
  ease: Ease;
}

export interface Filters {
  brightness: number;
  contrast: number;
  saturation: number;
  temperature: number;
  tint: number;
  hue: number;
  fade: number;
  blur: number;
  sharpen: number;
  vignette: number;
  grain: number;
  rgbSplit: number;
}

export interface Chroma {
  enabled: boolean;
  color: string;
  similarity: number;
  smoothness: number;
}

export type TransitionType =
  | 'crossfade'
  | 'dipBlack'
  | 'dipWhite'
  | 'slideLeft'
  | 'slideRight'
  | 'slideUp'
  | 'slideDown'
  | 'wipeLeft'
  | 'wipeRight'
  | 'wipeUp'
  | 'wipeDown'
  | 'zoomIn'
  | 'zoomOut'
  | 'blur'
  | 'spin';

export interface Transition {
  type: TransitionType;
  duration: number;
}

export type ClipAnim = 'none' | 'fade' | 'pop' | 'slideUp' | 'slideDown' | 'zoom' | 'typewriter';
export type BlendMode = 'normal' | 'screen' | 'multiply' | 'add';

export interface Word {
  text: string;
  start: number;
  end: number;
}

export interface TextStyle {
  fontFamily: string;
  fontSize: number;
  weight: number;
  italic: boolean;
  uppercase: boolean;
  color: string;
  align: 'left' | 'center' | 'right';
  lineHeight: number;
  letterSpacing: number;
  strokeColor: string;
  strokeWidth: number;
  shadowColor: string;
  shadowBlur: number;
  shadowY: number;
  bgColor: string;
  bgOpacity: number;
  bgPadding: number;
  bgRadius: number;
  maxWidth: number;
  highlight: 'none' | 'color' | 'box';
  highlightColor: string;
}

export interface Clip {
  id: string;
  trackId: string;
  type: ClipType;
  assetId?: string;
  name: string;
  start: number;
  duration: number;
  inPoint: number;
  speed: number;
  // visual
  x: number;
  y: number;
  scale: number;
  rotation: number;
  opacity: number;
  fit: 'contain' | 'cover';
  flipH: boolean;
  radius: number;
  blend: BlendMode;
  filters: Filters;
  look: string;
  chroma: Chroma;
  removeBg: boolean;
  transitionIn: Transition | null;
  keyframes: Partial<Record<AnimProp, Keyframe[]>>;
  animIn: ClipAnim;
  animOut: ClipAnim;
  animDur: number;
  // audio
  volume: number;
  muted: boolean;
  fadeIn: number;
  fadeOut: number;
  denoise: boolean;
  enhance: boolean;
  normalize: boolean;
  // text
  text: string;
  style: TextStyle;
  words?: Word[];
  caption?: boolean;
  // motion tracking
  tracking?: ClipTracking;
  attach?: ClipAttach;
}

// ───────────────────────────── motion tracking ─────────────────────────────

/** Which engine follows the target. */
export type TrackerKind = 'object' | 'flow' | 'features' | 'frame' | 'dual' | 'face' | 'hands' | 'body';
export type TrackRegionShape = 'circle' | 'rectangle' | 'quad';
export type TrackMotionModel = 'translation' | 'similarity' | 'affine' | 'perspective';
/** How a semantic tracker (face / hands / body) picks its target. */
export type TrackSelectionMode = 'entity' | 'points' | 'area';
export type TrackMask = 'none' | 'subject' | 'background';
export type TrackFlowMode = 'fast' | 'advanced';
export type MotionComponent = 'horizontal' | 'vertical' | 'scale' | 'rotation' | 'affine' | 'perspective';
export type MotionFilter = 'off' | 'lock' | 'smooth' | 'follow' | 'stabilize';
export type TrackMoveMode = 'current' | 'forward' | 'both';

export interface TrackSettings {
  tracker: TrackerKind;
  shape: TrackRegionShape;
  model: TrackMotionModel;
  flowMode: TrackFlowMode;
  followSize: boolean;
  followRotation: boolean;
  cornerPin: boolean;
  mask: TrackMask;
  /** long side of the frames handed to the tracker, px */
  resolution: number;
  selection: TrackSelectionMode;
  /** landmark id used when `selection` is 'points' */
  landmark: string;
  /** Edge Track: detect edges inside the tracking circle and place tracking dots along them */
  edgeTrack?: boolean;
  /** Edge-detection filter sensitivity, 10–100 */
  edgeSensitivity?: number;
  /** Target number of tracking dots distributed along detected edges */
  edgeDots?: number;
  /** Show the live detected-edge contour overlay inside the tracking circle */
  edgePreview?: boolean;
}

export interface MotionAdjust {
  components: Record<MotionComponent, boolean>;
  filter: MotionFilter;
  /** 0–100 */
  strength: number;
}

/**
 * One continuous run of tracked motion. Samples are keyed by *source* time, so trimming, moving, splitting or
 * re-timing the clip never desynchronises them. Geometry is normalised to the source frame (0–1 on both axes).
 */
export interface TrackSegment {
  id: string;
  settings: TrackSettings;
  /** source time of each sample, seconds, ascending */
  t: number[];
  /** tracked quad per sample: TL, TR, BR, BL → 8 numbers each */
  q: number[];
  /** confidence per sample, 0–1 */
  c: number[];
  /** continuity per sample: 0 tracked, 1 searching (held / bridged) */
  s: number[];
  adjust: MotionAdjust;
}

/**
 * Point-lock stabilizer. `x` / `y` are the canvas position the tracker middle is held on,
 * in the same units as `clip.x` / `clip.y` (0 = frame centre).
 */
export interface TrackLock {
  enabled: boolean;
  x: number;
  y: number;
  /** the user placed this point; otherwise turning Lock on captures the tracker centre */
  placed: boolean;
}

export interface ClipTracking {
  id: string;
  segments: TrackSegment[];
  /** zoom in just enough that lock-on never reveals the edges of the frame */
  autoFit: boolean;
  /** hold the tracker middle on one canvas point so the picture stays stable */
  lock?: TrackLock;
}

/** Makes a layer ride on another clip's tracked motion. */
export interface ClipAttach {
  targetId: string;
  trackingId: string;
  /** where the target was on the canvas when the layer was attached (same units as clip.x / clip.y) */
  ax: number;
  ay: number;
  /** on-screen scale and rotation (deg) of the target at that moment */
  k0: number;
  r0: number;
  position: boolean;
  scale: boolean;
  rotation: boolean;
  /** map the layer's four corners onto the tracked quad */
  cornerPin: boolean;
}

export interface Asset {
  id: string;
  name: string;
  kind: 'video' | 'image' | 'audio';
  size: number;
  mime: string;
  duration: number;
  width: number;
  height: number;
  hasAudio: boolean;
  thumb: string;
}

export interface Project {
  version: 1;
  id: string;
  name: string;
  width: number;
  height: number;
  fps: number;
  background: string;
  tracks: Track[];
  clips: Clip[];
  assets: Asset[];
}

export type PanelId = 'media' | 'text' | 'captions' | 'voice' | 'looks' | 'transitions' | 'script' | 'tracking' | 'inspector' | null;

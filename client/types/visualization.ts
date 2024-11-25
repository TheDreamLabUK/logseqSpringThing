import type { Scene, PerspectiveCamera, WebGLRenderer, Vector3, Quaternion } from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import type { VisualizationConfig, BloomConfig, FisheyeConfig } from './components';

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>;
}

export interface Node {
  id: string;
  label?: string;
  position?: [number, number, number];
  color?: string;
  size?: number;
  type?: string;
  [key: string]: any;
}

export interface Edge {
  source: string;
  target: string;
  weight?: number;
  color?: string;
  width?: number;
  [key: string]: any;
}

export interface XRSessionManager {
  session: XRSession | null;
  referenceSpace: XRReferenceSpace | null;
  init(): Promise<void>;
  update(): void;
  dispose(): void;
}

export interface NodeManagerOptions {
  minNodeSize: number;
  maxNodeSize: number;
  nodeColor: string;
  edgeColor: string;
  edgeOpacity: number;
  labelFontSize: number;
  labelFontFamily: string;
  materialMetalness: number;
  materialRoughness: number;
  materialClearcoat: number;
  materialOpacity: number;
}

export interface VisualizationSettings {
  visualization: VisualizationConfig;
  bloom: BloomConfig;
  fisheye: FisheyeConfig;
}

export interface WebXRVisualizationState {
  initialized: boolean;
  pendingInitialization: boolean;
  scene: Scene | null;
  camera: PerspectiveCamera | null;
  renderer: WebGLRenderer | null;
  controls: OrbitControls | null;
  xrSessionManager: XRSessionManager | null;
  canvas: HTMLCanvasElement | null;
}

export interface CameraState {
  position: Vector3;
  rotation: Quaternion;
  target: Vector3;
}

export interface RenderState {
  fps: number;
  lastFrameTime: number;
  frameCount: number;
}

// Constants
export const VISUALIZATION_CONSTANTS = {
  TRANSLATION_SPEED: 0.01,
  ROTATION_SPEED: 0.01,
  VR_MOVEMENT_SPEED: 0.05,
  MIN_CAMERA_DISTANCE: 50,
  MAX_CAMERA_DISTANCE: 500,
  DEFAULT_FOV: 50,
  NEAR_PLANE: 0.1,
  FAR_PLANE: 2000,
  DEFAULT_CAMERA_POSITION: [0, 75, 200] as [number, number, number],
  DEFAULT_CAMERA_TARGET: [0, 0, 0] as [number, number, number]
} as const;

// WebGL Context Attributes
export const WEBGL_CONTEXT_ATTRIBUTES: WebGLContextAttributes = {
  alpha: false,
  antialias: true,
  powerPreference: "high-performance",
  failIfMajorPerformanceCaveat: false,
  preserveDrawingBuffer: true,
  xrCompatible: true
} as const;

// Renderer Settings
export const RENDERER_SETTINGS = {
  clearColor: 0x000000,
  clearAlpha: 1,
  pixelRatio: Math.min(window.devicePixelRatio, 2),
  toneMapping: 'ACESFilmic',
  toneMappingExposure: 1.5,
  outputColorSpace: 'srgb'
} as const;

// Light Settings
export const LIGHT_SETTINGS = {
  ambient: {
    color: 0xffffff,
    intensity: 1.5
  },
  directional: {
    color: 0xffffff,
    intensity: 2.0,
    position: [10, 20, 10] as [number, number, number]
  },
  hemisphere: {
    skyColor: 0xffffff,
    groundColor: 0x444444,
    intensity: 1.5
  },
  points: [
    {
      color: 0xffffff,
      intensity: 1.0,
      distance: 300,
      position: [100, 100, 100] as [number, number, number]
    },
    {
      color: 0xffffff,
      intensity: 1.0,
      distance: 300,
      position: [-100, -100, -100] as [number, number, number]
    }
  ]
} as const;

// Controls Settings
export const CONTROLS_SETTINGS = {
  enableDamping: true,
  dampingFactor: 0.1,
  rotateSpeed: 0.4,
  panSpeed: 0.6,
  zoomSpeed: 1.2,
  minDistance: 50,
  maxDistance: 500
} as const;

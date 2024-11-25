import type { Vector3, WebGLRenderer, PerspectiveCamera, Scene, Camera, OrthographicCamera } from 'three';

// Graph Types
export interface Node {
  id: string;
  label: string;
  position?: [number, number, number];
  velocity?: [number, number, number];
  size?: number;
  color?: string;
  type?: string;
  metadata?: Record<string, any>;
  userData?: {
    bloomLayer?: boolean;
    type?: string;
    [key: string]: any;
  };
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  weight?: number;
  width?: number;
  color?: string;
  type?: string;
  metadata?: Record<string, any>;
  userData?: {
    bloomLayer?: boolean;
    type?: string;
    [key: string]: any;
  };
}

// Aliases for compatibility
export type GraphNode = Node;
export type GraphEdge = Edge;

// Platform Types
export interface CoreState {
  renderer: WebGLRenderer | null;
  camera: Camera | null;
  scene: Scene | null;
  canvas: HTMLCanvasElement | null;
  isInitialized: boolean;
  isXRSupported: boolean;
  isWebGL2: boolean;
}

export interface BrowserCoreState extends CoreState {
  renderer: WebGLRenderer;
  camera: PerspectiveCamera | OrthographicCamera;
  scene: Scene;
  canvas: HTMLCanvasElement;
}

export interface XRCoreState extends CoreState {
  renderer: WebGLRenderer;
  camera: PerspectiveCamera;
  scene: Scene;
  canvas: HTMLCanvasElement;
}

export interface Transform {
  position: Vector3;
  rotation: Vector3;
  scale: Vector3;
}

export interface Viewport {
  width: number;
  height: number;
  pixelRatio: number;
}

export interface SceneConfig {
  antialias?: boolean;
  alpha?: boolean;
  preserveDrawingBuffer?: boolean;
  powerPreference?: 'high-performance' | 'low-power' | 'default';
}

export interface PerformanceConfig {
  maxFPS?: number;
  targetFrameTime?: number;
  enableAdaptiveQuality?: boolean;
  enableFrustumCulling?: boolean;
  enableOcclusionCulling?: boolean;
}

export interface PlatformCapabilities {
  webgl2: boolean;
  xr: boolean;
  multiview: boolean;
  instancedArrays: boolean;
  floatTextures: boolean;
  depthTexture: boolean;
  drawBuffers: boolean;
  shaderTextureLOD: boolean;
}

export interface InitializationOptions {
  canvas: HTMLCanvasElement;
  scene?: SceneConfig;
  performance?: PerformanceConfig;
}

// Graph Data Types
export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>;
}

export interface NodePosition {
  id: string;
  position: Vector3;
}

export interface NodeVelocity {
  id: string;
  velocity: Vector3;
}

export interface NodeUpdate {
  id: string;
  position?: Vector3;
  velocity?: Vector3;
  metadata?: Record<string, any>;
}

export interface GraphUpdate {
  nodes?: NodeUpdate[];
  edges?: Edge[];
  metadata?: Record<string, any>;
}

export interface GraphMetrics {
  nodeCount: number;
  edgeCount: number;
  density: number;
  averageDegree: number;
  clusteringCoefficient: number;
}

export interface GraphLayout {
  type: 'force' | 'circular' | 'grid' | 'random';
  options?: Record<string, any>;
}

export interface GraphBounds {
  min: Vector3;
  max: Vector3;
  center: Vector3;
  size: Vector3;
}

export interface GraphSelection {
  nodes: Set<string>;
  edges: Set<string>;
}

export interface GraphFilter {
  nodes?: (node: Node) => boolean;
  edges?: (edge: Edge) => boolean;
}

export interface GraphStats {
  fps: number;
  drawCalls: number;
  triangles: number;
  points: number;
  lines: number;
}

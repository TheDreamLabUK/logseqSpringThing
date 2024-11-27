import type { WebGLRenderer, Scene, PerspectiveCamera, Object3D, Vector3 } from 'three';

/**
 * Core visualization state interface
 */
export interface CoreState {
  renderer: WebGLRenderer | null;
  camera: PerspectiveCamera | null;
  scene: Scene | null;
  canvas: HTMLCanvasElement | null;
  isInitialized: boolean;
  isXRSupported: boolean;
  isWebGL2: boolean;
  isGPUMode: boolean;
  fps: number;
  lastFrameTime: number;
}

/**
 * Platform-specific core states
 */
export interface BrowserCoreState extends CoreState {
  type: 'browser';
}

export interface XRCoreState extends CoreState {
  type: 'xr';
  xrSession: any; // XRSession type from WebXR
}

/**
 * Transform interface for object positioning
 */
export interface Transform {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
}

/**
 * Viewport configuration
 */
export interface Viewport {
  width: number;
  height: number;
  pixelRatio: number;
}

/**
 * Scene configuration
 */
export interface SceneConfig {
  antialias: boolean;
  alpha: boolean;
  preserveDrawingBuffer: boolean;
  powerPreference: 'high-performance' | 'low-power' | 'default';
}

/**
 * Performance configuration
 */
export interface PerformanceConfig {
  targetFPS: number;
  maxDrawCalls: number;
  enableStats: boolean;
}

/**
 * Platform capabilities
 */
export interface PlatformCapabilities {
  webgl2: boolean;
  xr: boolean;
  maxTextureSize: number;
  maxDrawCalls: number;
  gpuTier: number;
}

/**
 * Node interfaces
 */
export interface Node {
  id: string;
  label?: string;
  position?: [number, number, number];
  velocity?: [number, number, number];
  size?: number;
  color?: string;
  type?: string;
  metadata?: Record<string, any>;
  userData?: Record<string, any>;
  weight?: number;  // Added to match Rust struct
  group?: string;   // Added to match Rust struct
}

export interface GraphNode extends Node {
  edges: GraphEdge[];  // Changed from Edge[] to GraphEdge[]
  weight: number;      // Required in GraphNode
  group?: string;
}

/**
 * Edge interfaces
 */
export interface Edge {
  id: string;
  source: string;
  target: string;
  weight?: number;
  width?: number;
  color?: string;
  type?: string;
  metadata?: Record<string, any>;
  userData?: Record<string, any>;
  directed?: boolean;  // Added to match Rust struct
}

export interface GraphEdge extends Edge {
  sourceNode: GraphNode;
  targetNode: GraphNode;
  directed: boolean;   // Required in GraphEdge
}

/**
 * Graph data structure
 */
export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: Record<string, any>;
}

/**
 * Fisheye effect settings
 */
export interface FisheyeSettings {
  enabled: boolean;
  strength: number;
  focusPoint: [number, number, number];
  radius: number;
}

/**
 * Material settings
 */
export interface MaterialSettings {
  nodeSize: number;
  nodeColor: string;
  edgeWidth: number;
  edgeColor: string;
  highlightColor: string;
  opacity: number;
  metalness: number;
  roughness: number;
}

/**
 * Physics simulation settings
 */
export interface PhysicsSettings {
  enabled: boolean;
  gravity: number;
  springLength: number;
  springStrength: number;
  repulsion: number;
  damping: number;
  timeStep: number;
}

/**
 * Bloom effect settings
 */
export interface BloomSettings {
  enabled: boolean;
  strength: number;
  radius: number;
  threshold: number;
}

/**
 * Complete visualization settings
 */
export interface VisualizationSettings {
  material: MaterialSettings;
  physics: PhysicsSettings;
  bloom: BloomSettings;
  fisheye: FisheyeSettings;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  fps: number;
  drawCalls: number;
  triangles: number;
  points: number;
}

/**
 * Camera state
 */
export interface CameraState {
  position: [number, number, number];
  target: [number, number, number];
  zoom: number;
}

/**
 * Renderer capabilities
 */
export interface RendererCapabilities {
  isWebGL2: boolean;
  maxTextures: number;
  maxAttributes: number;
  maxVertices: number;
  precision: string;
}

/**
 * Initialization options
 */
export interface InitializationOptions {
  canvas: HTMLCanvasElement;
  scene?: Partial<SceneConfig>;
  performance?: Partial<PerformanceConfig>;
}

/**
 * Object3D with additional properties
 */
export interface EnhancedObject3D extends Object3D {
  userData: {
    id?: string;
    type?: string;
    originalPosition?: Vector3;
    velocity?: Vector3;
    [key: string]: any;
  };
}

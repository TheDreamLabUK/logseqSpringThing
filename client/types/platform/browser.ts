import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import type { BrowserCoreState, Transform, Viewport, SceneConfig, PerformanceConfig } from '../core';
import type { Vector2 } from 'three';

export interface BrowserInitOptions {
  canvas: HTMLCanvasElement;
  scene?: SceneConfig;
  performance?: PerformanceConfig;
  controls?: {
    enableDamping?: boolean;
    dampingFactor?: number;
    enableZoom?: boolean;
    enableRotate?: boolean;
    enablePan?: boolean;
    autoRotate?: boolean;
    autoRotateSpeed?: number;
    minDistance?: number;
    maxDistance?: number;
    minPolarAngle?: number;
    maxPolarAngle?: number;
  };
}

export interface BrowserState extends BrowserCoreState {
  controls: OrbitControls | null;
  viewport: Viewport;
  transform: Transform;
  mousePosition: Vector2;
  touchActive: boolean;
  pointerLocked: boolean;
  config: {
    scene: SceneConfig;
    performance: PerformanceConfig;
  };
}

export interface BrowserPlatform {
  state: BrowserState;
  initialize(options: BrowserInitOptions): Promise<void>;
  dispose(): void;
  render(): void;
  resize(width: number, height: number): void;
  setPixelRatio(ratio: number): void;
  getViewport(): Viewport;
  setTransform(transform: Partial<Transform>): void;
  getTransform(): Transform;
  enableVR(): Promise<void>;
  disableVR(): void;
  isVRSupported(): boolean;
  isVRActive(): boolean;
}

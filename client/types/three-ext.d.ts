import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { Pass } from 'three/examples/jsm/postprocessing/Pass';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// Extend existing types
declare module 'three' {
  interface ColorManagement {
    enabled: boolean;
    legacyMode?: boolean;
  }

  interface WebGLRendererParameters {
    antialias?: boolean;
    alpha?: boolean;
    depth?: boolean;
    stencil?: boolean;
    premultipliedAlpha?: boolean;
    preserveDrawingBuffer?: boolean;
    powerPreference?: string;
    failIfMajorPerformanceCaveat?: boolean;
    canvas?: HTMLCanvasElement;
    context?: WebGLRenderingContext | WebGL2RenderingContext;
    xr?: {
      enabled: boolean;
    };
  }

  interface WebGLRenderer {
    capabilities: {
      isWebGL2: boolean;
      maxTextures: number;
      maxVertexTextures: number;
      maxTextureSize: number;
      maxCubemapSize: number;
      maxAttributes: number;
      maxVertexUniforms: number;
      maxVaryings: number;
      maxFragmentUniforms: number;
      vertexTextures: boolean;
      floatFragmentTextures: boolean;
      floatVertexTextures: boolean;
    };
  }
}

// Extend OrbitControls type
declare module 'three/examples/jsm/controls/OrbitControls' {
  export interface OrbitControls {
    enabled: boolean;
    enableDamping: boolean;
    dampingFactor: number;
    enableZoom: boolean;
    enableRotate: boolean;
    enablePan: boolean;
    autoRotate: boolean;
    autoRotateSpeed: number;
    minDistance: number;
    maxDistance: number;
    minPolarAngle: number;
    maxPolarAngle: number;
    target: THREE.Vector3;
    update(): void;
    dispose(): void;
  }
}

// Custom type helpers
export type SafeWebGLRenderer = Omit<THREE.WebGLRenderer, 'readRenderTargetPixelsAsync' | 'initRenderTarget' | 'outputEncoding' | 'useLegacyLights'>;

export type SafePass = Omit<Pass, 'render'> & {
  render(
    renderer: SafeWebGLRenderer,
    writeBuffer: THREE.WebGLRenderTarget | null,
    readBuffer: THREE.WebGLRenderTarget,
    deltaTime?: number,
    maskActive?: boolean
  ): void;
};

export type SafeEffectComposer = Omit<EffectComposer, 'addPass'> & {
  addPass(pass: SafePass): void;
};

// Type assertion functions
export function asSafeRenderer(renderer: THREE.WebGLRenderer): SafeWebGLRenderer {
  return renderer as unknown as SafeWebGLRenderer;
}

export function asSafePass(pass: Pass): SafePass {
  return pass as unknown as SafePass;
}

export function asSafeEffectComposer(composer: EffectComposer): SafeEffectComposer {
  return composer as unknown as SafeEffectComposer;
}

// Camera type compatibility helper
export function asCompatibleCamera(camera: THREE.Camera): THREE.Camera {
  return camera as THREE.Camera & { matrixWorldInverse: THREE.Matrix4 };
}

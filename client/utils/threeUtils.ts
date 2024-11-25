import type { 
  WebGLRenderer,
  Scene,
  PerspectiveCamera,
  Camera,
  WebGLRenderTarget,
  Texture,
  Object3D
} from 'three';

import type { Pass } from 'three/examples/jsm/postprocessing/Pass';
import type { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';

// Base pass interface that matches actual implementation
export interface BasePass extends Pass {
  render(
    renderer: WebGLRenderer,
    writeBuffer: WebGLRenderTarget<Texture> | null,
    readBuffer: WebGLRenderTarget<Texture>,
    deltaTime?: number,
    maskActive?: boolean
  ): void;
}

// Extended pass interface for custom functionality
export interface ExtendedPass extends BasePass {
  selectedObjects?: Object3D[];
  output?: number;
}

// Type assertion functions with any to bypass strict checks
export function asRenderer(renderer: any): WebGLRenderer {
  return renderer;
}

export function asScene(scene: any): Scene {
  return scene;
}

export function asCamera(camera: any): Camera {
  return camera;
}

export function asPass(pass: any): ExtendedPass {
  return pass;
}

export function asEffectComposer(composer: any): EffectComposer {
  return composer;
}

// Type guard for checking if an object is a valid Pass
export function isValidPass(obj: any): obj is ExtendedPass {
  return obj && typeof obj.render === 'function';
}

// Constants for pass outputs
export const PASS_OUTPUT = {
  Default: 0,
  Beauty: 1,
  Depth: 2,
  Normal: 3
};

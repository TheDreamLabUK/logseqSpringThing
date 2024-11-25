import type { XRCoreState, Transform, Viewport, SceneConfig, PerformanceConfig } from '../core';
import type { Group, Object3D } from 'three';

// WebXR Types (since there's no @types/webxr)
export interface XRSession extends EventTarget {
  requestReferenceSpace(type: XRReferenceSpaceType): Promise<XRReferenceSpace>;
  updateRenderState(state: XRRenderStateInit): Promise<void>;
  requestAnimationFrame(callback: XRFrameRequestCallback): number;
  end(): Promise<void>;
}

export interface XRReferenceSpace {
  getOffsetReferenceSpace(originOffset: XRRigidTransform): XRReferenceSpace;
}

export interface XRWebGLLayer {
  framebuffer: WebGLFramebuffer;
  getViewport(view: XRView): XRViewport;
}

export interface XRRenderStateInit {
  baseLayer?: XRWebGLLayer;
  depthFar?: number;
  depthNear?: number;
  inlineVerticalFieldOfView?: number;
}

export interface XRView {
  eye: 'left' | 'right' | 'none';
  projectionMatrix: Float32Array;
  transform: XRRigidTransform;
}

export interface XRViewport {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface XRRigidTransform {
  position: { x: number; y: number; z: number };
  orientation: { x: number; y: number; z: number; w: number };
  matrix: Float32Array;
}

export type XRFrameRequestCallback = (time: number, frame: XRFrame) => void;
export type XRReferenceSpaceType = 'viewer' | 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';
export type XRSessionMode = 'inline' | 'immersive-vr' | 'immersive-ar';
export type XRHandedness = 'none' | 'left' | 'right';
export type XRTargetRayMode = 'gaze' | 'tracked-pointer' | 'screen';

export interface XRControllerEvent extends Event {
  data?: any;
  target: EventTarget & {
    handedness: XRHandedness;
    targetRayMode: XRTargetRayMode;
  };
}

export interface XRController {
  grip: Group;
  ray: Group;
  hand?: XRHand;
  handedness: XRHandedness;
  targetRayMode: XRTargetRayMode;
  gamepad?: Gamepad;
  controller: Object3D;
  model?: Object3D;
  visible: boolean;
  connected: boolean;
}

export interface XRHand {
  joints: Map<XRHandJoint, XRJointSpace>;
  hand: Object3D;
  model?: Object3D;
  visible: boolean;
  connected: boolean;
}

export type XRHandJoint = 
  | 'wrist'
  | 'thumb-metacarpal'
  | 'thumb-phalanx-proximal'
  | 'thumb-phalanx-distal'
  | 'thumb-tip'
  | 'index-finger-metacarpal'
  | 'index-finger-phalanx-proximal'
  | 'index-finger-phalanx-intermediate'
  | 'index-finger-phalanx-distal'
  | 'index-finger-tip';

export interface XRJointSpace extends XRSpace {
  jointRadius: number;
}

export interface XRSpace {}

export interface QuestInitOptions {
  canvas: HTMLCanvasElement;
  scene?: SceneConfig;
  performance?: PerformanceConfig;
  xr?: {
    referenceSpaceType?: XRReferenceSpaceType;
    sessionMode?: XRSessionMode;
    optionalFeatures?: string[];
    requiredFeatures?: string[];
  };
}

export interface QuestState extends XRCoreState {
  xrSession: XRSession | null;
  xrSpace: XRReferenceSpace | null;
  xrLayer: XRWebGLLayer | null;
  controllers: Map<XRHandedness, XRController>;
  hands: Map<XRHandedness, XRHand>;
  viewport: Viewport;
  transform: Transform;
  config: {
    scene: SceneConfig;
    performance: PerformanceConfig;
    xr: {
      referenceSpaceType: XRReferenceSpaceType;
      sessionMode: XRSessionMode;
      optionalFeatures: string[];
      requiredFeatures: string[];
    };
  };
}

export interface QuestPlatform {
  state: QuestState;
  initialize(options: QuestInitOptions): Promise<void>;
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
  getControllerGrip(handedness: XRHandedness): Group | null;
  getControllerRay(handedness: XRHandedness): Group | null;
  getHand(handedness: XRHandedness): XRHand | null;
  vibrate(handedness: XRHandedness, intensity?: number, duration?: number): void;
}

/**
 * WebXR API type extensions
 */

declare module 'three' {
  export interface Object3DEventMap {
    connected: XRControllerEvent;
    disconnected: XRControllerEvent;
    selectstart: XRControllerEvent;
    selectend: XRControllerEvent;
    squeezestart: XRControllerEvent;
    squeezeend: XRControllerEvent;
    pinchstart: XRHandEvent;
    pinchend: XRHandEvent;
    pinch: XRHandEvent;
    grip: XRHandEvent;
    release: XRHandEvent;
  }

  export interface XRControllerEvent extends Event {
    type: 'connected' | 'disconnected' | 'selectstart' | 'selectend' | 'squeezestart' | 'squeezeend';
    target: Group;
    data?: XRInputSource;
  }

  export interface XRHandEvent extends Event {
    type: 'pinchstart' | 'pinchend' | 'pinch' | 'grip' | 'release';
    target: Group;
    hand: XRHand;
  }

  export interface XRSessionInit {
    optionalFeatures?: string[];
    requiredFeatures?: string[];
  }

  export interface XRReferenceSpace {
    getOffsetReferenceSpace(originOffset: XRRigidTransform): XRReferenceSpace;
  }

  export interface XRRigidTransform {
    position: { x: number; y: number; z: number };
    orientation: { x: number; y: number; z: number; w: number };
    matrix: Float32Array;
    inverse: XRRigidTransform;
  }

  export interface XRFrame {
    session: XRSession;
    getViewerPose(referenceSpace: XRReferenceSpace): XRViewerPose | null;
    getPose(space: XRSpace, baseSpace: XRSpace): XRPose | null;
  }

  export interface XRViewerPose {
    transform: XRRigidTransform;
    views: XRView[];
  }

  export interface XRView {
    eye: 'left' | 'right' | 'none';
    projectionMatrix: Float32Array;
    transform: XRRigidTransform;
    viewport: { x: number; y: number; width: number; height: number };
  }

  export interface XRInputSource {
    handedness: 'none' | 'left' | 'right';
    targetRayMode: 'gaze' | 'tracked-pointer' | 'screen';
    targetRaySpace: XRSpace;
    gripSpace?: XRSpace;
    gamepad?: Gamepad;
    profiles: string[];
    hand?: XRHand;
  }

  export interface XRPose {
    transform: XRRigidTransform;
    emulatedPosition: boolean;
    linearVelocity?: DOMPointReadOnly;
    angularVelocity?: DOMPointReadOnly;
  }

  export interface XRSpace {
    readonly type: string;
    readonly mode: XRSessionMode;
    readonly session: XRSession;
  }

  export type XRReferenceSpaceType = 'viewer' | 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';
  export type XRSessionMode = 'inline' | 'immersive-vr' | 'immersive-ar';

  export interface XRHandJoint {
    readonly jointName: XRHandJointName;
    readonly jointSpace: XRJointSpace;
    readonly radius: number;
    readonly pose: XRJointPose | null;
  }

  export interface XRJointPose extends XRPose {
    readonly radius: number;
  }

  export type XRHandJointName =
    | 'wrist'
    | 'thumb-metacarpal'
    | 'thumb-phalanx-proximal'
    | 'thumb-phalanx-distal'
    | 'thumb-tip'
    | 'index-finger-metacarpal'
    | 'index-finger-phalanx-proximal'
    | 'index-finger-phalanx-intermediate'
    | 'index-finger-phalanx-distal'
    | 'index-finger-tip'
    | 'middle-finger-metacarpal'
    | 'middle-finger-phalanx-proximal'
    | 'middle-finger-phalanx-intermediate'
    | 'middle-finger-phalanx-distal'
    | 'middle-finger-tip'
    | 'ring-finger-metacarpal'
    | 'ring-finger-phalanx-proximal'
    | 'ring-finger-phalanx-intermediate'
    | 'ring-finger-phalanx-distal'
    | 'ring-finger-tip'
    | 'pinky-finger-metacarpal'
    | 'pinky-finger-phalanx-proximal'
    | 'pinky-finger-phalanx-intermediate'
    | 'pinky-finger-phalanx-distal'
    | 'pinky-finger-tip';

  export interface XRLightEstimate {
    primaryLightIntensity?: { value: number };
    primaryLightDirection?: { x: number; y: number; z: number };
  }
}

// Extend existing WebXR types
declare global {
  interface XRFrame {
    session: XRSession;
    getViewerPose(referenceSpace: XRReferenceSpace): XRViewerPose | null;
    getPose(space: XRSpace, baseSpace: XRSpace): XRPose | null;
    getHitTestResults(hitTestSource: XRHitTestSource): XRHitTestResult[];
    getLightEstimate?(): XRLightEstimate | null;
  }

  interface XRSession {
    requestReferenceSpace(type: XRReferenceSpaceType): Promise<XRReferenceSpace>;
    requestHitTestSource(options: XRHitTestOptionsInit): Promise<XRHitTestSource>;
    end(): Promise<void>;
    addEventListener(type: string, listener: EventListener): void;
    removeEventListener(type: string, listener: EventListener): void;
  }

  interface XRHitTestOptionsInit {
    space: XRSpace;
    offsetRay?: XRRay;
  }

  interface XRHitTestSource {
    cancel(): void;
  }

  interface XRHitTestResult {
    getPose(baseSpace: XRSpace): XRPose | null;
  }

  interface XRHand extends Map<XRHandJoint, XRJointSpace> {
    readonly size: number;
    get(joint: XRHandJoint): XRJointSpace | undefined;
  }

  interface XRJointSpace extends XRSpace {
    readonly jointName: XRHandJoint;
  }

  interface Navigator {
    xr?: {
      isSessionSupported(mode: string): Promise<boolean>;
      requestSession(mode: string, options?: XRSessionInit): Promise<XRSession>;
    };
  }
}

// Prevent conflicts with @types/webxr
declare module '@types/webxr' {
  export {};
}

export {};

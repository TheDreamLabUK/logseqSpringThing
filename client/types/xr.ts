import * as THREE from 'three';
import { Platform } from '../core/types';
import { Vector3 } from 'three';

// Core XR Types
export type XRSessionMode = 'immersive-ar' | 'immersive-vr' | 'inline';
export type XRHandedness = 'none' | 'left' | 'right';

export interface XRSystem {
    isSessionSupported(mode: XRSessionMode): Promise<boolean>;
    requestSession(mode: XRSessionMode, options?: XRSessionInit): Promise<XRSession>;
}

export interface XRHand extends THREE.Group {
    type: 'Group';
    joints: Map<string, XRJointSpace>;
    dispatchEvent<K extends keyof THREE.Object3DEventMap>(event: THREE.Object3DEventMap[K]): void;
}

export interface XRJointSpace extends THREE.Object3D<THREE.Object3DEventMap> {
    position: Vector3;
    quaternion: THREE.Quaternion;
    radius: number;
}

export interface CustomXRLightEstimate {
    primaryLightDirection?: { x: number; y: number; z: number };
    primaryLightIntensity: { value: number };
    sphericalHarmonicsCoefficients: Float32Array;
}

export interface XRSessionInit {
    optionalFeatures?: string[];
    requiredFeatures?: string[];
}

export interface XRSession {
    addEventListener(type: string, listener: EventListener): void;
    removeEventListener(type: string, listener: EventListener): void;
    requestReferenceSpace(type: XRReferenceSpaceType): Promise<XRReferenceSpace>;
    updateRenderState(renderState: XRRenderState): Promise<void>;
    requestAnimationFrame(callback: XRFrameRequestCallback): number;
    end(): Promise<void>;
}

export type XRReferenceSpaceType = 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';

export interface XRRenderState {
    baseLayer?: XRWebGLLayer;
    depthFar?: number;
    depthNear?: number;
    inlineVerticalFieldOfView?: number;
}

export interface XRWebGLLayer {
    framebuffer: WebGLFramebuffer;
    framebufferWidth: number;
    framebufferHeight: number;
}

export type XRFrameRequestCallback = (time: number, frame: XRFrame) => void;

export interface XRFrame {
    session: XRSession;
    getViewerPose(referenceSpace: XRReferenceSpace): XRViewerPose | null;
    getLightEstimate?(): CustomXRLightEstimate | null;
}

export interface XRViewerPose {
    transform: XRRigidTransform;
    views: XRView[];
}

export interface XRRigidTransform {
    position: Vector3;
    orientation: THREE.Quaternion;
    matrix: THREE.Matrix4;
}

export interface XRView {
    eye: XREye;
    projectionMatrix: Float32Array;
    transform: XRRigidTransform;
}

export type XREye = 'left' | 'right' | 'none';

// Input and Interaction Types
export interface HapticActuator {
    pulse: (intensity: number, duration: number) => Promise<boolean>;
}

export interface WorldObject3D extends THREE.Object3D {
    position: THREE.Vector3;
    quaternion: THREE.Quaternion;
    scale: THREE.Vector3;
    matrix: THREE.Matrix4;
    matrixWorld: THREE.Matrix4;
    getWorldPosition(target: THREE.Vector3): THREE.Vector3;
}

export interface XRControllerState {
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    hapticActuator?: HapticActuator;
    platform: Platform;
}

export interface XRHandState {
    position: THREE.Vector3;
    joints: Map<string, XRJointSpace>;
    pinchStrength: number;
    gripStrength: number;
    platform: Platform;
}

export interface XRHandWithHaptics extends THREE.Group {
    hapticActuators?: HapticActuator[];
    hand: {
        joints: { [key: string]: WorldObject3D };
    };
    pinchStrength: number;
    gripStrength: number;
    userData: {
        hapticActuator?: HapticActuator;
        platform: Platform;
    };
}

export interface XRHandController {
    hapticActuators?: HapticActuator[];
    hand: {
        joints: { [key: string]: WorldObject3D };
    };
    pinchStrength: number;
}

// Input Configuration
export interface XRInputConfig {
    controllers: boolean;
    hands: boolean;
    haptics: boolean;
}

// Event Types
export interface XRControllerEvent {
    controller: XRSpace;
    inputSource: XRInputSource;
    hapticActuator?: HapticActuator;
}

export interface XRHandEvent {
    hand: XRHandWithHaptics;
    inputSource: XRInputSource;
}

export interface XRInteractionState {
    pinching: boolean;
    pinchStrength: number;
    gripping: boolean;
    gripStrength: number;
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
}

// Platform-specific Types
export interface QuestHandTracking extends XRHandState {
    confidence: number;
    gestureId?: number;
}

export interface QuestControllerTracking extends XRControllerState {
    thumbstick: THREE.Vector2;
    trigger: number;
    grip: number;
}

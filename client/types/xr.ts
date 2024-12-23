import * as THREE from 'three';
import { Platform } from '../core/types';

// Using WebXR types from global declarations

// XR Controller and Hand Types
export interface XRControllerState {
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    hapticActuator?: {
        pulse: (intensity: number, duration: number) => Promise<boolean>;
    };
    platform: Platform;
}

export interface XRHandJointState {
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    radius?: number;
}

export interface XRHandState {
    position: THREE.Vector3;
    joints: Map<XRHandJoint, XRHandJointState>;
    pinchStrength: number;
    gripStrength: number;
    platform: Platform;
}

export interface XRHandWithHaptics extends THREE.Group {
    hapticActuators?: {
        pulse: (intensity: number, duration: number) => Promise<boolean>;
    }[];
    hand: {
        joints: {
            [key in XRHandJoint]?: THREE.Object3D;
        };
    };
    pinchStrength: number;
    gripStrength: number;
    userData: {
        hapticActuator?: {
            pulse: (intensity: number, duration: number) => Promise<boolean>;
        };
        platform: Platform;
    };
}

// XR Session Types
export interface XRSessionConfig {
    mode: XRSessionMode;
    features: {
        required?: string[];
        optional?: string[];
    };
    spaceType: XRReferenceSpaceType;
}

export type XRSessionMode = 'inline' | 'immersive-vr' | 'immersive-ar';

// XR Input Types
export interface XRInputConfig {
    controllers: boolean;
    hands: boolean;
    haptics: boolean;
}

// XR Event Types
export interface XRControllerEvent {
    controller: XRSpace;
    inputSource: XRInputSource;
    hapticActuator?: {
        pulse: (intensity: number, duration: number) => Promise<boolean>;
    };
}

export interface XRHandEvent {
    hand: XRHandWithHaptics;
    inputSource: XRInputSource;
}

// XR Interaction Types
export interface XRInteractionState {
    pinching: boolean;
    pinchStrength: number;
    gripping: boolean;
    gripStrength: number;
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
}

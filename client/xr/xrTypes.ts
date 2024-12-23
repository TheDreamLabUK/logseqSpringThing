import * as THREE from 'three';

export interface XRHandWithHaptics extends THREE.Group {
    hapticActuators?: {
        pulse: (intensity: number, duration: number) => Promise<boolean>;
    }[];
    hand: {
        joints: {
            [key: string]: THREE.Object3D;
        };
    };
    pinchStrength: number;
    gripStrength: number;
    userData: {
        hapticActuator?: {
            pulse: (intensity: number, duration: number) => Promise<boolean>;
        };
    };
}

export interface XRControllerState {
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    hapticActuator?: any;
}

export interface XRHandState {
    position: THREE.Vector3;
    joints: Map<string, THREE.Object3D>;
    pinchStrength: number;
    gripStrength: number;
}

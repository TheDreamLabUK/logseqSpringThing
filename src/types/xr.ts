export type XRHandedness = 'left' | 'right' | 'none';

export interface XRHandJoint {
    position: THREE.BufferAttribute;
    orientation: THREE.BufferAttribute;
    radius: number;
}

export interface XRHand extends THREE.Object3D {
    joints: {
        [key: string]: XRHandJoint;
    };
}

export interface XRInteractionState {
    isHolding: boolean;
    selectedObject: THREE.Object3D | null;
    interactionDistance: number;
    lastPinchTime: number;
}

export interface XRControllerState {
    left: XRInteractionState;
    right: XRInteractionState;
}

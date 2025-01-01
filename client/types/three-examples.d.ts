import { Group, Camera, Scene, WebGLRenderer, Material, Vector3, Color, Object3D } from 'three';

declare module 'three/examples/jsm/webxr/XRControllerModelFactory' {
    export class XRControllerModelFactory {
        constructor();
        createControllerModel(controller: Group): Group;
    }
}

declare module 'three/examples/jsm/controls/OrbitControls' {
    export class OrbitControls {
        constructor(camera: Camera, domElement?: HTMLElement);
        enabled: boolean;
        target: Vector3;
        enableDamping: boolean;
        dampingFactor: number;
        autoRotate: boolean;
        rotateSpeed: number;
        zoomSpeed: number;
        panSpeed: number;
        update(): void;
        dispose(): void;
    }
}

declare module 'three/examples/jsm/postprocessing/EffectComposer' {
    export class EffectComposer {
        constructor(renderer: WebGLRenderer);
        addPass(pass: any): void;
        render(deltaTime?: number): void;
        setSize(width: number, height: number): void;
        dispose(): void;
    }
}

declare module 'three/examples/jsm/postprocessing/RenderPass' {
    export class RenderPass {
        constructor(scene: Scene, camera: Camera);
        enabled: boolean;
    }
}

declare module 'three/examples/jsm/postprocessing/UnrealBloomPass' {
    export class UnrealBloomPass {
        constructor(resolution: Vector3, strength: number, radius: number, threshold: number);
        enabled: boolean;
        strength: number;
        radius: number;
        threshold: number;
        dispose(): void;
    }
}

// Extend existing Three.js types
declare module 'three' {
    export interface Object3D {
        geometry?: any;
        material?: Material | Material[];
    }

    export interface XRHand extends Group {
        joints: Map<string, Object3D>;
    }
}

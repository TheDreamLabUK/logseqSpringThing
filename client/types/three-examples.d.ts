import * as THREE from 'three';

declare module 'three/examples/jsm/controls/OrbitControls' {
    export class OrbitControls {
        constructor(camera: THREE.Camera, domElement: HTMLElement);
        enabled: boolean;
        autoRotate: boolean;
        autoRotateSpeed: number;
        rotateSpeed: number;
        zoomSpeed: number;
        panSpeed: number;
        update(): void;
        dispose(): void;
    }
}

declare module 'three/examples/jsm/postprocessing/EffectComposer' {
    export class EffectComposer {
        constructor(renderer: THREE.WebGLRenderer);
        addPass(pass: any): void;
        render(deltaTime?: number): void;
        dispose(): void;
    }
}

declare module 'three/examples/jsm/postprocessing/RenderPass' {
    export class RenderPass {
        constructor(scene: THREE.Scene, camera: THREE.Camera);
        enabled: boolean;
    }
}

declare module 'three/examples/jsm/postprocessing/UnrealBloomPass' {
    export class UnrealBloomPass {
        constructor(resolution: THREE.Vector2, strength: number, radius: number, threshold: number);
        enabled: boolean;
        strength: number;
        radius: number;
        threshold: number;
    }
}

declare module 'three/examples/jsm/webxr/XRControllerModelFactory' {
    export class XRControllerModelFactory {
        constructor();
        createControllerModel(controller: THREE.Group): THREE.Group;
    }
}

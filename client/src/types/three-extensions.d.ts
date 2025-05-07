// import { Camera, Color, Light, Object3D, PerspectiveCamera, Scene, Vector3, WebGLRenderer } from 'three';
// import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
//
// declare module 'three' {
//     interface Color {
//         setHex(hex: number): this;
//     }
//
//     interface Vector2 {
//         new(x: number, y: number): this;
//     }
//
//     interface Vector3 {
//         new(x: number, y: number, z: number): this;
//         set(x: number, y: number, z: number): this;
//         normalize(): this;
//     }
//
//     interface Object3D {
//         position: Vector3;
//         lookAt(v: Vector3): void;
//         lookAt(x: number, y: number, z: number): void;
//     }
//
//     interface Light extends Object3D {
//         intensity: number;
//         color: Color;
//     }
//
//     interface AmbientLight extends Light {}
//     interface DirectionalLight extends Light {}
//
//     interface PerspectiveCamera extends Camera {
//         fov: number;
//         near: number;
//         far: number;
//         position: Vector3;
//         updateProjectionMatrix(): void;
//     }
//
//     interface WebGLRenderer {
//         xr: {
//             enabled: boolean;
//             setReferenceSpaceType(type: string): void;
//         };
//         setClearColor(color: Color): void;
//     }
// }
//
// declare module '@react-three/fiber' {
//     interface ThreeElements {
//         ambientLight: Object3D;
//         directionalLight: Object3D;
//         perspectiveCamera: PerspectiveCamera;
//     }
// }
//
// declare module 'three/examples/jsm/postprocessing/EffectComposer' {
//     export class EffectComposer {
//         constructor(renderer: WebGLRenderer);
//         addPass(pass: any): void;
//         setSize(width: number, height: number): void;
//         render(): void;
//         dispose(): void;
//     }
// }
//
// declare module 'three/examples/jsm/postprocessing/UnrealBloomPass' {
//     import { Vector2 } from 'three';
//     export class UnrealBloomPass {
//         constructor(resolution: Vector2, strength: number, radius: number, threshold: number);
//     }
// }
//
// declare module 'three/examples/jsm/postprocessing/RenderPass' {
//     import { Scene, Camera } from 'three';
//     export class RenderPass {
//         constructor(scene: Scene, camera: Camera);
//     }
// }
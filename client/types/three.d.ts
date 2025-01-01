// Type definitions for Three.js with WebXR and GPU support
declare module 'three' {
    // Core interfaces
    export interface Object3DEventMap {
        added: { type: 'added'; target: Object3D };
        removed: { type: 'removed'; target: Object3D };
        'matrix-update': { type: 'matrix-update'; target: Object3D };
    }

    export interface MaterialEventMap {
        dispose: { type: 'dispose'; target: Material };
        'shader-update': { type: 'shader-update'; target: Material };
    }

    // Base types
    export type Side = 'FrontSide' | 'BackSide' | 'DoubleSide';
    export type Blending = 'NoBlending' | 'NormalBlending' | 'AdditiveBlending' | 'SubtractiveBlending' | 'MultiplyBlending';
    export type DrawMode = 'TrianglesDrawMode' | 'TriangleStripDrawMode' | 'TriangleFanDrawMode';

    // Core classes with proper type definitions
    export class Object3D {
        readonly isObject3D: true;
        readonly uuid: string;
        name: string;
        type: string;
        parent: Object3D | null;
        children: Object3D[];
        up: Vector3;
        position: Vector3;
        rotation: Euler;
        quaternion: Quaternion;
        scale: Vector3;
        modelViewMatrix: Matrix4;
        normalMatrix: Matrix3;
        matrix: Matrix4;
        matrixWorld: Matrix4;
        matrixAutoUpdate: boolean;
        matrixWorldAutoUpdate: boolean;
        matrixWorldNeedsUpdate: boolean;
        layers: Layers;
        visible: boolean;
        castShadow: boolean;
        receiveShadow: boolean;
        frustumCulled: boolean;
        renderOrder: number;
        animations: AnimationClip[];
        userData: Record<string, unknown>;
        customDepthMaterial?: Material;
        customDistanceMaterial?: Material;

        addEventListener<K extends keyof Object3DEventMap>(
            type: K,
            listener: (event: Object3DEventMap[K]) => void
        ): void;
        removeEventListener<K extends keyof Object3DEventMap>(
            type: K,
            listener: (event: Object3DEventMap[K]) => void
        ): void;
        dispatchEvent<K extends keyof Object3DEventMap>(event: Object3DEventMap[K]): void;
    }

    export class Material {
        transparent: boolean;
        opacity: number;
        depthWrite: boolean;
        depthTest: boolean;
        side: Side;
        color: Color;
        dispose(): void;
        addEventListener<K extends keyof MaterialEventMap>(
            type: K,
            listener: (event: MaterialEventMap[K]) => void
        ): void;
        removeEventListener<K extends keyof MaterialEventMap>(
            type: K,
            listener: (event: MaterialEventMap[K]) => void
        ): void;
        dispatchEvent<K extends keyof MaterialEventMap>(event: MaterialEventMap[K]): void;
    }

    export class BufferGeometry {
        dispose(): void;
        rotateX(angle: number): this;
        rotateY(angle: number): this;
        rotateZ(angle: number): this;
        setAttribute(name: string, attribute: BufferAttribute): this;
        setIndex(index: BufferAttribute): this;
        computeBoundingBox(): void;
        computeBoundingSphere(): void;
    }

    export class Group extends Object3D {
        constructor();
        type: 'Group';
    }

    export class Scene extends Object3D {
        constructor();
        type: 'Scene';
        fog: FogExp2 | null;
        background: Color | Texture | null;
        environment: Texture | null;
        overrideMaterial: Material | null;
    }

    export class Mesh extends Object3D {
        constructor(geometry?: BufferGeometry, material?: Material);
        type: 'Mesh';
        geometry: BufferGeometry;
        material: Material;
    }

    export class Camera extends Object3D {
        type: 'Camera';
        matrixWorldInverse: Matrix4;
        projectionMatrix: Matrix4;
        projectionMatrixInverse: Matrix4;
    }

    export class PerspectiveCamera extends Camera {
        constructor(fov?: number, aspect?: number, near?: number, far?: number);
        type: 'PerspectiveCamera';
        fov: number;
        aspect: number;
        near: number;
        far: number;
        updateProjectionMatrix(): void;
    }

    export class WebGLRenderer {
        constructor(parameters?: WebGLRendererParameters);
        domElement: HTMLCanvasElement;
        capabilities: {
            isWebGL2: boolean;
            maxTextures: number;
            maxVertexTextures: number;
            maxTextureSize: number;
            maxCubemapSize: number;
            maxAttributes: number;
            maxVertexUniforms: number;
            maxVaryings: number;
            maxFragmentUniforms: number;
            vertexTextures: boolean;
            floatFragmentTextures: boolean;
            floatVertexTextures: boolean;
        };
        xr: WebXRManager;
        setSize(width: number, height: number, updateStyle?: boolean): void;
        setPixelRatio(value: number): void;
        render(scene: Scene, camera: Camera): void;
        setAnimationLoop(callback: XRFrameRequestCallback | null): void;
        dispose(): void;
    }

    export interface WebXRManager {
        enabled: boolean;
        isPresenting: boolean;
        getController(index: number): Group;
        getControllerGrip(index: number): Group;
        setFramebufferScaleFactor(value: number): void;
        setReferenceSpaceType(value: XRReferenceSpaceType): void;
        getReferenceSpace(): XRReferenceSpace | null;
        getSession(): XRSession | null;
        setSession(value: XRSession): Promise<void>;
        getCamera(): Camera;
        updateCamera(camera: Camera): void;
    }

    // Utility types
    export type XRFrameRequestCallback = (time: number, frame?: XRFrame) => void;
    export type ColorRepresentation = Color | string | number;
    
    // Constants
    export const DoubleSide: Side;
    export const FrontSide: Side;
    export const BackSide: Side;
    
    export const NoBlending: Blending;
    export const NormalBlending: Blending;
    export const AdditiveBlending: Blending;
    export const SubtractiveBlending: Blending;
    export const MultiplyBlending: Blending;
}

// Prevent accidental use of ambient declarations
export {};

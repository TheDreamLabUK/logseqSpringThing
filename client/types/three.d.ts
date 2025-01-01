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

        add(...objects: Object3D[]): this;
        remove(...objects: Object3D[]): this;
        traverse(callback: (object: Object3D) => void): void;
        updateMatrix(): void;
        updateMatrixWorld(force?: boolean): void;
        lookAt(x: number | Vector3, y?: number, z?: number): void;
        rotateX(angle: number): this;
        rotateY(angle: number): this;
        rotateZ(angle: number): this;

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
        wireframe?: boolean;
        uniforms?: { [uniform: string]: { value: any } };
        defines?: { [key: string]: string };
        needsUpdate: boolean;

        dispose(): void;
        clone(): this;
        copy(source: Material): this;

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

    export class MeshBasicMaterial extends Material {
        color: Color;
        map: Texture | null;
        wireframe: boolean;
        wireframeLinewidth: number;
        fog: boolean;
        lights: boolean;
        isMeshBasicMaterial: true;
    }

    export class MeshPhongMaterial extends Material {
        color: Color;
        specular: Color;
        shininess: number;
        map: Texture | null;
        lightMap: Texture | null;
        lightMapIntensity: number;
        aoMap: Texture | null;
        aoMapIntensity: number;
        emissive: Color;
        emissiveIntensity: number;
        emissiveMap: Texture | null;
        bumpMap: Texture | null;
        bumpScale: number;
        normalMap: Texture | null;
        normalMapType: number;
        normalScale: Vector2;
        displacementMap: Texture | null;
        displacementScale: number;
        displacementBias: number;
        specularMap: Texture | null;
        alphaMap: Texture | null;
        envMap: Texture | null;
        combine: number;
        reflectivity: number;
        refractionRatio: number;
        wireframe: boolean;
        wireframeLinewidth: number;
        wireframeLinecap: string;
        wireframeLinejoin: string;
        fog: boolean;
        lights: boolean;
        isMeshPhongMaterial: true;
    }

    export class LineBasicMaterial extends Material {
        color: Color;
        linewidth: number;
        linecap: string;
        linejoin: string;
        isLineBasicMaterial: true;
    }

    export class ShaderMaterial extends Material {
        uniforms: { [uniform: string]: { value: any } };
        vertexShader: string;
        fragmentShader: string;
        linewidth: number;
        wireframe: boolean;
        wireframeLinewidth: number;
        lights: boolean;
        clipping: boolean;
        skinning: boolean;
        morphTargets: boolean;
        morphNormals: boolean;
        extensions: {
            derivatives: boolean;
            fragDepth: boolean;
            drawBuffers: boolean;
            shaderTextureLOD: boolean;
        };
        defaultAttributeValues: any;
        index0AttributeName: string | undefined;
        uniformsNeedUpdate: boolean;
        glslVersion: any;
        isShaderMaterial: true;
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
        isGroup: true;
    }

    export class Scene extends Object3D {
        constructor();
        type: 'Scene';
        fog: FogExp2 | null;
        background: Color | Texture | null;
        environment: Texture | null;
        isScene: true;
    }

    export class Mesh extends Object3D {
        constructor(geometry?: BufferGeometry, material?: Material);
        type: 'Mesh';
        geometry: BufferGeometry;
        material: Material;
        isMesh: true;
    }

    export class Camera extends Object3D {
        type: 'Camera';
        matrixWorldInverse: Matrix4;
        projectionMatrix: Matrix4;
        projectionMatrixInverse: Matrix4;
        isCamera: true;
        updateMatrixWorld(force?: boolean): void;
        lookAt(x: number | Vector3, y?: number, z?: number): void;
    }

    export class PerspectiveCamera extends Camera {
        constructor(fov?: number, aspect?: number, near?: number, far?: number);
        type: 'PerspectiveCamera';
        fov: number;
        aspect: number;
        near: number;
        far: number;
        isPerspectiveCamera: true;
        updateProjectionMatrix(): void;
    }

    export class Vector3 {
        constructor(x?: number, y?: number, z?: number);
        x: number;
        y: number;
        z: number;
        isVector3: true;
        
        set(x: number, y: number, z: number): this;
        setScalar(scalar: number): this;
        setX(x: number): this;
        setY(y: number): this;
        setZ(z: number): this;
        
        clone(): Vector3;
        copy(v: Vector3): this;
        add(v: Vector3): this;
        addScalar(s: number): this;
        addVectors(a: Vector3, b: Vector3): this;
        sub(v: Vector3): this;
        subScalar(s: number): this;
        subVectors(a: Vector3, b: Vector3): this;
        multiply(v: Vector3): this;
        multiplyScalar(scalar: number): this;
        multiplyVectors(a: Vector3, b: Vector3): this;
        divide(v: Vector3): this;
        divideScalar(scalar: number): this;
        min(v: Vector3): this;
        max(v: Vector3): this;
        clamp(min: Vector3, max: Vector3): this;
        length(): number;
        lengthSq(): number;
        manhattanLength(): number;
        normalize(): this;
        setLength(length: number): this;
        lerp(v: Vector3, alpha: number): this;
        lerpVectors(v1: Vector3, v2: Vector3, alpha: number): this;
        cross(v: Vector3): this;
        crossVectors(a: Vector3, b: Vector3): this;
        projectOnVector(v: Vector3): this;
        projectOnPlane(planeNormal: Vector3): this;
        reflect(normal: Vector3): this;
        angleTo(v: Vector3): number;
        distanceTo(v: Vector3): number;
        distanceToSquared(v: Vector3): number;
        manhattanDistanceTo(v: Vector3): number;
        setFromSpherical(s: Spherical): this;
        setFromSphericalCoords(radius: number, phi: number, theta: number): this;
        setFromCylindrical(c: Cylindrical): this;
        setFromCylindricalCoords(radius: number, theta: number, y: number): this;
        setFromMatrixPosition(m: Matrix4): this;
        setFromMatrixScale(m: Matrix4): this;
        setFromMatrixColumn(matrix: Matrix4, index: number): this;
        equals(v: Vector3): boolean;
        fromArray(array: number[], offset?: number): this;
        toArray(array?: number[], offset?: number): number[];
        fromBufferAttribute(attribute: BufferAttribute, index: number): this;
        random(): this;
    }

    export interface WebGLRendererParameters {
        canvas?: HTMLCanvasElement;
        context?: WebGLRenderingContext;
        precision?: string;
        alpha?: boolean;
        premultipliedAlpha?: boolean;
        antialias?: boolean;
        stencil?: boolean;
        preserveDrawingBuffer?: boolean;
        powerPreference?: string;
        failIfMajorPerformanceCaveat?: boolean;
        depth?: boolean;
        logarithmicDepthBuffer?: boolean;
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
        setClearColor(color: Color | string | number, alpha?: number): void;
        render(scene: Scene, camera: Camera): void;
        setAnimationLoop(callback: XRFrameRequestCallback | null): void;
        dispose(): void;
    }

    export class WebXRManager {
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
    export const TrianglesDrawMode: DrawMode;
    export const TriangleStripDrawMode: DrawMode;
    export const TriangleFanDrawMode: DrawMode;
}

// Prevent accidental use of ambient declarations
export {};

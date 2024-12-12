/**
 * THREE.js type declarations
 */

declare module 'three' {
  export interface Event {
    type: string;
    target: any;
  }

  // XR Controller Event Types
  export interface XRControllerEvent extends Event {
    type: 'connected' | 'disconnected';
    data: XRInputSource;
  }

  export interface Object3DEventMap {
    connected: XRControllerEvent;
    disconnected: XRControllerEvent;
  }

  export interface EventDispatcher<E extends Event = Event> {
    addEventListener<T extends E['type']>(type: T, listener: (event: E & { type: T }) => void): void;
    removeEventListener<T extends E['type']>(type: T, listener: (event: E & { type: T }) => void): void;
    dispatchEvent(event: E): void;
  }

  export class Object3D implements EventDispatcher<Event & XRControllerEvent> {
    position: Vector3;
    quaternion: Quaternion;
    scale: Vector3;
    matrix: Matrix4;
    matrixWorld: Matrix4;
    children: Object3D[];
    parent: Object3D | null;
    userData: any;
    visible: boolean;
    renderOrder: number;
    frustumCulled: boolean;
    matrixAutoUpdate: boolean;
    add(...objects: Object3D[]): this;
    remove(...objects: Object3D[]): this;
    rotateX(angle: number): this;
    rotateY(angle: number): this;
    rotateZ(angle: number): this;
    updateMatrix(): void;
    updateMatrixWorld(force?: boolean): void;
    traverse(callback: (object: Object3D) => void): void;
    lookAt(x: number | Vector3, y?: number, z?: number): void;
    addEventListener<K extends keyof Object3DEventMap>(
      type: K,
      listener: (event: Object3DEventMap[K]) => void
    ): void;
    addEventListener(
      type: string,
      listener: (event: Event) => void
    ): void;
    removeEventListener<K extends keyof Object3DEventMap>(
      type: K,
      listener: (event: Object3DEventMap[K]) => void
    ): void;
    removeEventListener(
      type: string,
      listener: (event: Event) => void
    ): void;
    dispatchEvent(event: Event): void;
  }

  export class Scene extends Object3D {
    constructor();
    fog: FogExp2 | null;
    background: Color | Texture | null;
  }

  export class Group extends Object3D {
    constructor();
  }

  export class Mesh extends Object3D {
    constructor(geometry: BufferGeometry, material: Material);
    geometry: BufferGeometry;
    material: Material;
  }

  export class InstancedMesh extends Mesh {
    constructor(geometry: BufferGeometry, material: Material, count: number);
    count: number;
    instanceMatrix: BufferAttribute;
    instanceColor: BufferAttribute | null;
    setMatrixAt(index: number, matrix: Matrix4): void;
    setColorAt(index: number, color: Color): void;
  }

  export class Sprite extends Object3D {
    constructor(material: SpriteMaterial);
    material: SpriteMaterial;
  }

  export class BufferGeometry {
    dispose(): void;
    rotateX(angle: number): this;
    rotateY(angle: number): this;
    rotateZ(angle: number): this;
  }

  export class Material {
    transparent: boolean;
    opacity: number;
    depthWrite: boolean;
    depthTest: boolean;
    side: Side;
    color: Color;
    dispose(): void;
  }

  export interface MaterialParameters {
    color?: ColorRepresentation;
    transparent?: boolean;
    opacity?: number;
    side?: Side;
    depthWrite?: boolean;
    depthTest?: boolean;
    map?: Texture;
  }

  export interface MeshBasicMaterialParameters extends MaterialParameters {}
  export interface MeshPhongMaterialParameters extends MaterialParameters {
    shininess?: number;
    specular?: ColorRepresentation;
  }
  export interface SpriteMaterialParameters extends MaterialParameters {
    depthTest?: boolean;
  }

  export class MeshBasicMaterial extends Material {
    constructor(parameters?: MeshBasicMaterialParameters);
  }

  export class MeshPhongMaterial extends Material {
    constructor(parameters?: MeshPhongMaterialParameters);
    shininess: number;
    specular: Color;
  }

  export class SpriteMaterial extends Material {
    constructor(parameters?: SpriteMaterialParameters);
    map: Texture | null;
  }

  export class BufferAttribute {
    array: ArrayLike<number>;
    needsUpdate: boolean;
  }

  export class GridHelper extends Object3D {
    constructor(size: number, divisions: number, color1?: ColorRepresentation, color2?: ColorRepresentation);
    material: Material;
    geometry: BufferGeometry;
  }

  export class DirectionalLight extends Object3D {
    constructor(color?: ColorRepresentation, intensity?: number);
    intensity: number;
  }

  export class AmbientLight extends Light {
    constructor(color?: ColorRepresentation, intensity?: number);
  }

  export class Light extends Object3D {
    constructor(color?: ColorRepresentation, intensity?: number);
    intensity: number;
  }

  export class Vector2 {
    x: number;
    y: number;
    constructor(x?: number, y?: number);
    set(x: number, y: number): this;
  }

  export class Vector3 {
    x: number;
    y: number;
    z: number;
    constructor(x?: number, y?: number, z?: number);
    set(x: number, y: number, z: number): this;
    copy(v: Vector3): this;
    add(v: Vector3): this;
    sub(v: Vector3): this;
    multiply(v: Vector3): this;
    multiplyScalar(s: number): this;
    normalize(): this;
    dot(v: Vector3): number;
    cross(v: Vector3): this;
    length(): number;
    lengthSq(): number;
    clone(): Vector3;
    fromArray(array: number[] | ArrayLike<number>, offset?: number): this;
    subVectors(a: Vector3, b: Vector3): this;
    addVectors(a: Vector3, b: Vector3): this;
    crossVectors(a: Vector3, b: Vector3): this;
    setFromMatrixPosition(m: Matrix4): this;
    distanceTo(v: Vector3): number;
    applyMatrix4(m: Matrix4): this;
    lookAt(v: Vector3): this;
  }

  export class Matrix4 {
    elements: number[];
    constructor();
    set(...elements: number[]): this;
    identity(): this;
    copy(m: Matrix4): this;
    compose(position: Vector3, quaternion: Quaternion, scale: Vector3): this;
    decompose(position: Vector3, quaternion: Quaternion, scale: Vector3): this;
    fromArray(array: ArrayLike<number>, offset?: number): this;
    extractRotation(m: Matrix4): this;
    makeRotationFromQuaternion(q: Quaternion): this;
  }

  export class Quaternion {
    x: number;
    y: number;
    z: number;
    w: number;
    constructor(x?: number, y?: number, z?: number, w?: number);
    setFromAxisAngle(axis: Vector3, angle: number): this;
    identity(): this;
  }

  export class Color {
    constructor(color?: ColorRepresentation);
    set(color: ColorRepresentation): this;
  }

  export class SphereGeometry extends BufferGeometry {
    constructor(radius?: number, widthSegments?: number, heightSegments?: number);
  }

  export class PlaneGeometry extends BufferGeometry {
    constructor(width?: number, height?: number, widthSegments?: number, heightSegments?: number);
  }

  export class CylinderGeometry extends BufferGeometry {
    constructor(
      radiusTop?: number,
      radiusBottom?: number,
      height?: number,
      radialSegments?: number
    );
  }

  export class RingGeometry extends BufferGeometry {
    constructor(
      innerRadius?: number,
      outerRadius?: number,
      thetaSegments?: number
    );
  }

  export class Texture {
    constructor(image?: HTMLImageElement | HTMLCanvasElement);
    needsUpdate: boolean;
    dispose(): void;
  }

  export class WebGLRenderer {
    constructor(parameters?: WebGLRendererParameters);
    domElement: HTMLCanvasElement;
    setSize(width: number, height: number, updateStyle?: boolean): void;
    setPixelRatio(value: number): void;
    render(scene: Scene, camera: Camera): void;
    dispose(): void;
    xr: WebXRManager;
  }

  export interface WebGLRendererParameters {
    canvas?: HTMLCanvasElement;
    antialias?: boolean;
    alpha?: boolean;
    powerPreference?: 'default' | 'high-performance' | 'low-power';
  }

  export class WebXRManager {
    enabled: boolean;
    setSession(session: XRSession): Promise<void>;
  }

  export class Camera extends Object3D {
    matrixWorldInverse: Matrix4;
    projectionMatrix: Matrix4;
    projectionMatrixInverse: Matrix4;
    lookAt(target: Vector3 | number, y?: number, z?: number): void;
  }

  export class PerspectiveCamera extends Camera {
    constructor(fov?: number, aspect?: number, near?: number, far?: number);
    fov: number;
    aspect: number;
    near: number;
    far: number;
    updateProjectionMatrix(): void;
    lookAt(target: Vector3 | number, y?: number, z?: number): void;
  }

  export class Raycaster {
    constructor();
    ray: Ray;
    near: number;
    far: number;
    setFromCamera(coords: Vector2, camera: Camera): void;
    intersectObject(object: Object3D, recursive?: boolean): Intersection[];
    intersectObjects(objects: Object3D[], recursive?: boolean): Intersection[];
  }

  export class Ray {
    origin: Vector3;
    direction: Vector3;
  }

  export interface Intersection {
    distance: number;
    point: Vector3;
    object: Object3D;
  }

  export class FogExp2 {
    constructor(color: ColorRepresentation, density?: number);
    color: Color;
    density: number;
  }

  export const DoubleSide: Side;
  export type Side = 0 | 1 | 2;
  export type ColorRepresentation = Color | string | number;

  export class MathUtils {
    static clamp(value: number, min: number, max: number): number;
  }
}

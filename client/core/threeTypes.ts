import * as THREE from 'three';
import { OrbitControls as ThreeOrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Re-export Three.js types
export type Scene = THREE.Scene;
export type Camera = THREE.Camera;
export type PerspectiveCamera = THREE.PerspectiveCamera;
export type Material = THREE.Material;
export type MeshBasicMaterial = THREE.MeshBasicMaterial;
export type MeshPhongMaterial = THREE.MeshPhongMaterial;
export type MeshStandardMaterial = THREE.MeshStandardMaterial;
export type Mesh = THREE.Mesh;
export type Group = THREE.Group;
export type Vector3 = THREE.Vector3;
export type Vector2 = THREE.Vector2;
export type Color = THREE.Color;
export type BufferGeometry = THREE.BufferGeometry;
export type Object3D = THREE.Object3D;
export type Line = THREE.Line;
export type LineBasicMaterial = THREE.LineBasicMaterial;
export type LineBasicMaterialParameters = THREE.LineBasicMaterialParameters;
export type GridHelper = THREE.GridHelper;
export type AmbientLight = THREE.AmbientLight;
export type DirectionalLight = THREE.DirectionalLight;
export type OrbitControls = ThreeOrbitControls;
export type BufferAttribute = THREE.BufferAttribute;
export type InterleavedBufferAttribute = THREE.InterleavedBufferAttribute;
export type WebGLRenderer = THREE.WebGLRenderer;
export type Clock = THREE.Clock;
export type Raycaster = THREE.Raycaster;
export type Intersection = THREE.Intersection;
export type InstancedMesh = THREE.InstancedMesh;
export type Box3 = THREE.Box3;
export type Sphere = THREE.Sphere;
export type EulerOrder = THREE.EulerOrder;
export type MaterialParameters = THREE.MaterialParameters;
export type MeshBasicMaterialParameters = THREE.MeshBasicMaterialParameters;
export type MeshPhongMaterialParameters = THREE.MeshPhongMaterialParameters;
export type MeshStandardMaterialParameters = THREE.MeshStandardMaterialParameters;
export type Side = THREE.Side;

// Export constants and classes
export const DoubleSide = THREE.DoubleSide;
export const Euler = THREE.Euler;
export const Quaternion = THREE.Quaternion;
export const Matrix4 = THREE.Matrix4;

// Factory functions with proper type handling
export function createLine(geometry: THREE.BufferGeometry, material: THREE.LineBasicMaterial): THREE.Line {
    // Workaround for Three.js 0.171.0 constructor type issues
    const line = new THREE.Line();
    (line as any).geometry = geometry;
    (line as any).material = material;
    return line;
}

export function createMeshBasicMaterial(params: THREE.MeshBasicMaterialParameters = {}): THREE.MeshBasicMaterial {
    // Workaround for Three.js 0.171.0 constructor type issues
    const material = new THREE.MeshBasicMaterial();
    return Object.assign(material, params);
}

export function createMeshPhongMaterial(params: THREE.MeshPhongMaterialParameters = {}): THREE.MeshPhongMaterial {
    // Workaround for Three.js 0.171.0 constructor type issues
    const material = new THREE.MeshPhongMaterial();
    return Object.assign(material, params);
}

export function createMeshStandardMaterial(params: THREE.MeshStandardMaterialParameters = {}): THREE.MeshStandardMaterial {
    // Workaround for Three.js 0.171.0 constructor type issues
    const material = new THREE.MeshStandardMaterial();
    return Object.assign(material, params);
}

export function createLineBasicMaterial(params: THREE.LineBasicMaterialParameters = {}): THREE.LineBasicMaterial {
    // Workaround for Three.js 0.171.0 constructor type issues
    const material = new THREE.LineBasicMaterial();
    return Object.assign(material, params);
}

export function createMesh(geometry: THREE.BufferGeometry, material: THREE.Material): THREE.Mesh {
    // Workaround for Three.js 0.171.0 constructor type issues
    const mesh = new THREE.Mesh();
    (mesh as any).geometry = geometry;
    (mesh as any).material = material;
    return mesh;
}

export function createBufferAttribute(array: Float32Array | number[], itemSize: number): THREE.BufferAttribute {
    return new THREE.BufferAttribute(array instanceof Float32Array ? array : new Float32Array(array), itemSize);
}

export function createVector3(x = 0, y = 0, z = 0): THREE.Vector3 {
    return new THREE.Vector3(x, y, z);
}

export function createVector2(x = 0, y = 0): THREE.Vector2 {
    return new THREE.Vector2(x, y);
}

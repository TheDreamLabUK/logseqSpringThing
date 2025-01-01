import * as THREE from 'three';

// Re-export basic Three.js types with proper type definitions
export type Scene = THREE.Scene;
export type Camera = THREE.Camera;
export type PerspectiveCamera = THREE.PerspectiveCamera;
export type Material = THREE.Material;
export type MeshBasicMaterial = THREE.MeshBasicMaterial;
export type MeshPhongMaterial = THREE.MeshPhongMaterial;
export type MeshStandardMaterial = THREE.MeshStandardMaterial;
export type Mesh<TGeometry extends THREE.BufferGeometry = THREE.BufferGeometry, 
                TMaterial extends THREE.Material | THREE.Material[] = THREE.Material> = THREE.Mesh<TGeometry, TMaterial>;
export type Group = THREE.Group;
export type Vector3 = THREE.Vector3;
export type Vector2 = THREE.Vector2;
export type Color = THREE.Color;
export type BufferGeometry = THREE.BufferGeometry;
export type Object3D = THREE.Object3D;
export type Side = THREE.Side;
export type Line = THREE.Line;
export type GridHelper = THREE.GridHelper;
export type AmbientLight = THREE.AmbientLight;
export type DirectionalLight = THREE.DirectionalLight;
export type OrbitControls = THREE.OrbitControls;
export type BufferAttribute = THREE.BufferAttribute;
export type InterleavedBufferAttribute = THREE.InterleavedBufferAttribute;

// Constants
export const DoubleSide = THREE.DoubleSide;

// Material Parameters Types
export type MaterialParameters = THREE.MaterialParameters;
export type MeshBasicMaterialParameters = THREE.MeshBasicMaterialParameters;
export type MeshPhongMaterialParameters = THREE.MeshPhongMaterialParameters;
export type MeshStandardMaterialParameters = THREE.MeshStandardMaterialParameters;

// Factory functions for creating materials with proper typing
export function createMeshBasicMaterial(params?: MeshBasicMaterialParameters): MeshBasicMaterial {
    return new THREE.MeshBasicMaterial(params ?? {});
}

export function createMeshPhongMaterial(params?: MeshPhongMaterialParameters): MeshPhongMaterial {
    return new THREE.MeshPhongMaterial(params ?? {});
}

export function createMeshStandardMaterial(params?: MeshStandardMaterialParameters): MeshStandardMaterial {
    return new THREE.MeshStandardMaterial(params ?? {});
}

// Factory function for creating meshes with proper generic typing
export function createMesh<TGeometry extends THREE.BufferGeometry = THREE.BufferGeometry,
                         TMaterial extends THREE.Material | THREE.Material[] = THREE.Material>(
    geometry: TGeometry,
    material: TMaterial
): Mesh<TGeometry, TMaterial> {
    return new THREE.Mesh(geometry, material);
}

// Other Three.js exports
export const Euler = THREE.Euler;
export type EulerOrder = THREE.EulerOrder;
export const Quaternion = THREE.Quaternion;
export const Matrix4 = THREE.Matrix4;

// Buffer Geometry helpers
export function createBufferAttribute(array: Float32Array | number[], itemSize: number): BufferAttribute {
    return new THREE.BufferAttribute(array instanceof Float32Array ? array : new Float32Array(array), itemSize);
}

// Vector helpers
export function createVector3(x = 0, y = 0, z = 0): Vector3 {
    return new THREE.Vector3(x, y, z);
}

export function createVector2(x = 0, y = 0): Vector2 {
    return new THREE.Vector2(x, y);
}

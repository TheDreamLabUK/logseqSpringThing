import * as THREE from 'three';

export interface NodeMetadata extends Record<string, unknown> {
    label?: string;
    type?: string;
}

export interface NodeData {
    id: string;
    position: THREE.Vector3;
    color?: THREE.Color;
    size?: number;
    metadata?: NodeMetadata;
    properties?: Record<string, unknown>;
}

export interface EdgeMetadata extends Record<string, unknown> {
    label?: string;
    type?: string;
}

export interface EdgeData {
    id: string;
    source: string;
    target: string;
    color?: THREE.Color;
    width?: number;
    metadata?: EdgeMetadata;
    properties?: Record<string, unknown>;
}

export interface NodeUserData {
    id: string;
    type: string;
    properties?: Record<string, unknown>;
    [key: string]: unknown;
}

export interface EdgeUserData {
    id: string;
    source: string;
    target: string;
    type: string;
    properties?: Record<string, unknown>;
    [key: string]: unknown;
}

// Define mesh types using Three.js base types
export interface NodeMesh extends THREE.Object3D {
    isMesh: true;
    geometry: THREE.BufferGeometry;
    material: THREE.Material;
    userData: NodeUserData;
}

export interface EdgeMesh extends THREE.Object3D {
    isLine: true;
    geometry: THREE.BufferGeometry;
    material: THREE.Material;
    userData: EdgeUserData;
}

// Type guard to check if an object is a NodeMesh
export function isNodeMesh(object: THREE.Object3D): object is NodeMesh {
    return object instanceof THREE.Mesh &&
           object.userData &&
           typeof object.userData === 'object' &&
           'id' in object.userData &&
           'type' in object.userData;
}

// Type guard to check if an object is an EdgeMesh
export function isEdgeMesh(object: THREE.Object3D): object is EdgeMesh {
    return object instanceof THREE.Line &&
           object.userData &&
           typeof object.userData === 'object' &&
           'id' in object.userData &&
           'source' in object.userData &&
           'target' in object.userData;
}

// Factory function to create a NodeMesh
export function createNodeMesh(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    userData: NodeUserData
): NodeMesh {
    // Create mesh with double type assertion
    const mesh = new THREE.Mesh(geometry as any, material as any);
    mesh.userData = userData;
    // Use double type assertion for safer casting
    return mesh as unknown as NodeMesh;
}

// Factory function to create an EdgeMesh
export function createEdgeMesh(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    userData: EdgeUserData
): EdgeMesh {
    // Create line with double type assertion
    const edge = new THREE.Line(geometry as any, material as any);
    edge.userData = userData;
    // Use double type assertion for safer casting
    return edge as unknown as EdgeMesh;
}

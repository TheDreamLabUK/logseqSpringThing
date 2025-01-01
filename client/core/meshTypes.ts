import {
    Vector3,
    Color,
    Material,
    BufferGeometry,
    Object3D,
    Mesh,
    Line
} from './threeTypes';

export interface NodeMetadata extends Record<string, unknown> {
    label?: string;
    type?: string;
}

export interface NodeData {
    id: string;
    position: Vector3;
    color?: Color;
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
    color?: Color;
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

export interface NodeMesh extends Mesh {
    userData: NodeUserData;
}

export interface EdgeMesh extends Line {
    userData: EdgeUserData;
}

// Type guard to check if an object is a NodeMesh
export function isNodeMesh(object: Object3D): object is NodeMesh {
    return 'userData' in object && 
           'id' in object.userData && 
           'type' in object.userData &&
           object instanceof Mesh;
}

// Type guard to check if an object is an EdgeMesh
export function isEdgeMesh(object: Object3D): object is EdgeMesh {
    return 'userData' in object && 
           'id' in object.userData && 
           'source' in object.userData && 
           'target' in object.userData &&
           object instanceof Line;
}

// Factory function to create a NodeMesh
export function createNodeMesh(
    geometry: BufferGeometry,
    material: Material,
    userData: NodeUserData
): NodeMesh {
    const mesh = new Mesh(geometry, material) as NodeMesh;
    mesh.userData = userData;
    return mesh;
}

// Factory function to create an EdgeMesh
export function createEdgeMesh(
    geometry: BufferGeometry,
    material: Material,
    userData: EdgeUserData
): EdgeMesh {
    const edge = new Line(geometry, material) as EdgeMesh;
    edge.userData = userData;
    return edge;
}

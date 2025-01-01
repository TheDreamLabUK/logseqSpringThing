import {
    Vector3,
    Color,
    Material,
    Mesh,
    BufferGeometry,
    Object3D,
    Scene,
    PerspectiveCamera as ThreePerspectiveCamera,
    WebGLRenderer,
    IUniform,
    Side,
    ShaderMaterial,
    Camera as ThreeCamera,
    MeshBasicMaterial,
    MeshPhongMaterial,
    DoubleSide,
    Group
} from 'three';

// Re-export Three.js types with our own names to avoid conflicts
export type PerspectiveCamera = ThreePerspectiveCamera;
export type Camera = ThreeCamera;

// Core types for the application
export interface GraphNode {
    id: string;
    label: string;
    position: Vector3;
    color?: Color;
    size?: number;
    group?: string;
    properties: Record<string, unknown>;
}

export interface GraphEdge {
    source: string;
    target: string;
    weight?: number;
    type?: string;
    properties: Record<string, unknown>;
}

export interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
    metadata?: Record<string, unknown>;
}

export type MessageType = 
    | 'GRAPH_UPDATE'
    | 'NODE_SELECT'
    | 'EDGE_SELECT'
    | 'CAMERA_MOVE'
    | 'ERROR'
    | 'INFO';

export interface WebSocketMessage<T = unknown> {
    type: MessageType;
    data: T;
    timestamp: number;
}

export interface SimulationMessage extends WebSocketMessage {
    type: 'GRAPH_UPDATE';
    data: {
        nodes: GraphNode[];
        edges: GraphEdge[];
        timestamp: number;
    };
}

export interface SelectionMessage extends WebSocketMessage {
    type: 'NODE_SELECT' | 'EDGE_SELECT';
    data: {
        id: string;
        selected: boolean;
    };
}

export interface CameraMessage extends WebSocketMessage {
    type: 'CAMERA_MOVE';
    data: {
        position: Vector3;
        target: Vector3;
    };
}

export interface ErrorMessage extends WebSocketMessage {
    type: 'ERROR';
    data: {
        code: string;
        message: string;
    };
}

export interface InfoMessage extends WebSocketMessage {
    type: 'INFO';
    data: {
        message: string;
    };
}

export type GraphEventHandler<T = unknown> = (message: WebSocketMessage<T>) => void;

export interface GraphEventMap {
    'graph-update': SimulationMessage;
    'node-select': SelectionMessage;
    'edge-select': SelectionMessage;
    'camera-move': CameraMessage;
    'error': ErrorMessage;
    'info': InfoMessage;
}

export interface GraphEventTarget {
    addEventListener<K extends keyof GraphEventMap>(
        type: K,
        listener: (event: GraphEventMap[K]) => void
    ): void;
    removeEventListener<K extends keyof GraphEventMap>(
        type: K,
        listener: (event: GraphEventMap[K]) => void
    ): void;
    dispatchEvent<K extends keyof GraphEventMap>(event: GraphEventMap[K]): void;
}

// Node types
export interface NodeMetadata extends Record<string, unknown> {
    label?: string;
    type?: string;
}

export interface NodeData {
    id: string;
    name: string;
    position: Vector3;
    color?: Color;
    size?: number;
    metadata?: NodeMetadata;
    properties?: Record<string, unknown>;
}

export interface NodeUserData extends Record<string, unknown> {
    id: string;
    type: 'node';
    data: NodeData;
}

export interface NodeMesh extends Mesh {
    userData: NodeUserData;
}

// Edge types
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

export interface EdgeUserData extends Record<string, unknown> {
    id: string;
    type: 'edge';
    data: EdgeData;
}

export interface EdgeMesh extends Mesh {
    userData: EdgeUserData;
}

// Hologram types
export interface HologramUniforms extends Record<string, IUniform> {
    time: IUniform<number>;
    opacity: IUniform<number>;
    color: IUniform<number>;
    glowIntensity: IUniform<number>;
}

export interface HologramShaderMaterial extends ShaderMaterial {
    uniforms: HologramUniforms;
    update(deltaTime: number): void;
    handleInteraction(intensity: number): void;
}

// Scene types
export interface SceneManager {
    scene: Scene;
    camera: PerspectiveCamera;
    renderer: WebGLRenderer;
    init(): void;
    update(deltaTime: number): void;
    resize(width: number, height: number): void;
    dispose(): void;
}

// Visualization types
export interface MaterialSettings {
    type?: 'basic' | 'phong' | 'hologram';
    color?: Color;
    opacity?: number;
    transparent?: boolean;
    side?: Side;
    wireframe?: boolean;
    glowIntensity?: number;
    emissive?: Color;
    shininess?: number;
}

export interface NodeSettings {
    defaultColor: Color;
    material?: MaterialSettings;
    size?: number;
    selected?: {
        color: Color;
        scale: number;
    };
}

export interface EdgeSettings {
    defaultColor: Color;
    material?: MaterialSettings;
    width?: number;
    selected?: {
        color: Color;
        width: number;
    };
}

export interface HologramSettings {
    color: Color;
    opacity: number;
    glowIntensity: number;
    ringCount: number;
    ringSizes: number[];
    rotationSpeed: number;
    enableBuckminster: boolean;
    buckminsterScale: number;
    buckminsterOpacity: number;
    enableGeodesic: boolean;
    geodesicScale: number;
    geodesicOpacity: number;
    enableTriangleSphere: boolean;
    triangleSphereScale: number;
    triangleSphereOpacity: number;
}

export interface VisualizationSettings {
    nodes: NodeSettings;
    edges: EdgeSettings;
    hologram: HologramSettings;
    camera?: {
        position: Vector3;
        target: Vector3;
        fov: number;
    };
}

// Export utility functions
export function createNodeMesh(geometry: BufferGeometry, material: Material, userData: NodeUserData): NodeMesh {
    const mesh = new Mesh(geometry, material);
    mesh.userData = userData;
    return mesh as NodeMesh;
}

export function createEdgeMesh(geometry: BufferGeometry, material: Material, userData: EdgeUserData): EdgeMesh {
    const mesh = new Mesh(geometry, material);
    mesh.userData = userData;
    return mesh as EdgeMesh;
}

export function isNodeMesh(object: Object3D): object is NodeMesh {
    return 'userData' in object && 'type' in object.userData && object.userData.type === 'node';
}

export function isEdgeMesh(object: Object3D): object is EdgeMesh {
    return 'userData' in object && 'type' in object.userData && object.userData.type === 'edge';
}

export function transformGraphData(data: GraphData, transformer: GraphDataTransformer): GraphData {
    return transformer.transform(data);
}

// Error types
export class GraphError extends Error {
    constructor(
        message: string,
        public code: string,
        public details?: Record<string, unknown>
    ) {
        super(message);
        this.name = 'GraphError';
    }
}

export class WebSocketError extends Error {
    constructor(
        message: string,
        public code: string,
        public details?: Record<string, unknown>
    ) {
        super(message);
        this.name = 'WebSocketError';
    }
}

// Logger types
export interface Logger {
    debug(message: string, ...args: unknown[]): void;
    info(message: string, ...args: unknown[]): void;
    warn(message: string, ...args: unknown[]): void;
    error(message: string, ...args: unknown[]): void;
    log(message: string, ...args: unknown[]): void;
}

export type ErrorCallback = (error: Error) => void;
export type SuccessCallback = () => void;

// Utility types
export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Mutable<T> = {
    -readonly [P in keyof T]: T[P];
};

export enum PlatformType {
    Desktop = 'desktop',
    Mobile = 'mobile',
    VR = 'vr',
    AR = 'ar'
}

export interface PlatformCapabilities {
    platform: PlatformType;
    xrSupported: boolean;
    touchSupported: boolean;
    mouseSupported: boolean;
    keyboardSupported: boolean;
}

export interface GraphDataTransformer {
    transform(data: GraphData): GraphData;
}

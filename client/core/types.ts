import {
    Scene,
    Camera,
    WebGLRenderer,
    Object3D,
    Vector3,
    Color,
    PerspectiveCamera
} from 'three';
import * as THREE from 'three';

// Re-export Three.js types with our own names to avoid conflicts
export {
    Scene,
    Camera,
    WebGLRenderer,
    Object3D,
    Vector3,
    Color,
    PerspectiveCamera
};

// Core types for the application
export interface Node {
    id: string;
    label: string;
    position: Vector3;
    color?: Color;
    size?: number;
    group?: string;
    properties: Record<string, unknown>;
    data?: {
        position: { x: number; y: number; z: number };
        type?: string;
    };
}

export interface Edge {
    source: string;
    target: string;
    weight?: number;
    type?: string;
    properties: Record<string, unknown>;
}

export interface GraphData {
    nodes: Node[];
    edges: Edge[];
    metadata?: Record<string, unknown>;
}

// Hologram types
export interface HologramUniforms extends Record<string, THREE.IUniform<any>> {
    time: THREE.IUniform<number>;
    opacity: THREE.IUniform<number>;
    color: THREE.IUniform<THREE.Color>;
    glowIntensity: THREE.IUniform<number>;
}

export interface HologramShaderMaterial extends THREE.ShaderMaterial {
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

// Material types
export interface MaterialSettings {
    type: 'basic' | 'phong' | 'hologram';
    color?: THREE.Color;
    transparent?: boolean;
    opacity?: number;
    side?: THREE.Side;
    glowIntensity?: number;
}

// Platform types
export interface PlatformCapabilities {
    webgl: {
        isSupported: boolean;
        version: number;
    };
    xr: {
        isSupported: boolean;
        isImmersiveSupported: boolean;
    };
}

export interface Platform {
    name: string;
    version: string;
    capabilities: PlatformCapabilities;
}

// Transform function
export interface GraphDataTransformer {
    transform(data: GraphData): GraphData;
}

export function transformGraphData(data: GraphData, transformer: GraphDataTransformer): GraphData {
    return transformer.transform(data);
}

// Logger types
export type LogLevel = 'error' | 'warn' | 'info' | 'debug' | 'trace';

export interface LoggerOptions {
    namespace?: string;
    level?: LogLevel;
    enableJsonFormatting?: boolean;
}

export interface Logger {
    debug(message: string, ...args: unknown[]): void;
    info(message: string, ...args: unknown[]): void;
    warn(message: string, ...args: unknown[]): void;
    error(message: string, ...args: unknown[]): void;
    trace(message: string, ...args: unknown[]): void;
    log(level: LogLevel, message: string, ...args: unknown[]): void;
    setLevel(level: LogLevel): void;
    getLevel(): LogLevel;
    setJsonFormatting(enabled: boolean): void;
}

// Node types
export interface NodeData {
    id: string;
    label: string;
    position: Vector3;
    color?: Color;
    size?: number;
    type?: string;
    properties: Record<string, unknown>;
}

export interface NodeMesh extends THREE.Mesh {
    userData: {
        id: string;
        type?: string;
        properties?: Record<string, unknown>;
        rotationSpeed?: number;
    };
}

// Visualization settings
export interface NodeSettings {
    color: string;
    defaultSize: number;
    minSize: number;
    maxSize: number;
    sizeProperty?: string;
    colorProperty?: string;
    baseColor: string;
    baseSize: number;
    sizeRange: [number, number];
    enableMetadataShape: boolean;
    colorRangeAge: [string, string];
    colorRangeLinks: [string, string];
    metalness: number;
    roughness: number;
    opacity: number;
    enableMetadataVisualization: boolean;
    enableHologram: boolean;
    enableInstancing: boolean;
    quality: 'low' | 'medium' | 'high';
    material: {
        type: 'basic' | 'phong';
        transparent: boolean;
        opacity: number;
    };
}

export interface EdgeSettings {
    color: string;
    defaultWidth: number;
    minWidth: number;
    maxWidth: number;
    widthProperty?: string;
    colorProperty?: string;
    arrowSize: number;
    baseWidth: number;
    enableArrows: boolean;
    opacity: number;
    widthRange: [number, number];
}

export interface HologramSettings {
    color: string;
    opacity: number;
    glowIntensity: number;
    rotationSpeed: number;
    enabled: boolean;
    ringCount: number;
    ringColor: string;
    ringOpacity: number;
    ringSizes: [number, number, number];
    ringRotationSpeed: number;
    enableBuckminster: boolean;
    buckminsterScale: number;
    buckminsterOpacity: number;
    enableGeodesic: boolean;
    geodesicScale: number;
    geodesicOpacity: number;
    enableTriangleSphere: boolean;
    triangleSphereScale: number;
    triangleSphereOpacity: number;
    globalRotationSpeed: number;
}

export interface VisualizationSettings {
    nodes: NodeSettings;
    edges: EdgeSettings;
    hologram: HologramSettings;
    labels?: {
        enabled: boolean;
        size: number;
        color: string;
    };
    render?: {
        showGrid: boolean;
        backgroundColor: string;
    };
    controls?: {
        autoRotate: boolean;
        rotateSpeed: number;
        zoomSpeed: number;
        panSpeed: number;
    };
}

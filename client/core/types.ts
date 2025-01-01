import {
    Vector3,
    Color,
    Scene,
    PerspectiveCamera as ThreePerspectiveCamera,
    WebGLRenderer,
    IUniform,
    Side,
    ShaderMaterial,
    Camera as ThreeCamera
} from 'three';
import * as THREE from 'three';

// Re-export Three.js types with our own names to avoid conflicts
export type PerspectiveCamera = ThreePerspectiveCamera;
export type Camera = ThreeCamera;

// Core types for the application
export interface Node {
    id: string;
    label: string;
    position: Vector3;
    color?: Color;
    size?: number;
    group?: string;
    properties: Record<string, unknown>;
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

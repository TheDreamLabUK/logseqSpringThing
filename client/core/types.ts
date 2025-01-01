import { Vector3, Color } from 'three';

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

// Settings types
export interface RenderSettings {
    nodeSize: number;
    edgeWidth: number;
    fontSize: number;
    backgroundColor: Color;
    defaultNodeColor: Color;
    defaultEdgeColor: Color;
    highlightColor: Color;
}

export interface PhysicsSettings {
    gravity: number;
    springLength: number;
    springStrength: number;
    repulsion: number;
    damping: number;
}

export interface ControlSettings {
    rotateSpeed: number;
    zoomSpeed: number;
    panSpeed: number;
    autoRotate: boolean;
    enableVR: boolean;
}

export interface Settings {
    render: RenderSettings;
    physics: PhysicsSettings;
    controls: ControlSettings;
}

// Platform-specific types
export interface Platform {
    name: string;
    version: string;
    capabilities: {
        webgl2: boolean;
        webxr: boolean;
        webgpu: boolean;
    };
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

// Utility types
export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Mutable<T> = {
    -readonly [P in keyof T]: T[P];
};

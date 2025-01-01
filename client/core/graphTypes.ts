import { Vector3, Color } from 'three';

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

export interface GraphDataTransformer {
    transform(data: GraphData): GraphData;
}

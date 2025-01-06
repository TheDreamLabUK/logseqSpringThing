import { Side } from 'three';

export interface Settings {
    visualization: {
        hologram: {
            enabled: boolean;
            color: string;
            opacity: number;
            glowIntensity: number;
            rotationSpeed: number;
            ringCount: number;
            ringColor: string;
            ringOpacity: number;
            ringSizes: number[];
            ringRotationSpeed: number;
        };
        nodes: {
            color: string;
            opacity: number;
            defaultSize: number;
            enableInstancing: boolean;
        };
        edges: {
            color: string;
            defaultWidth: number;
            opacity: number;
        };
    };
    nodes: {
        material: NodeMaterialSettings;
    };
    xr?: {
        quality: 'low' | 'medium' | 'high';
    };
}

export interface NodeMaterialSettings {
    type: 'basic' | 'phong' | 'hologram';
    color?: number;
    transparent?: boolean;
    opacity?: number;
    side?: Side;
    glowIntensity?: number;
} 
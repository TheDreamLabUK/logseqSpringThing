import { Side } from 'three';

export interface Settings {
    visualization: {
        hologram: {
            ringCount: number;
            ringSizes: number[];
            ringRotationSpeed: number;
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
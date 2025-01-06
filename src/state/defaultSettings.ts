import { Settings } from '../types/settings';
import { DoubleSide } from 'three';

export const defaultSettings: Settings = {
    visualization: {
        hologram: {
            enabled: true,
            color: '#00ff00',
            opacity: 0.5,
            glowIntensity: 0.8,
            rotationSpeed: 0.5,
            ringCount: 3,
            ringColor: '#00ff00',
            ringOpacity: 0.5,
            ringSizes: [1, 1.5, 2],
            ringRotationSpeed: 0.5
        },
        nodes: {
            color: '#ffffff',
            opacity: 1.0,
            defaultSize: 1.0,
            enableInstancing: true
        },
        edges: {
            color: '#ffffff',
            defaultWidth: 1.0,
            opacity: 1.0
        }
    },
    nodes: {
        material: {
            type: 'phong',
            color: 0xffffff,
            transparent: false,
            opacity: 1.0,
            side: DoubleSide
        }
    },
    xr: {
        quality: 'medium'
    }
}; 
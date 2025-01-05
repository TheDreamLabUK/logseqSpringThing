import { Settings } from '../types/settings';
import { DoubleSide } from 'three';

export const defaultSettings: Settings = {
    visualization: {
        hologram: {
            ringCount: 3,
            ringSizes: [20, 30, 40],
            ringRotationSpeed: 0.01
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
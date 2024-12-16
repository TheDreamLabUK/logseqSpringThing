import type { Settings } from '../types/settings';

export const defaultSettings: Settings = {
    nodes: {
        baseSize: 1.0,
        baseColor: '#4a90e2',
        opacity: 0.8,
        shape: 'sphere',
        clearcoat: 1.0,
        enableHoverEffect: true,
        highlightColor: '#ff0000',
        highlightScale: 1.2,
        outlineWidth: 1.0,
        outlineColor: '#ffffff',
        metalness: 0.5,
        roughness: 0.5
    },
    edges: {
        baseWidth: 1.0,
        baseColor: '#ffffff',
        opacity: 0.8,
        arrowScale: 1.0,
        enableArrows: true
    },
    rendering: {
        bloomEnabled: true,
        bloomStrength: 1.0,
        bloomThreshold: 0.5,
        bloomRadius: 1.0,
        fov: 75,
        near: 0.1,
        far: 1000
    },
    physics: {
        gravity: 0.0,
        friction: 0.1,
        springStrength: 0.1,
        springLength: 100,
        damping: 0.5,
        attractionStrength: 1.0,
        repulsionStrength: 1.0,
        enabled: true
    },
    labels: {
        desktopFontSize: 14,
        textColor: '#ffffff',
        offset: 1.5,
        maxVisible: 50,
        minScale: 0.5,
        maxScale: 2.0,
        enableLabels: true
    },
    bloom: {
        edgeBloomStrength: 0.5
    },
    clientDebug: {
        enabled: false,
        showFPS: true,
        showStats: true
    }
};

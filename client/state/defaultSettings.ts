import { Settings, VisualizationSettings } from '../types/settings';
import { NODE_COLOR, NODE_SIZE, LABEL_COLOR } from '../core/constants';

// Helper function to convert number to hex color
function toHexColor(num: number): string {
    return `#${num.toString(16).padStart(6, '0')}`;
}

// Export visualization defaults separately for reuse
export const defaultVisualizationSettings: VisualizationSettings = {
    animations: {
        enableMotionBlur: false,
        enableNodeAnimations: false,
        motionBlurStrength: 0.0,
        selectionWaveEnabled: false,
        pulseEnabled: false,
        pulseSpeed: 0.0,
        pulseStrength: 0.0,
        waveSpeed: 0.0
    },
    bloom: {
        enabled: false,
        strength: 0.0,
        radius: 0.0,
        threshold: 1.0,
        edgeBloomStrength: 0.0,
        nodeBloomStrength: 0.0,
        environmentBloomStrength: 0.0
    },
    edges: {
        arrowSize: 0.15,
        baseWidth: 2.0,
        color: '#917f18',
        enableArrows: false,
        opacity: 0.6,
        widthRange: [1.0, 3.0]
    },
    hologram: {
        ringCount: 0,
        ringColor: '#00ff00',
        ringOpacity: 0.0,
        ringSizes: [1.0, 1.2, 1.5],
        ringRotationSpeed: 0.0,
        enableBuckminster: false,
        buckminsterScale: 1.0,
        buckminsterOpacity: 0.0,
        enableGeodesic: false,
        geodesicScale: 1.0,
        geodesicOpacity: 0.0,
        enableTriangleSphere: false,
        triangleSphereScale: 1.0,
        triangleSphereOpacity: 0.0,
        globalRotationSpeed: 0.0
    },
    labels: {
        enableLabels: true,
        textColor: toHexColor(LABEL_COLOR),
        textOutlineColor: '#000000',
        textOutlineWidth: 0.1,
        textResolution: 512,
        textPadding: 16,
        desktopFontSize: 48,
        billboardMode: true
    },
    nodes: {
        baseColor: toHexColor(NODE_COLOR),
        baseSize: NODE_SIZE,
        sizeRange: [0.5, 1.0],
        enableMetadataShape: false,
        colorRangeAge: ['#ffffff', '#ffffff'],
        colorRangeLinks: ['#ffffff', '#ffffff'],
        metalness: 0.0,
        roughness: 0.5,
        opacity: 1.0,
        enableMetadataVisualization: false,
        enableHologram: false,
        enableInstancing: false,
        quality: 'low'
    },
    physics: {
        enabled: true,
        attractionStrength: 0.015,
        repulsionStrength: 1500.0,
        springStrength: 0.018,
        damping: 0.88,
        iterations: 500,
        maxVelocity: 2.5,
        collisionRadius: 0.25,
        enableBounds: true,
        boundsSize: 12.0
    },
    rendering: {
        ambientLightIntensity: 0.3,
        directionalLightIntensity: 1.0,
        environmentIntensity: 0.6,
        backgroundColor: '#000000',
        enableAmbientOcclusion: false,
        enableAntialiasing: false,
        enableShadows: false
    }
};

// Main settings object with all defaults
export const defaultSettings: Settings = {
    visualization: defaultVisualizationSettings,
    system: {
        websocket: {
            reconnectAttempts: 5,
            reconnectDelay: 5000,
            binaryChunkSize: 65536,
            compressionEnabled: true,
            compressionThreshold: 1024,
            updateRate: 60
        },
        debug: {
            enabled: true,
            enableDataDebug: true,
            enableWebsocketDebug: true,
            logBinaryHeaders: true,
            logFullJson: true
        }
    },
    xr: {
        mode: 'immersive-ar' as const,
        quality: 'high',
        roomScale: true,
        spaceType: 'local-floor',
        enableHandTracking: true,
        handMeshEnabled: true,
        handMeshColor: '#ffffff',
        handMeshOpacity: 0.5,
        handPointSize: 5,
        handRayEnabled: true,
        handRayColor: '#ffffff',
        handRayWidth: 2,
        gestureSsmoothing: 0.5,
        enableHaptics: true,
        hapticIntensity: 0.5,
        dragThreshold: 0.1,
        pinchThreshold: 0.5,
        rotationThreshold: 0.1,
        interactionRadius: 0.1,
        // Platform settings
        autoEnterAR: true,
        hideControlPanel: true,
        preferredMode: 'immersive-ar',
        // Scene understanding
        enableLightEstimation: true,
        enablePlaneDetection: true,
        enableSceneUnderstanding: true,
        planeColor: '#808080',
        planeOpacity: 0.5,
        showPlaneOverlay: true,
        snapToFloor: true
    }
};

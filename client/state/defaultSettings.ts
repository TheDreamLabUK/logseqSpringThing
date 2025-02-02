import { Settings, VisualizationSettings } from '../types/settings';
import { LABEL_COLOR } from '../core/constants';

// Helper function to convert number to hex color
function toHexColor(num: number): string {
    return `#${num.toString(16).padStart(6, '0')}`;
}

// Export visualization defaults separately for reuse
export const defaultVisualizationSettings: VisualizationSettings = {
    nodes: {
        quality: 'high',
        enableInstancing: true,
        enableMetadataShape: true,
        enableMetadataVisualization: true,
        baseSize: 1.5,
        sizeRange: [1.0, 3.0],
        baseColor: toHexColor(0x00ffff),
        opacity: 1,
        colorRangeAge: [toHexColor(0x00ffff), toHexColor(0xff00ff)],
        colorRangeLinks: [toHexColor(0x00ffff), toHexColor(0xff00ff)],
        metalness: 0.8,
        roughness: 0.2,
        enableHologram: false
    },
    edges: {
        color: toHexColor(0x888888),
        opacity: 0.6,
        arrowSize: 0,
        baseWidth: 1,
        enableArrows: false,
        widthRange: [1, 2]
    },
    animations: {
        enableNodeAnimations: true,
        enableMotionBlur: false,
        motionBlurStrength: 0.5,
        selectionWaveEnabled: true,
        pulseEnabled: true,
        pulseSpeed: 1.5,
        pulseStrength: 1,
        waveSpeed: 1
    },
    labels: {
        enableLabels: true,
        desktopFontSize: 16,
        textColor: toHexColor(LABEL_COLOR),
        textOutlineColor: toHexColor(0x000000),
        textOutlineWidth: 2,
        textResolution: 256,
        textPadding: 4,
        billboardMode: true
    },
    rendering: {
        ambientLightIntensity: 0.5,
        directionalLightIntensity: 1.5,
        environmentIntensity: 1,
        backgroundColor: toHexColor(0x121212),
        enableAmbientOcclusion: true,
        enableAntialiasing: true,
        enableShadows: true
    },
    bloom: {
        enabled: true,
        strength: 1.5,
        radius: 0.8,
        threshold: 0.3,
        edgeBloomStrength: 0.5,
        nodeBloomStrength: 2,
        environmentBloomStrength: 1
    },
    hologram: {
        ringCount: 3,
        ringSizes: [1, 1.5, 2],
        ringRotationSpeed: 0.1,
        globalRotationSpeed: 0.1,
        ringColor: toHexColor(0x00ffff),
        ringOpacity: 0.5,
        enableBuckminster: false,
        buckminsterScale: 1,
        buckminsterOpacity: 0.5,
        enableGeodesic: false,
        geodesicScale: 1,
        geodesicOpacity: 0.5,
        enableTriangleSphere: false,
        triangleSphereScale: 1,
        triangleSphereOpacity: 0.5
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
    }
};

// Main settings object with all defaults
export const defaultSettings: Settings = {
    visualization: defaultVisualizationSettings,
    system: {
        websocket: {
            reconnectAttempts: 5,
            reconnectDelay: 5000,
            binaryChunkSize: 1024,
            compressionEnabled: true,
            compressionThreshold: 1024,
            updateRate: 60
        },
        debug: {
            enabled: false,
            enableDataDebug: false,
            enableWebsocketDebug: false,
            logBinaryHeaders: false,
            logFullJson: false,
            logLevel: 'info',
            logFormat: 'json'  // Added missing field
        }
    },
    xr: {
        mode: 'immersive-ar' as const,
        quality: 'high',
        roomScale: true,
        spaceType: 'local-floor',
        enableHandTracking: true,
        handMeshEnabled: true,
        handMeshColor: toHexColor(0xffffff),
        handMeshOpacity: 0.5,
        handPointSize: 5,
        handRayEnabled: true,
        handRayColor: toHexColor(0xffffff),
        handRayWidth: 2,
        gestureSmoothing: 0.5,
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
        planeColor: toHexColor(0x808080),
        planeOpacity: 0.5,
        showPlaneOverlay: true,
        snapToFloor: true
    }
}

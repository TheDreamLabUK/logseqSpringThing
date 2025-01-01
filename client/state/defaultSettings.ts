import { Settings, VisualizationSettings } from '../types/settings';
import { NODE_COLOR, NODE_SIZE, EDGE_RADIUS, LABEL_COLOR } from '../core/constants';

// Helper function to convert number to hex color
function toHexColor(num: number): string {
    return `#${num.toString(16).padStart(6, '0')}`;
}

// Export visualization defaults separately for reuse
export const defaultVisualizationSettings: VisualizationSettings = {
    animations: {
        enableMotionBlur: false,
        enableNodeAnimations: true,
        motionBlurStrength: 0.5,
        selectionWaveEnabled: false,
        pulseEnabled: false,
        pulseSpeed: 1.0,
        pulseStrength: 0.5,
        waveSpeed: 1.0
    },
    bloom: {
        enabled: true,
        strength: 0.5,
        radius: 1,
        edgeBloomStrength: 0.5,
        nodeBloomStrength: 0.5,
        environmentBloomStrength: 0.5
    },
    edges: {
        color: '#666666',
        defaultWidth: 1,
        minWidth: 0.5,
        maxWidth: 3,
        arrowSize: 0.2,
        baseWidth: 1,
        enableArrows: true,
        opacity: 0.8,
        widthRange: [0.5, 3]
    },
    hologram: {
        color: '#00ff00',
        opacity: 0.5,
        glowIntensity: 0.8,
        rotationSpeed: 0.5,
        enabled: true,
        ringCount: 3,
        ringColor: '#00ff00',
        ringOpacity: 0.5,
        ringSizes: [1, 1.5, 2],
        ringRotationSpeed: 0.5,
        enableBuckminster: true,
        buckminsterScale: 1,
        buckminsterOpacity: 0.5,
        enableGeodesic: true,
        geodesicScale: 1,
        geodesicOpacity: 0.5,
        enableTriangleSphere: true,
        triangleSphereScale: 1,
        triangleSphereOpacity: 0.5,
        globalRotationSpeed: 0.2
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
        color: '#ffffff',
        defaultSize: 1,
        minSize: 0.5,
        maxSize: 2,
        baseColor: '#ffffff',
        baseSize: 1,
        sizeRange: [0.5, 2],
        enableMetadataShape: true,
        colorRangeAge: ['#ff0000', '#00ff00'],
        colorRangeLinks: ['#0000ff', '#ff00ff'],
        metalness: 0.5,
        roughness: 0.5,
        opacity: 1,
        enableMetadataVisualization: true,
        enableHologram: true,
        enableInstancing: true,
        quality: 'medium'
    },
    physics: {
        enabled: true,
        attractionStrength: 0.1,
        repulsionStrength: 0.1,
        springStrength: 0.1,
        damping: 0.5,
        iterations: 1,
        maxVelocity: 10,
        collisionRadius: 1,
        enableBounds: true,
        boundsSize: 100
    },
    rendering: {
        ambientLightIntensity: 0.5,
        directionalLightIntensity: 0.8,
        environmentIntensity: 1,
        backgroundColor: '#000000',
        enableAmbientOcclusion: true,
        enableAntialiasing: true,
        enableShadows: true
    }
};

// Main settings object with all defaults
export const defaultSettings: Settings = {
    visualization: defaultVisualizationSettings,
    xr: {
        mode: 'ar',
        roomScale: true,
        spaceType: 'local',
        quality: 'high',
        input: 'hands',
        visuals: {
            handMeshEnabled: true,
            handMeshColor: '#ffffff',
            handMeshOpacity: 0.5,
            handPointSize: 5,
            handRayEnabled: true,
            handRayColor: '#00ff00',
            handRayWidth: 2,
            gestureSsmoothing: 0.5
        },
        environment: {
            enableLightEstimation: true,
            enablePlaneDetection: true,
            enableSceneUnderstanding: true,
            planeColor: '#808080',
            planeOpacity: 0.5,
            showPlaneOverlay: true,
            snapToFloor: true
        },
        passthrough: false,
        haptics: true
    },
    system: {
        network: {
            bindAddress: '127.0.0.1',
            domain: 'localhost',
            port: 3000,
            enableHttp2: true,
            enableTls: false,
            minTlsVersion: 'TLS1.2',
            maxRequestSize: 10485760,
            enableRateLimiting: true,
            rateLimitRequests: 100,
            rateLimitWindow: 60,
            tunnelId: ''
        },
        websocket: {
            url: '',
            reconnectAttempts: 5,
            reconnectDelay: 5000,
            binaryChunkSize: 65536,
            compressionEnabled: true,
            compressionThreshold: 1024,
            maxConnections: 100,
            maxMessageSize: 32 * 1024 * 1024,
            updateRate: 60
        },
        security: {
            allowedOrigins: ['http://localhost:3000'],
            auditLogPath: './audit.log',
            cookieHttponly: true,
            cookieSamesite: 'Lax',
            cookieSecure: false,
            csrfTokenTimeout: 3600,
            enableAuditLogging: true,
            enableRequestValidation: true,
            sessionTimeout: 86400
        },
        debug: {
            enabled: true,
            enableDataDebug: true,
            enableWebsocketDebug: true,
            logBinaryHeaders: true,
            logFullJson: true,
            logLevel: 'info'
        }
    }
};

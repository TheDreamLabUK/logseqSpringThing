import { Settings, VisualizationSettings } from '../types/settings';
import { NODE_COLOR, NODE_SIZE, EDGE_RADIUS, LABEL_COLOR } from '../core/constants';

// Export visualization defaults separately for reuse
export const defaultVisualizationSettings: VisualizationSettings = {
    animations: {
        enableMotionBlur: false,
        enableNodeAnimations: true,
        motionBlurStrength: 0.5,
        selectionWaveEnabled: false,
        pulseEnabled: false,
        rippleEnabled: false,
        edgeAnimationEnabled: false,
        flowParticlesEnabled: false
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
        arrowSize: 3,
        baseWidth: EDGE_RADIUS * 2,
        color: '#ffffff',
        enableArrows: true,
        opacity: 0.8,
        widthRange: [1, 5]
    },
    hologram: {
        ringCount: 3,
        ringColor: '#00FFFF',
        ringOpacity: 0.5,
        ringSizes: [1.0, 1.5, 2.0],
        ringRotationSpeed: 0.1,
        enableBuckminster: true,
        buckminsterScale: 1.0,
        buckminsterOpacity: 0.3,
        enableGeodesic: true,
        geodesicScale: 1.2,
        geodesicOpacity: 0.4,
        enableTriangleSphere: true,
        triangleSphereScale: 1.1,
        triangleSphereOpacity: 0.35,
        globalRotationSpeed: 0.05
    },
    labels: {
        enableLabels: true,
        textColor: LABEL_COLOR.toString(16),
        textOutlineColor: '#000000',
        textOutlineWidth: 0.1,
        textResolution: 512,
        textPadding: 16,
        desktopFontSize: 48,
        billboardMode: 'camera'
    },
    nodes: {
        baseColor: NODE_COLOR.toString(16),
        baseSize: NODE_SIZE,
        sizeRange: [0.5, 2.0],
        enableMetadataShape: true,
        colorRangeAge: ['#ff0000', '#00ff00'],
        colorRangeLinks: ['#0000ff', '#ff00ff'],
        metalness: 0.5,
        roughness: 0.2,
        opacity: 0.8,
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
    labels: defaultVisualizationSettings.labels, // Add top-level labels property
    xr: {
        mode: 'immersive-ar',
        roomScale: true,
        spaceType: 'local-floor',
        quality: 'medium',
        input: {
            enableHandTracking: true,
            enableHaptics: true,
            hapticIntensity: 0.5,
            dragThreshold: 0.02,
            pinchThreshold: 0.7,
            rotationThreshold: 0.1,
            interactionRadius: 0.5
        },
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
        passthrough: {
            enabled: false,
            opacity: 1,
            brightness: 1,
            contrast: 1,
            portalSize: 2,
            portalEdgeColor: '#ffffff',
            portalEdgeWidth: 2
        }
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
            url: 'wss://www.visionflow.info/wss',
            heartbeatInterval: 30000,
            heartbeatTimeout: 60000,
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
            enabled: false,
            enableDataDebug: false,
            enableWebsocketDebug: false,
            logBinaryHeaders: false,
            logFullJson: false
        }
    }
};

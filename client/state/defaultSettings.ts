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
        radius: 0,
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
        ringCount: 3,
        ringColor: '#00ff00',
        ringOpacity: 0.5,
        ringSizes: [20.0, 25.0, 30.0],
        ringRotationSpeed: 0.001,
        enableBuckminster: false,
        buckminsterScale: 15.0,
        buckminsterOpacity: 0.3,
        enableGeodesic: false,
        geodesicScale: 15.0,
        geodesicOpacity: 0.3,
        enableTriangleSphere: false,
        triangleSphereScale: 15.0,
        triangleSphereOpacity: 0.3,
        globalRotationSpeed: 0.0005
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
        sizeRange: [1.0, 1.0],
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
            logFullJson: true
        }
    },
    xr: {
        quality: 'medium'
    }
};

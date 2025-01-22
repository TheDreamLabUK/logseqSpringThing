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
        arrowSize: 3,
        baseWidth: EDGE_RADIUS * 2,
        color: '#ffffff',
        enableArrows: true,
        opacity: 0.8,
        widthRange: [1, 5]
    },
    hologram: {
        ringCount: 3,
        ringColor: '#00ffff',
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

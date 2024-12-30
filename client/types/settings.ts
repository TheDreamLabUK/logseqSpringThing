import { XRSessionMode } from './xr';

// Core visualization settings
export interface VisualizationSettings {
    animations: AnimationSettings;
    bloom: BloomSettings;
    edges: EdgeSettings;
    hologram: HologramSettings;
    labels: LabelSettings;
    nodes: NodeSettings;
    physics: PhysicsSettings;
    rendering: RenderingSettings;
}

// XR-specific settings
export interface XRSettings {
    // Session settings
    mode: 'ar' | 'vr';
    roomScale: boolean;
    spaceType: 'local' | 'bounded' | 'unbounded';
    quality: 'low' | 'medium' | 'high';

    // Input and interaction
    input: 'none' | 'controllers' | 'hands';

    // Visual settings
    visuals: {
        handMeshEnabled: boolean;
        handMeshColor: string;
        handMeshOpacity: number;
        handPointSize: number;
        handRayEnabled: boolean;
        handRayColor: string;
        handRayWidth: number;
        gestureSsmoothing: number;
    };

    // Environment settings
    environment: {
        enableLightEstimation: boolean;
        enablePlaneDetection: boolean;
        enableSceneUnderstanding: boolean;
        planeColor: string;
        planeOpacity: number;
        showPlaneOverlay: boolean;
        snapToFloor: boolean;
    };

    // Passthrough settings
    passthrough: boolean;

    // Haptics settings
    haptics: boolean;
}

// System settings
export interface SystemSettings {
    network: NetworkSettings;
    websocket: WebSocketSettings;
    security: SecuritySettings;
    debug: DebugSettings;
}

// Component settings interfaces
export interface AnimationSettings {
    enableNodeAnimations: boolean;
    enableMotionBlur: boolean;
    motionBlurStrength: number;
    selectionWaveEnabled: boolean;
    pulseEnabled: boolean;
    pulseSpeed: number;
    pulseStrength: number;
    waveSpeed: number;
}

export interface BloomSettings {
    enabled: boolean;
    strength: number;
    radius: number;
    edgeBloomStrength: number;
    nodeBloomStrength: number;
    environmentBloomStrength: number;
}

export interface EdgeSettings {
    color: string;
    opacity: number;
    arrowSize: number;
    baseWidth: number;
    enableArrows: boolean;
    widthRange: [number, number];
}

export interface HologramSettings {
    ringCount: number;
    ringSizes: number[];
    ringRotationSpeed: number;
    globalRotationSpeed: number;
    ringColor: string;
    ringOpacity: number;
    enableBuckminster: boolean;
    buckminsterScale: number;
    buckminsterOpacity: number;
    enableGeodesic: boolean;
    geodesicScale: number;
    geodesicOpacity: number;
    enableTriangleSphere: boolean;
    triangleSphereScale: number;
    triangleSphereOpacity: number;
}

export interface LabelSettings {
    enableLabels: boolean;
    desktopFontSize: number;
    textColor: string;
    textOutlineColor: string;
    textOutlineWidth: number;
    textResolution: number;
    textPadding: number;
    billboardMode: boolean;
}

export interface NodeSettings {
    quality: 'low' | 'medium' | 'high';
    enableInstancing: boolean;
    enableHologram: boolean;
    enableMetadataShape: boolean;
    enableMetadataVisualization: boolean;
    baseSize: number;
    sizeRange: [number, number];
    baseColor: string;
    opacity: number;
    colorRangeAge: [string, string];
    colorRangeLinks: [string, string];
    metalness: number;
    roughness: number;
}

export interface PhysicsSettings {
    enabled: boolean;
    attractionStrength: number;
    repulsionStrength: number;
    springStrength: number;
    damping: number;
    iterations: number;
    maxVelocity: number;
    collisionRadius: number;
    enableBounds: boolean;
    boundsSize: number;
}

export interface RenderingSettings {
    ambientLightIntensity: number;
    directionalLightIntensity: number;
    environmentIntensity: number;
    backgroundColor: string;
    enableAmbientOcclusion: boolean;
    enableAntialiasing: boolean;
    enableShadows: boolean;
}

export interface NetworkSettings {
    bindAddress: string;
    domain: string;
    port: number;
    enableHttp2: boolean;
    enableTls: boolean;
    minTlsVersion: string;
    maxRequestSize: number;
    enableRateLimiting: boolean;
    rateLimitRequests: number;
    rateLimitWindow: number;
    tunnelId: string;
}

export interface WebSocketSettings {
    url: string;
    reconnectAttempts: number;
    reconnectDelay: number;
    binaryChunkSize: number;
    compressionEnabled: boolean;
    compressionThreshold: number;
    maxConnections: number;
    maxMessageSize: number;
    updateRate: number;
}

export interface SecuritySettings {
    allowedOrigins: string[];
    auditLogPath: string;
    cookieHttponly: boolean;
    cookieSamesite: string;
    cookieSecure: boolean;
    csrfTokenTimeout: number;
    enableAuditLogging: boolean;
    enableRequestValidation: boolean;
    sessionTimeout: number;
}

export interface DebugSettings {
    enabled: boolean;
    enableDataDebug: boolean;
    enableWebsocketDebug: boolean;
    logBinaryHeaders: boolean;
    logFullJson: boolean;
}

// Main settings interface
export interface Settings {
    visualization: {
        nodes: NodeSettings;
        edges: EdgeSettings;
        labels: LabelSettings;
        hologram: HologramSettings;
        animations: AnimationSettings;
    };
    xr: XRSettings;
}

export * from './settings/base';
export * from './settings/utils';

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
    mode: XRSessionMode;
    roomScale: boolean;
    spaceType: XRReferenceSpaceType;
    quality: 'low' | 'medium' | 'high';

    // Input and interaction
    input: {
        enableHandTracking: boolean;
        enableHaptics: boolean;
        hapticIntensity: number;
        dragThreshold: number;
        pinchThreshold: number;
        rotationThreshold: number;
        interactionRadius: number;
    };

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
    passthrough: {
        enabled: boolean;
        opacity: number;
        brightness: number;
        contrast: number;
        portalSize: number;
        portalEdgeColor: string;
        portalEdgeWidth: number;
    };
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
    enableMotionBlur: boolean;
    enableNodeAnimations: boolean;
    motionBlurStrength: number;
    selectionWaveEnabled: boolean;
    pulseEnabled: boolean;
    rippleEnabled: boolean;
    edgeAnimationEnabled: boolean;
    flowParticlesEnabled: boolean;
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
    arrowSize: number;
    baseWidth: number;
    color: string;
    enableArrows: boolean;
    opacity: number;
    widthRange: [number, number];
}

export interface HologramSettings {
    ringCount: number;
    ringColor: string;
    ringOpacity: number;
    ringSizes: number[];
    ringRotationSpeed: number;
    enableBuckminster: boolean;
    buckminsterScale: number;
    buckminsterOpacity: number;
    enableGeodesic: boolean;
    geodesicScale: number;
    geodesicOpacity: number;
    enableTriangleSphere: boolean;
    triangleSphereScale: number;
    triangleSphereOpacity: number;
    globalRotationSpeed: number;
}

export interface LabelSettings {
    enableLabels: boolean;
    textColor: string;
    textOutlineColor: string;
    textOutlineWidth: number;
    textResolution: number;
    textPadding: number;
    desktopFontSize: number;
    billboardMode: 'camera' | 'up';
}

export interface NodeSettings {
    baseColor: string;
    baseSize: number;
    sizeRange: [number, number];
    enableMetadataShape: boolean;
    colorRangeAge: [string, string];
    colorRangeLinks: [string, string];
    metalness: number;
    roughness: number;
    opacity: number;
    enableMetadataVisualization: boolean;
    enableHologram: boolean;
    enableInstancing: boolean;
    quality: 'low' | 'medium' | 'high';
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
    heartbeatInterval: number;
    heartbeatTimeout: number;
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
    visualization: VisualizationSettings;
    xr: XRSettings;
    system: SystemSettings;
    labels: LabelSettings;
}

export * from './settings/base';
export * from './settings/utils';

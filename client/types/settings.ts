// Core visualization settings
export interface VisualizationSettings {
    nodes: NodeSettings;
    edges: EdgeSettings;
    hologram: HologramSettings;
    labels: LabelSettings;
    animations: AnimationSettings;
    physics: PhysicsSettings;
    bloom: BloomSettings;
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
    defaultWidth: number;
    minWidth: number;
    maxWidth: number;
    widthProperty?: string;
    colorProperty?: string;
    arrowSize: number;
    baseWidth: number;
    enableArrows: boolean;
    opacity: number;
    widthRange: [number, number];
}

export interface HologramSettings {
    color: string;
    opacity: number;
    glowIntensity: number;
    rotationSpeed: number;
    enabled: boolean;
    ringCount: number;
    ringColor: string;
    ringOpacity: number;
    ringSizes: [number, number, number];
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
    enabled: boolean;
    size: number;
    color: string;
    enableLabels: boolean;
    desktopFontSize: number;
    textColor: string;
}

export interface NodeSettings {
    color: string;
    defaultSize: number;
    minSize: number;
    maxSize: number;
    sizeProperty?: string;
    colorProperty?: string;
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
    material: {
        type: 'basic' | 'phong';
        transparent: boolean;
        opacity: number;
    };
}

export interface NodeMeshUserData {
    id: string;
    type?: string;
    properties?: Record<string, unknown>;
    rotationSpeed?: number;
    data?: any;
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
    showGrid: boolean;
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
    logLevel: 'error' | 'warn' | 'info' | 'debug' | 'trace';
}

// Main settings interface
export interface Settings {
    visualization: VisualizationSettings;
    xr: XRSettings;
    system: SystemSettings;
    render: RenderingSettings;
    controls: {
        autoRotate: boolean;
        rotateSpeed: number;
        zoomSpeed: number;
        panSpeed: number;
    };
}

export * from './settings/base';
export * from './settings/utils';

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
    threshold: number;
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

// Client-side WebSocket settings (non-sensitive)
export interface WebSocketSettings {
    reconnectAttempts: number;
    reconnectDelay: number;
    binaryChunkSize: number;
    compressionEnabled: boolean;
    compressionThreshold: number;
    updateRate: number;
}

// Debug settings (UI-only)
export interface DebugSettings {
    enabled: boolean;
    enableDataDebug: boolean;
    enableWebsocketDebug: boolean;
    logBinaryHeaders: boolean;
    logFullJson: boolean;
    logLevel: 'error' | 'warn' | 'info' | 'debug' | 'trace';
    logFormat: string;
}

// XR settings
export interface XRSettings {
    // Session Settings
    mode: XRSessionMode;
    quality: 'low' | 'medium' | 'high';
    roomScale: boolean;
    spaceType: 'viewer' | 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';
    
    // Platform Settings
    autoEnterAR?: boolean;
    hideControlPanel?: boolean;
    preferredMode?: XRSessionMode;
    
    // Hand Tracking
    enableHandTracking: boolean;
    handMeshEnabled: boolean;
    handMeshColor: string;
    handMeshOpacity: number;
    handPointSize: number;
    handRayEnabled: boolean;
    handRayColor: string;
    handRayWidth: number;
    gestureSsmoothing: number;
    
    // Interaction
    enableHaptics: boolean;
    hapticIntensity: number;
    dragThreshold: number;
    pinchThreshold: number;
    rotationThreshold: number;
    interactionRadius: number;
    
    // Scene Understanding
    enableLightEstimation?: boolean;
    enablePlaneDetection?: boolean;
    enableSceneUnderstanding?: boolean;
    planeColor?: string;
    planeOpacity?: number;
    showPlaneOverlay?: boolean;
    snapToFloor?: boolean;
}

// Main settings interface
export interface Settings {
    visualization: {
        nodes: NodeSettings;
        edges: EdgeSettings;
        physics: PhysicsSettings;
        rendering: RenderingSettings;
        animations: AnimationSettings;
        labels: LabelSettings;
        bloom: BloomSettings;
        hologram: HologramSettings;
    };
    system: {
        websocket: WebSocketSettings;
        debug: DebugSettings;
    };
    xr: XRSettings;
}

export * from './settings/base';
export * from './settings/utils';

import { XRSessionMode } from '../xr';

export interface XRSettings {
    // Session Settings
    mode: XRSessionMode;
    roomScale: boolean;
    spaceType: 'viewer' | 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';
    quality: 'low' | 'medium' | 'high';
    
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
    gestureSmoothing: number;
    
    // Interaction
    enableHaptics: boolean;
    hapticIntensity: number;
    dragThreshold: number;
    pinchThreshold: number;
    rotationThreshold: number;
    interactionRadius: number;
    movementSpeed: number;
    
    // Scene Understanding
    enableLightEstimation: boolean;
    enablePlaneDetection: boolean;
    enableSceneUnderstanding: boolean;
    planeColor: string;
    planeOpacity: number;
    showPlaneOverlay: boolean;
    snapToFloor: boolean;
    
    // Passthrough
    enablePassthroughPortal: boolean;
    passthroughOpacity: number;
    passthroughBrightness: number;
    passthroughContrast: number;
    portalSize: number;
    portalEdgeColor: string;
    portalEdgeWidth: number;
}

// Platform-specific XR settings
export interface QuestXRSettings extends XRSettings {
    enableHandMeshes: boolean;
    enableControllerModel: boolean;
    controllerProfile: string;
}

export interface WebXRSettings extends XRSettings {
    fallbackToInline: boolean;
    requireFeatures: string[];
    optionalFeatures: string[];
}

// Default XR settings
export const defaultXRSettings: XRSettings = {
    // Session Settings
    mode: 'immersive-ar',
    roomScale: true,
    spaceType: 'local-floor',
    quality: 'high',
    
    // Platform Settings
    autoEnterAR: true,
    hideControlPanel: true,
    preferredMode: 'immersive-ar',
    
    // Hand Tracking
    enableHandTracking: true,
    handMeshEnabled: true,
    handMeshColor: '#ffffff',
    handMeshOpacity: 0.5,
    handPointSize: 5,
    handRayEnabled: true,
    handRayColor: '#00ff00',
    handRayWidth: 2,
    gestureSmoothing: 0.5,
    
    // Interaction
    enableHaptics: true,
    hapticIntensity: 0.5,
    dragThreshold: 0.02,
    pinchThreshold: 0.7,
    rotationThreshold: 0.1,
    interactionRadius: 0.5,
    movementSpeed: 0.05,
    
    // Scene Understanding
    enableLightEstimation: true,
    enablePlaneDetection: true,
    enableSceneUnderstanding: true,
    planeColor: '#808080',
    planeOpacity: 0.5,
    showPlaneOverlay: true,
    snapToFloor: true,
    
    // Passthrough
    enablePassthroughPortal: false,
    passthroughOpacity: 1.0,
    passthroughBrightness: 1.0,
    passthroughContrast: 1.0,
    portalSize: 2.0,
    portalEdgeColor: '#ffffff',
    portalEdgeWidth: 2.0
};

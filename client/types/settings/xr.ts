import { XRSessionMode } from '../xr';

export interface XRSettings {
    // Session Settings
    mode: 'immersive-ar' | 'immersive-vr';
    roomScale: boolean;
    spaceType: XRReferenceSpaceType;
    
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
    
    // Quality Settings
    quality: 'low' | 'medium' | 'high';
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
    mode: 'immersive-ar',
    roomScale: true,
    spaceType: 'local-floor',
    
    enableHandTracking: true,
    handMeshEnabled: true,
    handMeshColor: '#ffffff',
    handMeshOpacity: 0.5,
    handPointSize: 5,
    handRayEnabled: true,
    handRayColor: '#00ff00',
    handRayWidth: 2,
    gestureSsmoothing: 0.5,
    
    enableHaptics: true,
    hapticIntensity: 0.5,
    dragThreshold: 0.02,
    pinchThreshold: 0.7,
    rotationThreshold: 0.1,
    interactionRadius: 0.5,
    
    enableLightEstimation: true,
    enablePlaneDetection: true,
    enableSceneUnderstanding: true,
    planeColor: '#808080',
    planeOpacity: 0.5,
    showPlaneOverlay: true,
    snapToFloor: true,
    
    enablePassthroughPortal: false,
    passthroughOpacity: 1,
    passthroughBrightness: 1,
    passthroughContrast: 1,
    portalSize: 2,
    portalEdgeColor: '#ffffff',
    portalEdgeWidth: 2,
    
    quality: 'medium'
};

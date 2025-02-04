import { Settings, VisualizationSettings } from '../types/settings';

// Helper function to convert number to hex color
function toHexColor(num: number): string {
    return `#${num.toString(16).padStart(6, '0')}`;
}

// Export visualization defaults separately for reuse
export const defaultVisualizationSettings: VisualizationSettings = {
    nodes: {
        quality: 'high',
        enableInstancing: true,
        enableMetadataShape: true,
        enableMetadataVisualization: true,
        baseSize: 1.0,  // Default size with range up to 10.0
        sizeRange: [0.8, 5.0],  // Wider range for more size variation
        baseColor: toHexColor(0x4287f5),  // Softer blue
        opacity: 0.9,  // Slightly transparent
        colorRangeAge: [toHexColor(0x4287f5), toHexColor(0xf542a1)],  // Blue to pink
        colorRangeLinks: [toHexColor(0x4287f5), toHexColor(0xa142f5)],  // Blue to purple
        metalness: 0.3,  // Much less metallic for reduced reflections
        roughness: 0.7,  // More diffuse for reduced specular highlights
        enableHologram: false
    },
    edges: {
        color: toHexColor(0x6e7c91),  // Softer gray-blue that complements node colors
        opacity: 0.4,  // More subtle connections
        arrowSize: 0.8,  // Small arrows when enabled
        baseWidth: 0.8,  // Thinner base width
        enableArrows: false,  // Default to no arrows
        widthRange: [0.6, 1.6]  // More subtle width variation
    },
    animations: {
        enableNodeAnimations: true,
        enableMotionBlur: true,  // Enable for smoother transitions
        motionBlurStrength: 0.35,  // Subtle motion blur
        selectionWaveEnabled: true,
        pulseEnabled: true,
        pulseSpeed: 1.2,  // Slightly slower, more gentle pulse
        pulseStrength: 0.8,  // Less intense pulse
        waveSpeed: 0.8  // Slower, more graceful wave
    },
    labels: {
        enableLabels: true,
        desktopFontSize: 14,  // Slightly smaller for better integration
        textColor: toHexColor(0xe1e5eb),  // Softer white for better contrast
        textOutlineColor: toHexColor(0x1a1a1a),  // Darker outline for better readability
        textOutlineWidth: 3,  // Thicker outline for better visibility
        textResolution: 512,  // Higher resolution for sharper text
        textPadding: 6,  // More padding for better spacing
        billboardMode: true
    },
    rendering: {
        ambientLightIntensity: 0.4,  // Reduced ambient for less overall brightness
        directionalLightIntensity: 0.8,  // Reduced directional for less harsh lighting
        environmentIntensity: 0.8,  // Reduced environment intensity for better balance
        backgroundColor: toHexColor(0x1a1a2e),  // Slightly blue-tinted dark background
        enableAmbientOcclusion: true,
        enableAntialiasing: true,
        enableShadows: true,
        shadowMapSize: 2048,  // Higher resolution shadows
        shadowBias: 0.0001  // Fine-tuned shadow bias
    },
    bloom: {
        enabled: false,  // Disabled to reduce overall brightness
        strength: 1.2,        // Slightly reduced overall bloom
        radius: 0.6,         // Tighter bloom radius
        threshold: 0.4,      // Higher threshold for more selective bloom
        edgeBloomStrength: 0.4,  // Subtler edge glow
        nodeBloomStrength: 0.8,  // Much less intense node glow
        environmentBloomStrength: 0.8  // Reduced environment bloom
    },
    hologram: {
        ringCount: 4,                    // One more ring for better effect
        ringSizes: [0.8, 1.2, 1.6, 2],  // More gradual size progression
        ringRotationSpeed: 0.08,         // Slower, more graceful rotation
        globalRotationSpeed: 0.06,       // Slower global rotation
        ringColor: toHexColor(0x4287f5), // Match node base color
        ringOpacity: 0.2,                // Much more subtle rings
        enableBuckminster: false,
        buckminsterScale: 1.2,           // Slightly larger when enabled
        buckminsterOpacity: 0.3,         // More subtle buckminster
        enableGeodesic: false,
        geodesicScale: 1.1,              // Slightly larger when enabled
        geodesicOpacity: 0.3,            // More subtle geodesic
        enableTriangleSphere: false,
        triangleSphereScale: 1.1,        // Slightly larger when enabled
        triangleSphereOpacity: 0.3       // More subtle triangle sphere
    },
    physics: {
        enabled: true,
        attractionStrength: 0.015,     // Range: 0.001-1.0, proven value from Initial phase
        repulsionStrength: 1200.0,     // Range: 1.0-10000.0, proven value from settings
        springStrength: 0.018,         // Range: 0.001-1.0, proven value from Initial phase
        damping: 0.95,                 // Range: 0.5-0.95, proven value from Initial phase
        iterations: 300,               // Range: 1-500, proven value from Initial phase
        maxVelocity: 0.2,             // Range: 0.1-1.0, matches time_step
        collisionRadius: 0.25,         // From Rust default
        enableBounds: true,
        boundsSize: 100.0             // Range: 10-500, from SimulationParams
    }
};

// Main settings object with all defaults
export const defaultSettings: Settings = {
    visualization: defaultVisualizationSettings,
    system: {
        websocket: {
            reconnectAttempts: 3,         // Fewer attempts but more frequent
            reconnectDelay: 3000,         // Shorter delay between attempts
            binaryChunkSize: 2048,        // Larger chunks for better performance
            compressionEnabled: true,
            compressionThreshold: 512,     // Lower threshold to compress more data
            updateRate: 90                 // Higher update rate for smoother animations
        },
        debug: {
            enabled: false,
            enableDataDebug: false,
            enableWebsocketDebug: false,
            logBinaryHeaders: false,
            logFullJson: false,
            logLevel: 'warn',             // Default to warnings only
            logFormat: 'pretty'           // More readable logs by default
        }
    },
    xr: {
        // Mode and Space Settings
        mode: 'immersive-ar' as const,
        quality: 'high',
        roomScale: true,
        spaceType: 'local-floor',

        // Hand Tracking Settings
        enableHandTracking: true,
        handMeshEnabled: true,
        handMeshColor: toHexColor(0x4287f5),  // Match visualization theme
        handMeshOpacity: 0.3,                 // More subtle hand visualization
        handPointSize: 3,                     // Smaller, less intrusive points
        handRayEnabled: true,
        handRayColor: toHexColor(0x4287f5),   // Match visualization theme
        handRayWidth: 1.5,                    // Thinner ray
        gestureSmoothing: 0.7,                // Smoother gesture tracking

        // Interaction Settings
        enableHaptics: true,
        hapticIntensity: 0.3,                 // Gentler haptics
        dragThreshold: 0.08,                  // Lower threshold for easier dragging
        pinchThreshold: 0.3,                  // Lower threshold for easier pinching
        rotationThreshold: 0.08,              // Lower threshold for easier rotation
        interactionRadius: 0.15,              // Larger interaction radius
        movementSpeed: 0.08,                  // Faster movement
        deadZone: 0.12,                       // Slightly larger deadzone
        movementAxes: {
            horizontal: 2,                     // Right joystick X
            vertical: 3                        // Right joystick Y
        },

        // Platform Settings
        autoEnterAR: false,                   // Let user choose when to enter
        hideControlPanel: false,              // Show control panel by default
        preferredMode: 'immersive-ar',

        // Scene Understanding
        enableLightEstimation: true,
        enablePlaneDetection: true,
        enableSceneUnderstanding: true,
        planeColor: toHexColor(0x4287f5),     // Match visualization theme
        planeOpacity: 0.2,                    // More subtle planes
        showPlaneOverlay: true,
        snapToFloor: true,

        // Passthrough Settings
        enablePassthroughPortal: false,
        passthroughOpacity: 0.8,              // Slightly transparent
        passthroughBrightness: 1.1,           // Slightly brighter
        passthroughContrast: 1.2,             // Higher contrast
        portalSize: 2.5,                      // Larger portal
        portalEdgeColor: toHexColor(0x4287f5), // Match visualization theme
        portalEdgeWidth: 0.5                  // 20% of portal size (2.5 * 0.2 = 0.5)
    }
};

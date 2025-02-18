import { Settings } from '../types/settings/base';

export const defaultSettings: Settings = {
    visualization: {
        nodes: {
            baseColor: '#32aeae',
            metalness: 0.8,
            opacity: 1.0,
            roughness: 0.2,
            sizeRange: [0.05, 0.2],  // 5cm to 20cm
            quality: 'medium',
            enableInstancing: true,
            enableHologram: true,
            enableMetadataShape: false,
            enableMetadataVisualization: false,
            colorRangeAge: ['#ff0000', '#00ff00'],
            colorRangeLinks: ['#0000ff', '#ff00ff']
        },
        edges: {
            arrowSize: 0.02,         // 2cm
            baseWidth: 0.005,        // 5mm
            color: '#888888',
            enableArrows: false,
            opacity: 0.8,
            widthRange: [0.005, 0.01],  // 5mm to 10mm
            quality: 'medium',
            enableFlowEffect: true,
            flowSpeed: 1.0,
            flowIntensity: 0.6,
            glowStrength: 0.4,
            distanceIntensity: 0.3,
            useGradient: false,
            gradientColors: ['#888888', '#aaaaaa']
        },
        physics: {
            enabled: true,
            iterations: 100,              // Balanced for performance and stability
            attractionStrength: 0.02,     // Balanced with quadratic repulsion
            repulsionStrength: 0.05,      // Reduced due to quadratic distance falloff
            repulsionDistance: 0.2,       // 20cm for tighter layout
            springStrength: 0.08,         // Linear spring force for edge following
            damping: 0.85,                // Balanced damping for stability
            maxVelocity: 0.2,             // 20cm/s maximum velocity
            collisionRadius: 0.1,         // 10cm collision detection radius
            massScale: 1.0,               // Default mass scaling
            boundaryDamping: 0.9,         // Strong boundary damping
            enableBounds: true,           // Enable bounds by default
            boundsSize: 0.5               // 50cm cube for better spacing
        },
        rendering: {
            ambientLightIntensity: 0.2,
            backgroundColor: '#1a1a1a',
            directionalLightIntensity: 0.2,
            enableAmbientOcclusion: false,
            enableAntialiasing: true,
            enableShadows: false,
            environmentIntensity: 0.2,
            shadowMapSize: 2048,
            shadowBias: 0.00001,
            context: 'desktop'
        },
        animations: {
            enableMotionBlur: true,
            enableNodeAnimations: true,
            motionBlurStrength: 1.0,
            selectionWaveEnabled: false,
            pulseEnabled: false,
            pulseSpeed: 1.0,
            pulseStrength: 0.5,
            waveSpeed: 1.0
        },
        labels: {
            desktopFontSize: 14,
            enableLabels: true,
            textColor: '#ffffff',
            textOutlineColor: '#000000',
            textOutlineWidth: 0.1,
            textResolution: 16,
            textPadding: 2,
            billboardMode: 'camera'
        },
        bloom: {
            edgeBloomStrength: 2.0,
            enabled: true,
            environmentBloomStrength: 3.0,
            nodeBloomStrength: 3.0,
            radius: 2.0,
            strength: 3.0,
            threshold: 0.0
        },
        hologram: {
            ringCount: 2,
            sphereSizes: [0.08, 0.16],    // 8cm and 16cm
            ringRotationSpeed: 1.0,
            ringColor: '#00ffff',
            ringOpacity: 0.6,
            enableBuckminster: false,
            enableGeodesic: false,
            buckminsterSize: 0.0,         // Disabled by default
            buckminsterOpacity: 0,
            geodesicSize: 0.0,            // Disabled by default
            geodesicOpacity: 0,
            enableTriangleSphere: true,
            triangleSphereSize: 0.16,     // 16cm
            triangleSphereOpacity: 0.15,
            globalRotationSpeed: 0.03
        }
    },
    system: {
        websocket: {
            binaryChunkSize: 32768,
            compressionEnabled: true,
            compressionThreshold: 1024,
            reconnectAttempts: 5,
            reconnectDelay: 5000,
            updateRate: 30
        },
        debug: {
            enabled: false,
            enableDataDebug: false,
            enableWebsocketDebug: false,
            logBinaryHeaders: false,
            logFullJson: false,
            logLevel: 'info',
            logFormat: 'json'
        }
    },
    xr: {
        mode: 'immersive-vr',
        roomScale: 1.0,                   // Real-world 1:1 scale
        spaceType: 'local-floor',
        quality: 'high',
        autoEnterAR: false,
        hideControlPanel: true,
        preferredMode: 'immersive-vr',
        enableHandTracking: true,
        handMeshEnabled: true,
        handMeshColor: '#4287f5',
        handMeshOpacity: 0.3,
        handPointSize: 0.006,             // 6mm
        handRayEnabled: true,
        handRayColor: '#4287f5',
        handRayWidth: 0.003,              // 3mm
        gestureSmoothing: 0.5,
        enableHaptics: true,
        hapticIntensity: 0.5,
        dragThreshold: 0.02,              // 2cm movement required to start drag
        pinchThreshold: 0.3,              // 30% pinch required for activation
        rotationThreshold: 0.08,          // 8% rotation required for activation
        interactionRadius: 0.15,          // 15cm interaction sphere
        movementSpeed: 0.08,              // 8cm per frame at full stick deflection
        deadZone: 0.12,                   // 12% stick movement required
        movementAxes: {
            horizontal: 2,                 // Right joystick X
            vertical: 3                    // Right joystick Y
        },
        enableLightEstimation: false,
        enablePlaneDetection: true,
        enableSceneUnderstanding: true,
        planeColor: '#808080',
        planeOpacity: 0.5,
        showPlaneOverlay: false,
        snapToFloor: false,
        planeDetectionDistance: 3.0,      // 3m maximum plane detection distance
        enablePassthroughPortal: false,
        passthroughOpacity: 0.8,
        passthroughBrightness: 1.1,
        passthroughContrast: 1.0,
        portalSize: 2.5,                  // 2.5m portal size
        portalEdgeColor: '#ffffff',
        portalEdgeWidth: 0.02             // 2cm edge width
    }
};

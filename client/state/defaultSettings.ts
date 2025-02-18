import { Settings } from '../types/settings/base';

export const defaultSettings: Settings = {
    visualization: {
        nodes: {
            baseColor: '#32aeae',
            metalness: 0.8,
            opacity: 1.0,
            roughness: 0.2,
            sizeRange: [0.2, 0.7], // In meters
            quality: 'medium',
            enableInstancing: true,
            enableHologram: true,
            enableMetadataShape: false,
            enableMetadataVisualization: false,
            colorRangeAge: ['#ff0000', '#00ff00'],
            colorRangeLinks: ['#0000ff', '#ff00ff']
        },
        edges: {
            arrowSize: 5.0,
            baseWidth: 3.0,
            color: '#888888',
            enableArrows: false,
            opacity: 0.8,
            widthRange: [1.5, 4.0],
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
            iterations: 100,                // Balanced for performance and stability
            attractionStrength: 0.015,        // Moderate attraction for connected nodes
            repulsionStrength: 1500.0,       // Strong repulsion for good spacing
            repulsionDistance: 0.5,      // In meters (0.5m) for initial layout
            springStrength: 0.018,            // Moderate spring force for stability
            damping: 0.88,                   // Higher damping for stability
            maxVelocity: 2.5,               // Moderate velocity limit
            collisionRadius: 0.25,           // Default collision detection radius
            massScale: 1.0,                 // Default mass scaling
            boundaryDamping: 0.5,           // Strong boundary damping to prevent oscillation
            enableBounds: true,             // Enable bounds by default
            boundsSize: 10.0                // In meters (10m cube)
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
            sphereSizes: [0.04, 0.08],      // In meters
            ringRotationSpeed: 1.0,
            ringColor: '#00ffff',
            ringOpacity: 0.6,
            enableBuckminster: false,
            enableGeodesic: false,
            buckminsterSize: 0.0,           // In meters
            buckminsterOpacity: 0,
            geodesicSize: 0.0,              // In meters
            geodesicOpacity: 0,
            enableTriangleSphere: true,
            triangleSphereSize: 0.08,       // In meters
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
        roomScale: 1.0,                     // Base scale in meters (1.0 = real-world scale)
        spaceType: 'local-floor',
        quality: 'high',
        autoEnterAR: false,
        hideControlPanel: true,
        preferredMode: 'immersive-vr',
        enableHandTracking: true,
        handMeshEnabled: true,
        handMeshColor: '#4287f5',
        handMeshOpacity: 0.3,
        handPointSize: 3.0,
        handRayEnabled: true,
        handRayColor: '#4287f5',
        handRayWidth: 1.5,
        gestureSmoothing: 0.5,
        enableHaptics: true,
        hapticIntensity: 0.5,
        dragThreshold: 0.02,
        pinchThreshold: 0.5,
        rotationThreshold: 0.1,
        interactionRadius: 0.5,
        movementSpeed: 0.05,
        deadZone: 0.1,
        movementAxes: {
            horizontal: 2,
            vertical: 3
        },
        enableLightEstimation: false,
        enablePlaneDetection: true,
        enableSceneUnderstanding: true,
        planeColor: '#808080',
        planeOpacity: 0.5,
        showPlaneOverlay: false,
        snapToFloor: false,
        planeDetectionDistance: 3.0,
        enablePassthroughPortal: false,
        passthroughOpacity: 0.8,
        passthroughBrightness: 1.1,
        passthroughContrast: 1.0,
        portalSize: 2.0,
        portalEdgeColor: '#ffffff',
        portalEdgeWidth: 2.0
    }
};

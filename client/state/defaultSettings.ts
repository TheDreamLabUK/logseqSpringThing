import { Settings } from '../types/settings/base';

export const defaultSettings: Settings = {
    visualization: {
        nodes: {
            baseColor: '#32aeae',
            baseSize: 1.5,
            metalness: 0.8,
            opacity: 1.0,
            roughness: 0.2,
            sizeRange: [1.0, 3.0],
            quality: 'high',
            enableInstancing: false,
            enableHologram: false,
            enableMetadataShape: false,
            enableMetadataVisualization: false,
            colorRangeAge: ['#ff0000', '#00ff00'],
            colorRangeLinks: ['#0000ff', '#ff00ff']
        },
        edges: {
            arrowSize: 5.0,
            baseWidth: 0.0,
            color: '#888888',
            enableArrows: false,
            opacity: 0.3,
            widthRange: [1.0, 2.0]
        },
        physics: {
            attractionStrength: 0.015,
            boundsSize: 100.0,
            collisionRadius: 0.25,
            damping: 0.5,
            enableBounds: false,
            enabled: true,
            iterations: 100,
            maxVelocity: 2.0,
            repulsionStrength: 0.4,
            springStrength: 0.6
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
            shadowBias: 0.00001
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
            textOutlineWidth: 2,
            textResolution: 32,
            textPadding: 2,
            billboardMode: true
        },
        bloom: {
            edgeBloomStrength: 1.1,
            enabled: true,
            environmentBloomStrength: 1.0,
            nodeBloomStrength: 2.0,
            radius: 0.5,
            strength: 0.6,
            threshold: 0.5
        },
        hologram: {
            ringCount: 3,
            ringColor: '#00ffff',
            ringOpacity: 0.8,
            ringSizes: [0.8, 1.2, 1.6, 2.0],
            ringRotationSpeed: 1.0,
            enableBuckminster: true,
            buckminsterScale: 0.6,
            buckminsterOpacity: 0.6,
            enableGeodesic: false,
            geodesicScale: 0.4,
            geodesicOpacity: 0.5,
            enableTriangleSphere: true,
            triangleSphereScale: 0.5,
            triangleSphereOpacity: 0.5,
            globalRotationSpeed: 1.0
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
        roomScale: 0.01,
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

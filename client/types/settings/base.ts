// Base types for settings
export interface BaseSettings {
    visualization: {
        animations: {
            enableMotionBlur: boolean;
            enableNodeAnimations: boolean;
            motionBlurStrength: number;
            selectionWaveEnabled: boolean;
            pulseEnabled: boolean;
            rippleEnabled: boolean;
            edgeAnimationEnabled: boolean;
            flowParticlesEnabled: boolean;
        };
        bloom: {
            enabled: boolean;
            strength: number;
            radius: number;
            edgeBloomStrength: number;
            nodeBloomStrength: number;
            environmentBloomStrength: number;
        };
        edges: {
            arrowSize: number;
            baseWidth: number;
            color: string;
            enableArrows: boolean;
            opacity: number;
            widthRange: [number, number];
        };
        hologram: {
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
        };
        labels: {
            enableLabels: boolean;
            textColor: string;
            fontSize: number;
            fontFamily: string;
            strokeWidth: number;
            strokeColor: string;
            backgroundColor: string;
            backgroundOpacity: number;
            maxLength: number;
            minDistance: number;
            maxDistance: number;
        };
        nodes: {
            baseSize: number;
            sizeRange: [number, number];
            color: string;
            opacity: number;
            outlineWidth: number;
            outlineColor: string;
            enableGlow: boolean;
            glowStrength: number;
            glowColor: string;
            enablePulse: boolean;
            pulseSpeed: number;
            pulseStrength: number;
        };
        physics: {
            enabled: boolean;
            gravity: number;
            springLength: number;
            springStrength: number;
            damping: number;
            repulsion: number;
            timeStep: number;
            maxVelocity: number;
            minVelocity: number;
            maxIterations: number;
        };
    };
    xr: {
        mode: 'immersive-ar' | 'immersive-vr';
        roomScale: boolean;
        quality: 'low' | 'medium' | 'high';
        input: {
            handTracking: boolean;
            controllerModel: string;
            hapticFeedback: boolean;
            gestureThreshold: number;
            pinchThreshold: number;
            grabThreshold: number;
        };
        visuals: {
            shadowQuality: 'none' | 'low' | 'medium' | 'high';
            antiAliasing: boolean;
            foveatedRendering: boolean;
            foveationLevel: number;
            resolution: number;
        };
        environment: {
            skybox: boolean;
            skyboxColor: string;
            groundPlane: boolean;
            groundColor: string;
            fog: boolean;
            fogColor: string;
            fogDensity: number;
        };
        passthrough: {
            enabled: boolean;
            opacity: number;
            brightness: number;
            contrast: number;
            saturation: number;
            blendMode: 'alpha-blend' | 'additive' | 'multiply';
        };
    };
    system: {
        debug: {
            enabled: boolean;
            enableDataDebug: boolean;  // Added required field
            enableWebsocketDebug: boolean;  // Added to match server
            logBinaryHeaders: boolean;  // Added to match server
            logLevel: 'error' | 'warn' | 'info' | 'debug' | 'trace';
            logFormat: string;  // Added to match server
            logFullJson: boolean;
            showStats?: boolean;  // Made optional since not in server
            showFPS?: boolean;  // Made optional since not in server
            showMemory?: boolean;  // Made optional since not in server
        };
        network: {
            websocketUrl: string;
            reconnectInterval: number;
            maxReconnectAttempts: number;
            heartbeatInterval: number;
            compressionEnabled: boolean;
            batchUpdates: boolean;
            batchInterval: number;
        };
        security: {
            enableEncryption: boolean;
            encryptionAlgorithm: string;
            encryptionKeySize: number;
            enableAuthentication: boolean;
            authenticationMethod: string;
            tokenExpiration: number;
        };
    };
}

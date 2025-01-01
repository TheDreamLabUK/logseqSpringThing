import { Scene, PerspectiveCamera, WebGLRenderer } from 'three';
import { Settings } from './types/settings';
import { EnhancedNodeManager } from './rendering/EnhancedNodeManager';
import { EdgeManager } from './rendering/EdgeManager';
import { HologramManager } from './rendering/HologramManager';
import { TextRenderer } from './rendering/textRenderer';
import { WebSocketService } from './websocket/websocketService';
import { LoggerImpl } from './logging/loggerImpl'; // Assuming LoggerImpl is defined in this file

export class GraphVisualization {
    private scene: Scene;
    private camera: PerspectiveCamera;
    private renderer: WebGLRenderer;
    private nodeManager: EnhancedNodeManager;
    private edgeManager: EdgeManager;
    private hologramManager: HologramManager;
    private textRenderer: TextRenderer;
    private websocketService: WebSocketService;
    constructor(settings: Settings) {
        // Initialize logger settings first
        LoggerImpl.setSettings(settings);
        
        this.scene = new Scene();
        this.camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(this.renderer.domElement);

        this.nodeManager = new EnhancedNodeManager(this.scene, settings);
        this.edgeManager = new EdgeManager(this.scene, settings);
        this.hologramManager = new HologramManager(this.scene, this.renderer, settings);
        this.textRenderer = new TextRenderer(this.camera);
        this.websocketService = WebSocketService.getInstance();

        this.setupEventListeners();
        this.animate();
    }

    private setupEventListeners() {
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    private animate() {
        requestAnimationFrame(() => this.animate());
        this.nodeManager.update(0.016);
        this.hologramManager.update(0.016);
        this.renderer.render(this.scene, this.camera);
    }

    public handleSettingsUpdate(settings: Settings) {
        this.nodeManager.handleSettingsUpdate(settings);
        this.edgeManager.handleSettingsUpdate(settings);
        this.hologramManager.updateSettings(settings);
        this.textRenderer.handleSettingsUpdate(settings.visualization.labels);
    }

    public dispose() {
        this.nodeManager.dispose();
        this.edgeManager.dispose();
        this.hologramManager.dispose();
        this.textRenderer.dispose();
        this.websocketService.close();
        document.body.removeChild(this.renderer.domElement);
    }
}

// Initialize the visualization
const settings: Settings = {
    visualization: {
        bloom: {
            enabled: false,
            strength: 0.5,
            radius: 1,
            edgeBloomStrength: 0.5,
            nodeBloomStrength: 0.5,
            environmentBloomStrength: 0.5
        },
        physics: {
            enabled: false,
            attractionStrength: 0.1,
            repulsionStrength: 0.1,
            springStrength: 0.1,
            damping: 0.5,
            iterations: 1,
            maxVelocity: 10,
            collisionRadius: 1,
            enableBounds: true,
            boundsSize: 100
        },
        rendering: {
            ambientLightIntensity: 0.5,
            directionalLightIntensity: 0.8,
            environmentIntensity: 1,
            backgroundColor: '#000000',
            enableAmbientOcclusion: true,
            enableAntialiasing: true,
            enableShadows: true
        },
        nodes: {
            quality: 'medium',
            enableInstancing: true,
            enableHologram: true,
            enableMetadataShape: false,
            enableMetadataVisualization: false,
            baseSize: 1,
            sizeRange: [0.5, 2],
            baseColor: '#ffffff',
            opacity: 0.8,
            colorRangeAge: ['#ff0000', '#00ff00'],
            colorRangeLinks: ['#0000ff', '#ff00ff'],
            metalness: 0.5,
            roughness: 0.5
        },
        edges: {
            color: '#888888',
            opacity: 0.6,
            arrowSize: 0.2,
            baseWidth: 0.1,
            enableArrows: false,
            widthRange: [0.1, 0.3]
        },
        labels: {
            enableLabels: true,
            desktopFontSize: 14,
            textColor: '#ffffff',
            textOutlineColor: '#000000',
            textOutlineWidth: 2,
            textResolution: 32,
            textPadding: 2,
            billboardMode: true
        },
        hologram: {
            ringCount: 3,
            ringSizes: [20, 15, 10],
            ringRotationSpeed: 0.1,
            globalRotationSpeed: 0.05,
            ringColor: '#00ffff',
            ringOpacity: 0.5,
            enableBuckminster: false,
            buckminsterScale: 30,
            buckminsterOpacity: 0.3,
            enableGeodesic: false,
            geodesicScale: 25,
            geodesicOpacity: 0.3,
            enableTriangleSphere: false,
            triangleSphereScale: 20,
            triangleSphereOpacity: 0.3
        },
        animations: {
            enableNodeAnimations: false,
            enableMotionBlur: false,
            motionBlurStrength: 0.5,
            selectionWaveEnabled: false,
            pulseEnabled: false,
            pulseSpeed: 1.0,
            pulseStrength: 0.5,
            waveSpeed: 1.0
        }
    },
    xr: {
        quality: 'high',
        mode: 'vr',
        roomScale: true,
        spaceType: 'local',
        input: 'hands',
        haptics: true,
        passthrough: false,
        visuals: {
            handMeshEnabled: true,
            handMeshColor: '#ffffff',
            handMeshOpacity: 0.5,
            handPointSize: 5,
            handRayEnabled: true,
            handRayColor: '#00ff00',
            handRayWidth: 2,
            gestureSsmoothing: 0.5
        },
        environment: {
            enableLightEstimation: true,
            enablePlaneDetection: true,
            enableSceneUnderstanding: true,
            planeColor: '#808080',
            planeOpacity: 0.5,
            showPlaneOverlay: true,
            snapToFloor: true
        }
    },
    system: {
        network: {
            bindAddress: '127.0.0.1',
            domain: 'localhost',
            port: 3000,
            enableHttp2: true,
            enableTls: false,
            minTlsVersion: 'TLS1.2',
            maxRequestSize: 10485760,
            enableRateLimiting: true,
            rateLimitRequests: 100,
            rateLimitWindow: 60,
            tunnelId: ''
        },
        websocket: {
            url: '',
            reconnectAttempts: 5,
            reconnectDelay: 5000,
            binaryChunkSize: 65536,
            compressionEnabled: true,
            compressionThreshold: 1024,
            maxConnections: 100,
            maxMessageSize: 32 * 1024 * 1024,
            updateRate: 60
        },
        security: {
            allowedOrigins: ['http://localhost:3000'],
            auditLogPath: './audit.log',
            cookieHttponly: true,
            cookieSamesite: 'Lax',
            cookieSecure: false,
            csrfTokenTimeout: 3600,
            enableAuditLogging: true,
            enableRequestValidation: true,
            sessionTimeout: 86400
        },
        debug: {
            enabled: true,
            enableDataDebug: true,
            enableWebsocketDebug: true,
            logBinaryHeaders: true,
            logFullJson: true
        }
    }
};

new GraphVisualization(settings);

import { Scene, PerspectiveCamera, WebGLRenderer } from 'three';
import { Settings } from './types/settings';
import { EnhancedNodeManager } from './rendering/EnhancedNodeManager';
import { EdgeManager } from './rendering/EdgeManager';
import { HologramManager } from './rendering/HologramManager';
import { TextRenderer } from './rendering/textRenderer';
import { WebSocketService } from './websocket/websocketService';

export class GraphVisualization {
    private scene: Scene;
    private camera: PerspectiveCamera;
    private renderer: WebGLRenderer;
    private nodeManager: EnhancedNodeManager;
    private edgeManager: EdgeManager;
    private hologramManager: HologramManager;
    private textRenderer: TextRenderer;
    private websocketService: WebSocketService;
    private settings: Settings;

    constructor(settings: Settings) {
        this.settings = settings;
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
        this.settings = settings;
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
            enableNodeAnimations: true,
            enableMotionBlur: false,
            motionBlurStrength: 0.5,
            selectionWaveEnabled: true,
            pulseEnabled: true,
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
        passthrough: false
    }
};

const visualization = new GraphVisualization(settings);

import { Scene, PerspectiveCamera, WebGLRenderer } from 'three';
import { Settings } from './types/settings';
import { EnhancedNodeManager } from './rendering/EnhancedNodeManager';
import { EdgeManager } from './rendering/EdgeManager';
import { HologramManager } from './rendering/HologramManager';
import { TextRenderer } from './rendering/textRenderer';
import { WebSocketService } from './websocket/websocketService';
import { SettingsStore } from './state/SettingsStore';

export class GraphVisualization {
    private scene: Scene;
    private camera: PerspectiveCamera;
    private renderer: WebGLRenderer;
    private nodeManager: EnhancedNodeManager;
    private edgeManager: EdgeManager;
    private hologramManager: HologramManager;
    private textRenderer: TextRenderer;
    private websocketService!: WebSocketService;

    private async initializeWebSocket(): Promise<void> {
        // Initialize settings first
        const settingsStore = SettingsStore.getInstance();
        await settingsStore.initialize();

        // Then initialize WebSocket
        this.websocketService = WebSocketService.getInstance();
        this.websocketService.onBinaryMessage((nodes) => {
            this.nodeManager.updateNodePositions(nodes);
        });
        this.websocketService.onSettingsUpdate((updatedSettings) => {
            this.handleSettingsUpdate(updatedSettings);
        });
        this.websocketService.onConnectionStatusChange((connected) => {
            if (connected) {
                console.log('WebSocket connected, requesting initial data');
                this.websocketService.sendMessage({ type: 'requestInitialData' });
            }
        });
        this.websocketService.connect();
    }
    constructor(settings: Settings) {
        this.scene = new Scene();
        this.camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(this.renderer.domElement);

        this.nodeManager = new EnhancedNodeManager(this.scene, settings);
        this.edgeManager = new EdgeManager(this.scene, settings);
        this.hologramManager = new HologramManager(this.scene, this.renderer, settings);
        this.textRenderer = new TextRenderer(this.camera);
        
        // Initialize WebSocket after settings are loaded
        this.initializeWebSocket();

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

// Import default settings
import { defaultSettings } from './state/defaultSettings';

// Initialize the visualization with default settings
new GraphVisualization(defaultSettings);

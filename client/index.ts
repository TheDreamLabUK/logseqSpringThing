import { Scene, PerspectiveCamera, WebGLRenderer } from 'three';
import { Settings } from './types/settings';
import { EnhancedNodeManager } from './rendering/EnhancedNodeManager';
import { EdgeManager } from './rendering/EdgeManager';
import { HologramManager } from './rendering/HologramManager';
import { TextRenderer } from './rendering/textRenderer';
import { WebSocketService } from './websocket/websocketService';
import { SettingsStore } from './state/SettingsStore';
import { LoggerConfig, createLogger } from './core/logger';

const logger = createLogger('GraphVisualization');

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
        logger.debug('Initializing WebSocket connection');
        // Initialize settings first
        const settingsStore = SettingsStore.getInstance();
        await settingsStore.initialize();
        logger.debug('Settings store initialized');

        // Then initialize WebSocket
        this.websocketService = WebSocketService.getInstance();
        this.websocketService.onBinaryMessage((nodes) => {
            logger.debug('Received binary node update', { nodeCount: nodes.length });
            this.nodeManager.updateNodePositions(nodes);
        });
        this.websocketService.onSettingsUpdate((updatedSettings) => {
            logger.debug('Received settings update');
            this.handleSettingsUpdate(updatedSettings);
        });
        this.websocketService.onConnectionStatusChange((connected) => {
            logger.info(`WebSocket connection status changed: ${connected}`);
            if (connected) {
                logger.debug('Requesting initial data');
                this.websocketService.sendMessage({ type: 'requestInitialData' });
            }
        });
        this.websocketService.connect();
        logger.debug('WebSocket initialization complete');
    }

    constructor(settings: Settings) {
        logger.debug('Initializing GraphVisualization');
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
        logger.debug('GraphVisualization initialization complete');
    }

    private setupEventListeners() {
        logger.debug('Setting up event listeners');
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            logger.debug('Window resized, updated renderer dimensions');
        });
    }

    private animate() {
        requestAnimationFrame(() => this.animate());
        this.nodeManager.update(0.016);
        this.hologramManager.update(0.016);
        this.renderer.render(this.scene, this.camera);
    }

    public handleSettingsUpdate(settings: Settings) {
        logger.debug('Handling settings update');
        this.nodeManager.handleSettingsUpdate(settings);
        this.edgeManager.handleSettingsUpdate(settings);
        this.hologramManager.updateSettings(settings);
        this.textRenderer.handleSettingsUpdate(settings.visualization.labels);
    }

    public dispose() {
        logger.debug('Disposing GraphVisualization');
        this.nodeManager.dispose();
        this.edgeManager.dispose();
        this.hologramManager.dispose();
        this.textRenderer.dispose();
        this.websocketService.close();
        document.body.removeChild(this.renderer.domElement);
        logger.debug('GraphVisualization disposed');
    }
}

// Import default settings
import { defaultSettings } from './state/defaultSettings';

// Enable debug logging
LoggerConfig.setGlobalDebug(true);
LoggerConfig.setFullJson(true);

logger.log('Starting application...');
new GraphVisualization(defaultSettings);

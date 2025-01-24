import { Settings } from './types/settings';
import { EnhancedNodeManager } from './rendering/EnhancedNodeManager';
import { EdgeManager } from './rendering/EdgeManager';
import { HologramManager } from './rendering/HologramManager';
import { TextRenderer } from './rendering/textRenderer';
import { WebSocketService } from './websocket/websocketService';
import { SettingsStore } from './state/SettingsStore';
import { LoggerConfig, createLogger } from './core/logger';
import { SceneManager } from './rendering/scene';

const logger = createLogger('GraphVisualization');

export class GraphVisualization {
    private sceneManager: SceneManager;
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
        
        // Create canvas element
        const canvas = document.createElement('canvas');
        document.body.appendChild(canvas);
        
        // Initialize SceneManager
        this.sceneManager = SceneManager.getInstance(canvas);
        
        // Initialize managers with SceneManager's scene and renderer
        const scene = this.sceneManager.getScene();
        const renderer = this.sceneManager.getRenderer();
        const camera = this.sceneManager.getCamera();
        
        this.nodeManager = new EnhancedNodeManager(scene, settings);
        this.edgeManager = new EdgeManager(scene, settings);
        this.hologramManager = new HologramManager(scene, renderer, settings);
        this.textRenderer = new TextRenderer(camera);
        
        // Initialize WebSocket after settings are loaded
        this.initializeWebSocket();
        
        // Start rendering
        this.sceneManager.start();
        logger.debug('GraphVisualization initialization complete');
    }


    public handleSettingsUpdate(settings: Settings) {
        logger.debug('Handling settings update');
        this.nodeManager.handleSettingsUpdate(settings);
        this.edgeManager.handleSettingsUpdate(settings);
        this.hologramManager.updateSettings(settings);
        this.textRenderer.handleSettingsUpdate(settings.visualization.labels);
        this.sceneManager.handleSettingsUpdate(settings);
    }

    public dispose() {
        logger.debug('Disposing GraphVisualization');
        this.nodeManager.dispose();
        this.edgeManager.dispose();
        this.hologramManager.dispose();
        this.textRenderer.dispose();
        this.websocketService.close();
        SceneManager.cleanup();
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

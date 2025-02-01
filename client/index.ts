import { Settings } from './types/settings';
import { EnhancedNodeManager } from './rendering/EnhancedNodeManager';
import { EdgeManager } from './rendering/EdgeManager';
import { HologramManager } from './rendering/HologramManager';
import { TextRenderer } from './rendering/textRenderer';
import { WebSocketService } from './websocket/websocketService';
import { SettingsStore } from './state/SettingsStore';
import { LoggerConfig, createLogger } from './core/logger';
import { platformManager } from './platform/platformManager';
import { XRSessionManager } from './xr/xrSessionManager';
import { XRInitializer } from './xr/xrInitializer';
import { SceneManager } from './rendering/scene';
import { graphDataManager } from './state/graphData';
import { debugState } from './core/debugState';
import './ui'; // Import UI initialization

const logger = createLogger('GraphVisualization');

// Helper for conditional debug logging
function debugLog(message: string, ...args: any[]) {
    if (debugState.isDataDebugEnabled()) {
        logger.debug(message, ...args);
    }
}

export class GraphVisualization {
    private sceneManager: SceneManager;
    private nodeManager: EnhancedNodeManager;
    private edgeManager: EdgeManager;
    private hologramManager: HologramManager;
    private textRenderer: TextRenderer;
    private websocketService!: WebSocketService;

    private async initializeWebSocket(): Promise<void> {
        debugLog('Initializing WebSocket connection');
        
        // Load initial graph data
        debugLog('Loading initial graph data');
        try {
            await graphDataManager.fetchInitialData();
            // Update visualization with initial data
            const graphData = graphDataManager.getGraphData();
            this.nodeManager.updateNodes(graphData.nodes);
            this.edgeManager.updateEdges(graphData.edges);
            debugLog('Initial graph data loaded and visualization updated');
        } catch (error) {
            logger.error('Failed to load initial graph data:', error);
        }

        // Then initialize WebSocket for position updates
        this.websocketService = WebSocketService.getInstance();
        this.websocketService.onBinaryMessage((nodes) => {
            debugLog('Received binary node update', { nodeCount: nodes.length });
            this.nodeManager.updateNodePositions(nodes);
        });
        this.websocketService.onSettingsUpdate((updatedSettings) => {
            debugLog('Received settings update');
            this.handleSettingsUpdate(updatedSettings);
        });
        this.websocketService.onConnectionStatusChange((connected) => {
            logger.info(`WebSocket connection status changed: ${connected}`);
            if (connected) {
                debugLog('Requesting position updates');
                this.websocketService.sendMessage({ type: 'requestInitialData' });
            }
        });
        this.websocketService.connect();
        debugLog('WebSocket initialization complete');
    }

    constructor(settings: Settings) {
        debugLog('Initializing GraphVisualization');
        
        // Get existing canvas element
        const canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
        if (!canvas) {
            throw new Error('Could not find #main-canvas element');
        }
        
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
        debugLog('GraphVisualization initialization complete');
    }


    public handleSettingsUpdate(settings: Settings) {
        debugLog('Handling settings update');
        this.nodeManager.handleSettingsUpdate(settings);
        this.edgeManager.handleSettingsUpdate(settings);
        this.hologramManager.updateSettings(settings);
        this.textRenderer.handleSettingsUpdate(settings.visualization.labels);
        this.sceneManager.handleSettingsUpdate(settings);
    }

    public dispose() {
        debugLog('Disposing GraphVisualization');
        this.nodeManager.dispose();
        this.edgeManager.dispose();
        this.hologramManager.dispose();
        this.textRenderer.dispose();
        this.websocketService.close();
        SceneManager.cleanup();
        debugLog('GraphVisualization disposed');
    }
}

// Import default settings
import { defaultSettings } from './state/defaultSettings';

// Initialize settings and logging
async function init() {
    // Initialize settings first
    const settingsStore = SettingsStore.getInstance();
    await settingsStore.initialize();
    
    // Configure logging based on settings
    const debugEnabled = settingsStore.get('system.debug.enabled') as boolean;
    const logFullJson = settingsStore.get('system.debug.log_full_json') as boolean;
    LoggerConfig.setGlobalDebug(debugEnabled);
    LoggerConfig.setFullJson(logFullJson);
    
    // Subscribe to debug setting changes
    settingsStore.subscribe('system.debug.enabled', (_, value) => {
        LoggerConfig.setGlobalDebug(value as boolean);
    });
    settingsStore.subscribe('system.debug.log_full_json', (_, value) => {
        LoggerConfig.setFullJson(value as boolean);
    });
    
    logger.log('Starting application...');
    
    try {
        // Initialize settings first
        const settingsStore = SettingsStore.getInstance();
        await settingsStore.initialize();

        // Initialize UI with settings
        const { initializeUI } = await import('./ui');
        await initializeUI();
        
        // Initialize main visualization and store globally
        const viz = new GraphVisualization(defaultSettings);
        (window as any).visualization = viz;
        
        // Initialize platform and XR components
        await platformManager.initialize(defaultSettings);
        
        // Get canvas and scene manager
        const canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
        if (!canvas) {
            throw new Error('Could not find #main-canvas element');
        }
        const sceneManager = SceneManager.getInstance(canvas);
        
        // Initialize XR components
        const xrSessionManager = XRSessionManager.getInstance(sceneManager);
        XRInitializer.getInstance(xrSessionManager);
        
        logger.info('Application initialized successfully');
    } catch (error) {
        logger.error('Failed to initialize application components:', error);
        throw error;
    }
}

init().catch(error => {
    console.error('Failed to initialize application:', error);
});

import { platformManager } from './platform/platformManager';
import { Settings } from './types/settings';
import { setDebugEnabled } from './core/logger';
import { createLogger } from './core/logger';
import { SceneManager } from './rendering/scene';
import { WebSocketService } from './websocket/websocketService';
import { graphDataManager } from './state/graphData';
import { XRSessionManager } from './xr/xrSessionManager';
import { ControlPanel } from './ui/ControlPanel';
import { TextRenderer } from './rendering/textRenderer';
import { EnhancedNodeManager } from './rendering/EnhancedNodeManager';
import { SettingsStore } from './state/SettingsStore';

const logger = createLogger('Application');

class Application {
    private sceneManager: SceneManager | null = null;
    private nodeManager: EnhancedNodeManager | null = null;
    private xrManager: XRSessionManager | null = null;
    private textRenderer: TextRenderer | null = null;
    private controlPanel: ControlPanel | null = null;
    private settingsStore: SettingsStore;

    constructor() {
        this.settingsStore = SettingsStore.getInstance();
        this.initializeApplication();
    }

    private async initializeApplication(): Promise<void> {
        try {
            // Initialize settings store with default settings
            await this.settingsStore.initialize();
            const settings = this.settingsStore.get('') as Settings;

            // Initialize platform manager with settings
            await platformManager.initialize(settings);

            setDebugEnabled(settings.system.debug.enabled);
            logger.info(`Debug logging ${settings.system.debug.enabled ? 'enabled' : 'disabled'}`);

            const canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
            if (!canvas) {
                throw new Error('Canvas element not found');
            }
            this.sceneManager = SceneManager.getInstance(canvas);

            if (!this.sceneManager) {
                throw new Error('SceneManager not initialized');
            }
            this.textRenderer = new TextRenderer(this.sceneManager.getCamera());
            this.sceneManager.add(this.textRenderer.getGroup());

            this.nodeManager = new EnhancedNodeManager(this.sceneManager.getScene(), settings);

            const controlPanelContainer = document.getElementById('control-panel');
            if (!controlPanelContainer) {
                throw new Error('Control panel container not found');
            }
            this.controlPanel = new ControlPanel(controlPanelContainer);

            // Initialize XR manager with scene manager
            this.xrManager = XRSessionManager.getInstance(this.sceneManager);

            await graphDataManager.loadInitialGraphData();

            const webSocket = WebSocketService.getInstance();
            webSocket.onBinaryMessage((nodes: { position: [number, number, number]; velocity: [number, number, number] }[]) => {
                if (this.nodeManager) {
                    this.nodeManager.updateNodePositionsAndVelocities(nodes);
                }
            });
            webSocket.onSettingsUpdate((newSettings: Partial<Settings>) => {
                // Update settings store with new settings
                Object.entries(newSettings).forEach(([key, value]) => {
                    this.settingsStore.set(key, value);
                });
            });
            webSocket.onConnectionStatusChange((status: boolean) => {
                this.updateConnectionStatus(status);
            });
            webSocket.connect();

            // Subscribe to settings changes
            this.settingsStore.subscribe('visualization', () => {
                const currentSettings = this.settingsStore.get('') as Settings;
                if (this.nodeManager) {
                    this.nodeManager.handleSettingsUpdate(currentSettings);
                }
                if (this.sceneManager) {
                    this.sceneManager.handleSettingsUpdate(currentSettings);
                }
            });
            
            this.settingsStore.subscribe('labels', () => {
                const currentSettings = this.settingsStore.get('') as Settings;
                if (this.textRenderer) {
                    this.textRenderer.handleSettingsUpdate(currentSettings.labels);
                }
            });

            this.sceneManager.startRendering();

            this.hideLoadingOverlay();
        } catch (error) {
            logger.error('Failed to initialize application:', error);
            this.showError('Failed to initialize application. Check console for details.');
        }
    }

    private hideLoadingOverlay(): void {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay instanceof HTMLElement) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 500);
        }
    }

    private showError(message: string): void {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay instanceof HTMLElement) {
            const spinner = overlay.querySelector('.spinner');
            if (spinner) {
                spinner.remove();
            }

            const error = document.createElement('div');
            error.className = 'error-message';
            error.textContent = message;
            overlay.appendChild(error);

            overlay.style.display = 'flex';
            overlay.style.opacity = '1';
        }
    }

    public dispose(): void {
        if (this.controlPanel) {
            this.controlPanel.dispose();
        }
        if (this.sceneManager) {
            this.sceneManager.dispose();
        }
        
        const webSocket = WebSocketService.getInstance();
        if (webSocket) {
            webSocket.dispose();
        }
        
        graphDataManager.clear();
        
        if (this.xrManager) {
            this.xrManager.dispose();
        }
        
        const container = document.getElementById('scene-container');
        if (container) {
            container.innerHTML = '';
        }
        
        logger.log('Application disposed');
    }

    private updateConnectionStatus(status: boolean): void {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = status ? 'Connected' : 'Disconnected';
        }
    }
}

const app = new Application();

window.addEventListener('unload', () => {
    app.dispose();
});

console.info('LogseqXR application starting...');

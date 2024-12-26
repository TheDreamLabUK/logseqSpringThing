import { platformManager } from './platform/platformManager';
import { settingsManager } from './state/settings';
import { setDebugEnabled } from './core/logger';
import { createLogger } from './core/logger';
import { SceneManager } from './rendering/scene';
import { NodeManager } from './rendering/nodes';
import { WebSocketService } from './websocket/websocketService';
import { graphDataManager } from './state/graphData';
import { XRSessionManager } from './xr/xrSessionManager';
import { ControlPanel } from './ui/ControlPanel';

const logger = createLogger('Application');

class Application {
    private sceneManager: SceneManager | null = null;
    private nodeManager: NodeManager | null = null;
    private xrManager: XRSessionManager | null = null;

    constructor() {
        this.initializeApplication();
    }

    private async initializeApplication(): Promise<void> {
        try {
            await platformManager.initialize();
            await settingsManager.initialize();

            const settings = settingsManager.getCurrentSettings();
            setDebugEnabled(settings.system.debug.enabled);
            logger.info(`Debug logging ${settings.system.debug.enabled ? 'enabled' : 'disabled'}`);

            this.initializeScene();

            try {
                await graphDataManager.loadInitialGraphData();
                const webSocket = WebSocketService.getInstance();
                webSocket.onBinaryMessage((nodes: { position: [number, number, number]; velocity: [number, number, number] }[]) => {
                    const float32Array = new Float32Array(nodes.length * 6);
                    nodes.forEach((node, i) => {
                        const baseIndex = i * 6;
                        float32Array[baseIndex] = node.position[0];
                        float32Array[baseIndex + 1] = node.position[1];
                        float32Array[baseIndex + 2] = node.position[2];
                        float32Array[baseIndex + 3] = node.velocity[0];
                        float32Array[baseIndex + 4] = node.velocity[1];
                        float32Array[baseIndex + 5] = node.velocity[2];
                    });
                    graphDataManager.updatePositions(float32Array.buffer);

                    if (this.nodeManager) {
                        const positionsArray = new Float32Array(nodes.length * 3);
                        nodes.forEach((node, i) => {
                            const baseIndex = i * 3;
                            positionsArray[baseIndex] = node.position[0];
                            positionsArray[baseIndex + 1] = node.position[1];
                            positionsArray[baseIndex + 2] = node.position[2];
                        });
                        this.nodeManager.updatePositions(positionsArray);
                    }
                });
            } catch (error) {
                logger.error('Failed to initialize data services:', error);
                this.showError('Failed to initialize data services');
            }

            try {
                await this.initializeXR();
            } catch (xrError) {
                logger.error('Failed to initialize XR:', xrError);
            }

            const controlPanelContainer = document.getElementById('control-panel');
            if (!controlPanelContainer) {
                logger.warn('Control panel container not found, skipping UI initialization');
            } else {
                new ControlPanel(controlPanelContainer);
                this.setupUIEventListeners();
            }

            graphDataManager.subscribe(() => {
                this.hideLoadingOverlay();
            });

            logger.log('Application initialized successfully');
            this.hideLoadingOverlay();

        } catch (error) {
            logger.error('Failed to initialize application:', error);
            this.showError('Failed to initialize application');
            this.hideLoadingOverlay();
        }
    }

    private initializeScene(): void {
        const container = document.getElementById('scene-container');
        if (!container) {
            throw new Error('Scene container not found');
        }

        const canvas = document.createElement('canvas');
        container.appendChild(canvas);

        this.sceneManager = SceneManager.getInstance(canvas);
        this.nodeManager = NodeManager.getInstance();

        const nodeMeshes = this.nodeManager.getAllNodeMeshes();
        nodeMeshes.forEach(mesh => this.sceneManager?.add(mesh));

        this.sceneManager.start();
        logger.log('Scene initialized with node meshes');
    }

    private async initializeXR(): Promise<void> {
        if (platformManager.getCapabilities().xrSupported) {
            this.xrManager = XRSessionManager.getInstance(this.sceneManager!);
            const xrButton = document.getElementById('xr-button');
            if (xrButton) {
                xrButton.style.display = 'block';
                xrButton.addEventListener('click', () => this.toggleXRSession());
            }
        }
    }

    private setupUIEventListeners(): void {
        const saveButton = document.getElementById('save-settings');
        if (saveButton) {
            saveButton.addEventListener('click', () => this.saveSettings());
        }
        this.setupSettingsInputListeners();
    }

    private setupSettingsInputListeners(): void {
        // Add any settings input listeners here
    }

    private async saveSettings(): Promise<void> {
        try {
            const currentSettings = settingsManager.getCurrentSettings();
            const visualizationSettings = currentSettings.visualization;

            for (const [category, settings] of Object.entries(visualizationSettings)) {
                if (typeof settings !== 'object' || !settings) continue;

                for (const [setting, value] of Object.entries(settings)) {
                    if (typeof value === 'object' && value !== null) continue;
                    if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean' && !Array.isArray(value)) continue;

                    try {
                        const response = await fetch(`/api/settings/${String(category)}/${String(setting)}`, {
                            method: 'PUT',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ value })
                        });

                        if (!response.ok) {
                            throw new Error(`Failed to update setting: ${response.statusText}`);
                        }

                        const path = `visualization.${category}.${setting}`;
                        await settingsManager.updateSetting(path, value);
                    } catch (error) {
                        logger.error(`Failed to update setting ${String(category)}.${String(setting)}:`, error);
                    }
                }
            }
        } catch (error) {
            logger.error('Failed to save settings:', error);
            throw error;
        }
    }

    private async toggleXRSession(): Promise<void> {
        if (!this.xrManager) return;

        try {
            if (this.xrManager.isXRPresenting()) {
                await this.xrManager.endXRSession();
            } else {
                await this.xrManager.initXRSession();
            }
        } catch (error) {
            logger.error('Failed to toggle XR session:', error);
            this.showError('Failed to start XR session');
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
        if (this.sceneManager) {
            SceneManager.cleanup();
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
}

const app = new Application();

window.addEventListener('unload', () => {
    app.dispose();
});

console.info('LogseqXR application starting...');

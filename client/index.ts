/**
 * LogseqXR Application Entry Point
 */

import { platformManager } from './platform/platformManager';
import { Settings, SettingCategory, SettingKey, SettingValueType } from './types/settings';
import { settingsManager } from './state/settings';
import { graphDataManager } from './state/graphData';
import { WebSocketService } from './websocket/websocketService';
import { SceneManager } from './rendering/scene';
import { NodeManager } from './rendering/nodes';
import { XRSessionManager } from './xr/xrSessionManager';
import { createLogger, setDebugEnabled } from './utils/logger';
import { ControlPanel } from './ui/ControlPanel';

const logger = createLogger('Application');

class Application {
    private webSocket!: WebSocketService;
    private sceneManager!: SceneManager;
    private nodeManager!: NodeManager;
    private xrManager: XRSessionManager | null = null;

    constructor() {
        this.initializeApplication();
    }

    private async initializeApplication(): Promise<void> {
        try {
            // Initialize platform manager
            await platformManager.initialize();

            // Initialize settings
            await settingsManager.initialize();
            
            // Update logger debug state from settings
            const settings = settingsManager.getCurrentSettings();
            setDebugEnabled(settings.clientDebug.enabled);
            logger.info('Debug logging ' + (settings.clientDebug.enabled ? 'enabled' : 'disabled'));

            // Initialize scene first so we can render nodes when data arrives
            this.initializeScene();

            // Track initialization state
            let graphDataLoaded = false;
            let websocketConnected = false;
            let binaryUpdatesEnabled = false;

            // Function to check if we can hide loading overlay
            const checkInitComplete = () => {
                if (graphDataLoaded && websocketConnected && binaryUpdatesEnabled) {
                    this.hideLoadingOverlay();
                    logger.info('All initialization checks passed, application ready');
                }
            };

            try {
                // Load initial graph data from REST endpoint
                await graphDataManager.loadInitialGraphData();
                graphDataLoaded = true;
                checkInitComplete();
            } catch (graphError) {
                logger.error('Failed to load graph data:', graphError);
                // Continue initialization even if graph data fails
            }

            try {
                // Initialize WebSocket for real-time updates
                this.webSocket = new WebSocketService(settingsManager);

                // Setup WebSocket event handlers
                this.webSocket.onMessage('connectionStatus', (data: { status: string, details?: any }) => {
                    logger.info('WebSocket connection status:', data);
                    if (data.status === 'CONNECTED') {
                        websocketConnected = true;
                        checkInitComplete();
                    }
                });

                this.webSocket.onMessage('enableBinaryUpdates', (data: { enabled: boolean }) => {
                    logger.info('Binary updates status:', data);
                    // Update graph data manager binary state
                    graphDataManager.setBinaryUpdatesEnabled(data.enabled);
                    if (data.enabled) {
                        binaryUpdatesEnabled = true;
                        checkInitComplete();
                        // Request initial data after binary updates are enabled
                        this.webSocket.send(JSON.stringify({
                            type: 'requestInitialData'
                        }));
                    }
                });

                // Setup WebSocket event handler for binary position updates
                this.webSocket.onMessage('binaryPositionUpdate', (data: any['data']) => {
                    if (data && data.nodes) {
                        // Convert nodes data to ArrayBuffer for position updates
                        const buffer = new ArrayBuffer(data.nodes.length * 24); // 6 floats per node
                        const floatArray = new Float32Array(buffer);
                        
                        data.nodes.forEach((node: { data: { position: any; velocity: any } }, index: number) => {
                            const baseIndex = index * 6;
                            const pos = node.data.position;
                            const vel = node.data.velocity;
                            
                            // Position
                            floatArray[baseIndex] = pos.x;
                            floatArray[baseIndex + 1] = pos.y;
                            floatArray[baseIndex + 2] = pos.z;
                            // Velocity
                            floatArray[baseIndex + 3] = vel.x;
                            floatArray[baseIndex + 4] = vel.y;
                            floatArray[baseIndex + 5] = vel.z;
                        });

                        // Update graph data and visual representation
                        graphDataManager.updatePositions(buffer);
                        this.nodeManager.updatePositions(floatArray);
                    }
                });
            } catch (wsError) {
                logger.error('Failed to initialize WebSocket:', wsError);
                // Continue initialization even if WebSocket fails
            }

            try {
                // Initialize XR if supported
                await this.initializeXR();
            } catch (xrError) {
                logger.error('Failed to initialize XR:', xrError);
                // Continue initialization even if XR fails
            }

            // Initialize UI components
            const controlPanelContainer = document.getElementById('control-panel');
            if (!controlPanelContainer) {
                logger.warn('Control panel container not found, skipping UI initialization');
            } else {
                new ControlPanel(controlPanelContainer);
                // Setup UI event listeners
                this.setupUIEventListeners();
            }

            // Subscribe to graph data updates
            graphDataManager.subscribe(() => {
                // Hide loading overlay after initial data is loaded
                this.hideLoadingOverlay();
            });

            logger.log('Application initialized successfully');
            // Always hide loading overlay after initialization
            this.hideLoadingOverlay();
        } catch (error) {
            logger.error('Failed to initialize application:', error);
            this.showError('Failed to initialize application');
            // Still try to hide loading overlay
            this.hideLoadingOverlay();
        }
    }

    private initializeScene(): void {
        // Get canvas element
        const container = document.getElementById('scene-container');
        if (!container) {
            throw new Error('Scene container not found');
        }

        // Create canvas
        const canvas = document.createElement('canvas');
        container.appendChild(canvas);

        // Initialize scene manager
        this.sceneManager = SceneManager.getInstance(canvas);

        // Initialize node manager
        this.nodeManager = NodeManager.getInstance();

        // Add node meshes to scene
        const nodeMeshes = this.nodeManager.getAllNodeMeshes();
        nodeMeshes.forEach(mesh => this.sceneManager.add(mesh));

        // Start rendering
        this.sceneManager.start();
        logger.log('Scene initialized with node meshes');
    }

    private async initializeXR(): Promise<void> {
        if (platformManager.getCapabilities().xrSupported) {
            // Initialize XR manager
            this.xrManager = XRSessionManager.getInstance(this.sceneManager);

            // Setup XR button
            const xrButton = document.getElementById('xr-button');
            if (xrButton) {
                xrButton.style.display = 'block';
                xrButton.addEventListener('click', () => this.toggleXRSession());
            }
        }
    }

    private setupUIEventListeners(): void {
        // Settings panel save button
        const saveButton = document.getElementById('save-settings');
        if (saveButton) {
            saveButton.addEventListener('click', () => this.saveSettings());
        }

        // Settings inputs
        this.setupSettingsInputListeners();
    }

    private setupSettingsInputListeners(): void {
        // Node appearance settings
        this.setupSettingInput<'nodes', 'baseSize'>('nodes', 'baseSize');
        this.setupSettingInput<'nodes', 'baseColor'>('nodes', 'baseColor');
        this.setupSettingInput<'nodes', 'opacity'>('nodes', 'opacity');

        // Edge appearance settings
        this.setupSettingInput<'edges', 'color'>('edges', 'color');
        this.setupSettingInput<'edges', 'opacity'>('edges', 'opacity');
        this.setupSettingInput<'edges', 'enableArrows'>('edges', 'enableArrows');

        // Visual effects settings
        this.setupSettingInput<'bloom', 'edgeBloomStrength'>('bloom', 'edgeBloomStrength');

        // Physics settings
        this.setupSettingInput<'physics', 'enabled'>('physics', 'enabled');
        this.setupSettingInput<'physics', 'springStrength'>('physics', 'springStrength');
    }

    private setupSettingInput<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K
    ): void {
        const input = document.getElementById(`${String(category)}-${String(setting)}`) as HTMLInputElement;
        if (input) {
            input.addEventListener('change', async (event) => {
                const currentValue = (event.target as HTMLInputElement).value;

                try {
                    const response = await fetch(`/api/visualization/settings/${String(category)}/${String(setting)}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ value: currentValue }),
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    await settingsManager.updateSetting(
                        category,
                        setting,
                        this.parseSettingValue<T, K>(currentValue, category, setting)
                    );
                } catch (error) {
                    logger.error(`Failed to update setting ${String(category)}.${String(setting)}:`, error);
                    this.showError(`Failed to update ${String(category)} ${String(setting)}`);
                }
            });
        }
    }

    private parseSettingValue<T extends SettingCategory, K extends SettingKey<T>>(
        value: string,
        category: T,
        setting: K
    ): SettingValueType<T, K> {
        const currentSettings = settingsManager.getCurrentSettings();
        const currentValue = currentSettings[category][setting];
        
        switch (typeof currentValue) {
            case 'number':
                return Number(value) as SettingValueType<T, K>;
            case 'boolean':
                return (value === 'true') as SettingValueType<T, K>;
            default:
                return value as SettingValueType<T, K>;
        }
    }

    private async saveSettings(): Promise<void> {
        try {
            const currentSettings = settingsManager.getCurrentSettings();
            const categories = ['nodes', 'edges', 'rendering', 'physics', 'labels', 'bloom', 'clientDebug'] as const;
            
            for (const category of categories) {
                const categorySettings = currentSettings[category];
                for (const [setting, value] of Object.entries(categorySettings)) {
                    try {
                        const response = await fetch(`/api/visualization/settings/${String(category)}/${String(setting)}`, {
                            method: 'PUT',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ value })
                        });

                        if (!response.ok) {
                            throw new Error(`Failed to update setting: ${response.statusText}`);
                        }

                        await settingsManager.updateSetting(
                            category,
                            setting as keyof Settings[typeof category],
                            value as SettingValueType<typeof category, keyof Settings[typeof category]>
                        );
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
        if (overlay && overlay instanceof HTMLElement) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 500); // Match this with CSS transition duration
        }
    }

    private showError(message: string): void {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay && overlay instanceof HTMLElement) {
            const spinner = overlay.querySelector('.spinner');
            if (spinner) {
                spinner.remove();
            }
            
            const error = document.createElement('div');
            error.className = 'error-message';
            error.textContent = message;
            overlay.appendChild(error);
            
            // Keep the overlay visible
            overlay.style.display = 'flex';
            overlay.style.opacity = '1';
        }
    }

    dispose(): void {
        // Stop rendering
        if (this.sceneManager) {
            SceneManager.cleanup();
        }

        // Dispose of WebSocket
        if (this.webSocket) {
            this.webSocket.dispose();
        }

        // Clear graph data
        graphDataManager.clear();

        // Dispose of XR
        if (this.xrManager) {
            this.xrManager.dispose();
        }

        // Remove canvas
        const container = document.getElementById('scene-container');
        if (container) {
            container.innerHTML = '';
        }

        logger.log('Application disposed');
    }
}

// Create application instance
const app = new Application();

// Handle window unload
window.addEventListener('unload', () => {
    app.dispose();
});

// Log application start
console.info('LogseqXR application starting...');

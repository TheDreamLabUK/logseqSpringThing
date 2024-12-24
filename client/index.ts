/**
 * LogseqXR Application Entry Point
 */

import { platformManager } from './platform/platformManager';
import { Settings } from './types/settings';
import { SettingValue } from './types/settings/utils';
import { settingsManager } from './state/settings';
import { graphDataManager } from './state/graphData';
import { WebSocketService } from './websocket/websocketService';
import { SceneManager } from './rendering/scene';
import { NodeManager } from './rendering/nodes';
import { XRSessionManager } from './xr/xrSessionManager';
import { createLogger, setDebugEnabled } from './core/logger';
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
            setDebugEnabled(settings.system.debug.enabled);
            logger.info('Debug logging ' + (settings.system.debug.enabled ? 'enabled' : 'disabled'));

            // Initialize scene first so we can render nodes when data arrives
            this.initializeScene();

            try {
                // Load initial graph data from REST endpoint
                await graphDataManager.loadInitialGraphData();
                
                // Initialize WebSocket for real-time position updates
                this.webSocket = WebSocketService.getInstance();

                // Setup binary position update handler
                this.webSocket.onBinaryMessage((nodes) => {
                    // Convert NodeData[] to ArrayBuffer for graph data manager
                    const float32Array = new Float32Array(nodes.length * 6); // 6 floats per node (position + velocity)
                    nodes.forEach((node, i) => {
                        const baseIndex = i * 6;
                        // Position
                        float32Array[baseIndex] = node.position[0];
                        float32Array[baseIndex + 1] = node.position[1];
                        float32Array[baseIndex + 2] = node.position[2];
                        // Velocity
                        float32Array[baseIndex + 3] = node.velocity[0];
                        float32Array[baseIndex + 4] = node.velocity[1];
                        float32Array[baseIndex + 5] = node.velocity[2];
                    });

                    // Update graph data with positions as ArrayBuffer
                    graphDataManager.updatePositions(float32Array.buffer);
                    
                    // Create Float32Array for node manager with just positions
                    const positionsArray = new Float32Array(nodes.length * 3); // 3 floats per position
                    nodes.forEach((node, i) => {
                        const baseIndex = i * 3;
                        positionsArray[baseIndex] = node.position[0];
                        positionsArray[baseIndex + 1] = node.position[1];
                        positionsArray[baseIndex + 2] = node.position[2];
                    });
                    
                    // Update visual representation with Float32Array
                    this.nodeManager.updatePositions(positionsArray);
                });

            } catch (error) {
                logger.error('Failed to initialize data services:', error);
                this.showError('Failed to initialize data services');
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
            // Hide loading overlay after initialization
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
        const visualizationSettings = {
            nodes: ['baseSize', 'baseColor', 'opacity'],
            edges: ['color', 'opacity', 'enableArrows'],
            bloom: ['edgeBloomStrength'],
            physics: ['enabled', 'springStrength']
        } as const;

        for (const [category, settings] of Object.entries(visualizationSettings)) {
            for (const setting of settings) {
                this.setupSettingInput(category, setting);
            }
        }
    }

    private setupSettingInput(
        category: string,
        setting: string
    ): void {
        const input = document.getElementById(`${String(category)}-${String(setting)}`) as HTMLInputElement;
        if (input) {
            input.addEventListener('change', async (event) => {
                const currentValue = (event.target as HTMLInputElement).value;

                try {
                    const response = await fetch(`/api/settings/${String(category)}/${String(setting)}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ value: currentValue }),
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const path = `visualization.${category}.${setting}` as const;
                    const parsedValue = this.parseSettingValue(currentValue, category, setting);
                    await settingsManager.updateSetting(path, parsedValue);
                } catch (error) {
                    logger.error(`Failed to update setting ${String(category)}.${String(setting)}:`, error);
                    this.showError(`Failed to update ${String(category)} ${String(setting)}`);
                }
            });
        }
    }

    private parseSettingValue(
        value: string,
        category: string,
        setting: string
    ): any {
        const currentSettings = settingsManager.getCurrentSettings();
        const categorySettings = currentSettings.visualization[category as keyof Settings['visualization']];
        if (!categorySettings || typeof categorySettings !== 'object') {
            throw new Error(`Invalid category: ${category}`);
        }
        
        const currentValue = (categorySettings as any)[setting];
        
        if (currentValue === undefined) {
            throw new Error(`Invalid setting path: visualization.${category}.${setting}`);
        }
        
        switch (typeof currentValue) {
            case 'number':
                return Number(value);
            case 'boolean':
                return value === 'true';
            default:
                return value;
        }
    }

    private async saveSettings(): Promise<void> {
        try {
            const currentSettings = settingsManager.getCurrentSettings();
            const visualizationSettings = currentSettings.visualization;
            
            for (const [category, settings] of Object.entries(visualizationSettings)) {
                if (typeof settings !== 'object' || !settings) continue;
                
                for (const [setting, value] of Object.entries(settings)) {
                    // Skip if value is an object (nested settings)
                    if (typeof value === 'object' && value !== null) continue;
                    
                    // Only process primitive values that match SettingValue type
                    if (typeof value !== 'string' && 
                        typeof value !== 'number' && 
                        typeof value !== 'boolean' && 
                        !Array.isArray(value)) continue;
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

                        const path = `visualization.${category}.${setting}` as const;
                        await settingsManager.updateSetting(path, value as SettingValue);
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

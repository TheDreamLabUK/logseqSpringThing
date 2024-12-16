/**
 * LogseqXR Application Entry Point
 */

import { platformManager } from './platform/platformManager';
import { Settings, SettingsCategory, SettingKey, SettingValueType } from './types/settings';
import { settingsManager } from './state/settings';
import { graphDataManager } from './state/graphData';
import { WebSocketService } from './websocket/websocketService';
import { SceneManager } from './rendering/scene';
import { NodeManager } from './rendering/nodes';
import { TextRenderer } from './rendering/textRenderer';
import { XRSessionManager } from './xr/xrSessionManager';
import { XRInteraction } from './xr/xrInteraction';
import { createLogger } from './utils/logger';
import { ControlPanel } from './ui/ControlPanel';

const logger = createLogger('Application');

class Application {
    private webSocket!: WebSocketService;
    private sceneManager!: SceneManager;
    private nodeManager!: NodeManager;
    private textRenderer!: TextRenderer;
    private xrManager: XRSessionManager | null = null;
    private xrInteraction: XRInteraction | null = null;

    constructor() {
        this.initializeApplication();
    }

    private async initializeApplication(): Promise<void> {
        try {
            // Initialize platform manager
            await platformManager.initialize();

            // Initialize settings
            await settingsManager.initialize();

            // Initialize scene first so we can render nodes when data arrives
            this.initializeScene();

            // Load initial graph data from REST endpoint
            await graphDataManager.loadInitialGraphData();

            // Initialize WebSocket for real-time updates
            this.webSocket = new WebSocketService();

            // Setup WebSocket event handler for binary position updates
            this.webSocket.on('binaryPositionUpdate', (data: any['data']) => {
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

            // Initialize XR if supported
            await this.initializeXR();

            // Initialize UI components
            const controlPanelContainer = document.getElementById('control-panel');
            if (!controlPanelContainer) {
                throw new Error('Control panel container not found');
            }
            new ControlPanel(controlPanelContainer);

            // Setup UI event listeners
            this.setupUIEventListeners();

            // Hide loading overlay
            this.hideLoadingOverlay();

            logger.log('Application initialized successfully');
        } catch (error) {
            logger.error('Failed to initialize application:', error);
            this.showError('Failed to initialize application');
        }
    }

    private initializeScene(): void {
        // Get canvas element
        const container = document.getElementById('canvas-container');
        if (!container) {
            throw new Error('Canvas container not found');
        }

        // Create canvas
        const canvas = document.createElement('canvas');
        container.appendChild(canvas);

        // Initialize scene manager
        this.sceneManager = SceneManager.getInstance(canvas);

        // Initialize node manager
        this.nodeManager = NodeManager.getInstance(this.sceneManager);

        // Initialize text renderer
        this.textRenderer = new TextRenderer(this.sceneManager.getCamera());

        // Start rendering
        this.sceneManager.start();
    }

    private async initializeXR(): Promise<void> {
        if (platformManager.getCapabilities().xrSupported) {
            // Initialize XR manager
            this.xrManager = XRSessionManager.getInstance(this.sceneManager);

            // Initialize XR interaction
            if (this.xrManager && this.nodeManager) {
                this.xrInteraction = XRInteraction.getInstance(this.xrManager, this.nodeManager);
            }

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
        this.setupSettingInput<'edges', 'baseWidth'>('edges', 'baseWidth');
        this.setupSettingInput<'edges', 'baseColor'>('edges', 'baseColor');
        this.setupSettingInput<'edges', 'opacity'>('edges', 'opacity');

        // Visual effects settings
        this.setupSettingInput<'bloom', 'edgeBloomStrength'>('bloom', 'edgeBloomStrength');

        // Physics settings
        this.setupSettingInput<'physics', 'enabled'>('physics', 'enabled');
        this.setupSettingInput<'physics', 'springStrength'>('physics', 'springStrength');
    }

    private setupSettingInput<T extends SettingsCategory, K extends SettingKey<T>>(
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

    private parseSettingValue<T extends SettingsCategory, K extends SettingKey<T>>(
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
        if (overlay) {
            overlay.remove();
        }
    }

    private showError(message: string): void {
        logger.error(message);
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            z-index: 1000;
        `;
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    dispose(): void {
        // Dispose of managers in reverse order of initialization
        settingsManager.dispose();
        this.xrInteraction?.dispose();
        this.xrManager?.dispose();
        this.textRenderer.dispose();
        this.nodeManager.dispose();
        this.sceneManager.dispose();

        // Stop rendering
        this.sceneManager.stop();

        // Close WebSocket connection if it exists
        if (this.webSocket) {
            this.webSocket.dispose();
        }
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

/**
 * LogseqXR Application Entry Point
 */

import { platformManager } from './platform/platformManager';
import { settingsManager } from './state/settings';
import { graphDataManager } from './state/graphData';
import { WebSocketService } from './websocket/websocketService';
import { SceneManager } from './rendering/scene';
import { NodeManager } from './rendering/nodes';
import { TextRenderer } from './rendering/textRenderer';
import { XRSessionManager } from './xr/xrSessionManager';
import { XRInteraction } from './xr/xrInteraction';
import { createLogger } from './utils/logger';
import { WS_URL } from './core/constants';
import { BinaryPositionUpdateMessage, Settings, SettingValue } from './core/types';
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

            // Initialize scene first so we can render nodes when data arrives
            this.initializeScene();

            // Load initial graph data from REST endpoint
            await graphDataManager.loadInitialGraphData();
            
            // Load settings from REST endpoint
            await this.loadSettings();

            // Initialize WebSocket for real-time updates
            this.initializeWebSocket();

            // Initialize XR if supported
            await this.initializeXR();

            // Initialize UI components after settings are loaded
            new ControlPanel();

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

    private async loadSettings(): Promise<void> {
        try {
            // Load all settings at once using the REST endpoint
            await settingsManager.loadAllSettings();
        } catch (error) {
            logger.error('Failed to load settings:', error);
            logger.info('Continuing with default settings');
        }
    }

    private initializeWebSocket(): void {
        // Create WebSocket service with environment-aware URL
        this.webSocket = new WebSocketService(WS_URL);

        // Setup WebSocket event handler for binary position updates
        this.webSocket.on('binaryPositionUpdate', (data: BinaryPositionUpdateMessage['data']) => {
            if (data && data.nodes) {
                // Convert nodes data to ArrayBuffer for position updates
                const buffer = new ArrayBuffer(data.nodes.length * 24); // 6 floats per node
                const floatArray = new Float32Array(buffer);
                
                data.nodes.forEach((node, index) => {
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

        // Connect to server
        this.webSocket.connect();
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
        this.textRenderer = TextRenderer.getInstance(
            this.sceneManager.getScene(),
            this.sceneManager.getCamera()
        );

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
        this.setupSettingInput('nodes', 'baseSize', 'number');
        this.setupSettingInput('nodes', 'baseColor', 'color');
        this.setupSettingInput('nodes', 'opacity', 'number');

        // Edge appearance settings
        this.setupSettingInput('edges', 'baseWidth', 'number');
        this.setupSettingInput('edges', 'color', 'color');
        this.setupSettingInput('edges', 'opacity', 'number');

        // Visual effects settings
        this.setupSettingInput('bloom', 'enabled', 'checkbox');
        this.setupSettingInput('bloom', 'strength', 'number');
    }

    private setupSettingInput(
        category: keyof Settings,
        setting: string,
        type: 'number' | 'color' | 'checkbox'
    ): void {
        const input = document.getElementById(`${category}-${setting}`) as HTMLInputElement;
        if (input) {
            input.addEventListener('change', async () => {
                let currentValue: SettingValue;
                let previousValue: SettingValue;

                if (type === 'checkbox') {
                    previousValue = input.checked;
                    currentValue = input.checked;
                } else if (type === 'number') {
                    previousValue = input.valueAsNumber;
                    currentValue = input.valueAsNumber;
                } else {
                    previousValue = input.value;
                    currentValue = input.value;
                }

                try {
                    await settingsManager.updateSetting(category, setting, currentValue);
                } catch (error) {
                    logger.error(`Failed to update setting ${category}.${setting}:`, error);
                    // Revert UI on error
                    if (type === 'checkbox') {
                        input.checked = previousValue as boolean;
                    } else {
                        input.value = String(previousValue);
                    }
                    this.showError(`Failed to update ${category} ${setting}`);
                }
            });
        }
    }

    private async saveSettings(): Promise<void> {
        try {
            const currentSettings = settingsManager.getCurrentSettings();
            await settingsManager.updateAllSettings(currentSettings);
            logger.log('Settings saved successfully');
        } catch (error) {
            logger.error('Failed to save settings:', error);
            this.showError('Failed to save settings');
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
            this.webSocket.disconnect();
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

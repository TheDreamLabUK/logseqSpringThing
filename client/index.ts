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
import { createLogger } from './core/utils';
import { WS_URL, IS_PRODUCTION } from './core/constants';

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

      // Initialize WebSocket connection
      this.initializeWebSocket();

      // Initialize scene
      this.initializeScene();

      // Initialize settings
      await this.initializeSettings();

      // Initialize XR if supported
      await this.initializeXR();

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

  private initializeWebSocket(): void {
    // Create WebSocket service with environment-aware URL
    this.webSocket = new WebSocketService(WS_URL);

    // Setup WebSocket event handlers
    this.webSocket.on('initialData', (data) => {
      graphDataManager.updateGraphData(data);
    });

    this.webSocket.on('graphUpdate', (data) => {
      graphDataManager.updateGraphData(data);
    });

    this.webSocket.on('binaryPositionUpdate', (data) => {
      graphDataManager.updatePositions(data);
    });

    this.webSocket.on('error', (error) => {
      logger.error('WebSocket error:', error);
      this.showError('Connection error');
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

  private async initializeSettings(): Promise<void> {
    try {
      // Load settings from server
      await settingsManager.loadSettings();

      // Update UI with current settings
      this.updateSettingsUI();
    } catch (error) {
      logger.error('Failed to load settings:', error);
      // Continue with default settings
    }
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
    this.setupSettingInput('nodeSize', 'number');
    this.setupSettingInput('nodeColor', 'color');
    this.setupSettingInput('nodeOpacity', 'number');

    // Edge appearance settings
    this.setupSettingInput('edgeWidth', 'number');
    this.setupSettingInput('edgeColor', 'color');
    this.setupSettingInput('edgeOpacity', 'number');

    // Visual effects settings
    this.setupSettingInput('enableBloom', 'checkbox');
    this.setupSettingInput('bloomIntensity', 'number');
  }

  private setupSettingInput(id: string, type: 'number' | 'color' | 'checkbox'): void {
    const input = document.getElementById(id) as HTMLInputElement;
    if (input) {
      input.addEventListener('change', () => {
        const value = type === 'checkbox' ? input.checked :
                     type === 'number' ? parseFloat(input.value) :
                     input.value;
        settingsManager.updateSettings({ [id]: value });
      });
    }
  }

  private updateSettingsUI(): void {
    const settings = settingsManager.getSettings();
    Object.entries(settings).forEach(([key, value]) => {
      const input = document.getElementById(key) as HTMLInputElement;
      if (input) {
        if (input.type === 'checkbox') {
          input.checked = value as boolean;
        } else {
          input.value = value.toString();
        }
      }
    });
  }

  private async saveSettings(): Promise<void> {
    try {
      await settingsManager.saveSettings();
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
    // TODO: Implement proper error UI
    alert(message);
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    // Stop rendering
    this.sceneManager.stop();

    // Dispose of managers
    this.nodeManager.dispose();
    this.textRenderer.dispose();
    this.xrManager?.dispose();
    this.xrInteraction?.dispose();
    this.sceneManager.dispose();

    // Close WebSocket connection
    this.webSocket.disconnect();
  }
}

// Prevent context menu in production
if (!IS_PRODUCTION) {
  document.addEventListener('contextmenu', event => event.preventDefault());
}

// Create application instance
const app = new Application();

// Handle window unload
window.addEventListener('unload', () => {
  app.dispose();
});

// Log application start
console.info('LogseqXR application starting...');

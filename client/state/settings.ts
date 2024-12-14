/**
 * Settings management with simplified visualization configuration
 */

import { VisualizationSettings } from '../core/types';
import { DEFAULT_VISUALIZATION_SETTINGS } from '../core/constants';
import { createLogger } from '../core/utils';
import { WebSocketService } from '../websocket/websocketService';

const logger = createLogger('SettingsManager');

export class SettingsManager {
  private static instance: SettingsManager;
  private settings: VisualizationSettings;
  private settingsListeners: Set<(settings: VisualizationSettings) => void>;
  private webSocket: WebSocketService;

  private constructor(webSocket: WebSocketService) {
    this.settings = { ...DEFAULT_VISUALIZATION_SETTINGS };
    this.settingsListeners = new Set();
    this.webSocket = webSocket;

    // Listen for settings updates from server
    this.webSocket.on('settingsUpdated', (data) => {
      if (data && data.settings) {
        this.settings = data.settings;
        this.notifyListeners();
      }
    });

    logger.log('Initialized with default settings');
  }

  static getInstance(webSocket: WebSocketService): SettingsManager {
    if (!SettingsManager.instance) {
      SettingsManager.instance = new SettingsManager(webSocket);
    }
    return SettingsManager.instance;
  }

  /**
   * Load settings from the server via WebSocket
   */
  async loadSettings(): Promise<void> {
    // Settings will be received through the settingsUpdated WebSocket message
    // No need to explicitly request them as they're sent with initial data
    logger.log('Settings will be received through WebSocket');
  }

  /**
   * Save current settings to the server via WebSocket
   */
  async saveSettings(): Promise<void> {
    try {
      this.webSocket.send({
        type: 'updateSettings',
        data: {
          settings: this.settings
        }
      });
      logger.log('Settings update sent through WebSocket');
    } catch (error) {
      logger.error('Error sending settings update:', error);
      throw error;
    }
  }

  /**
   * Update settings and notify listeners
   */
  updateSettings(newSettings: Partial<VisualizationSettings>): void {
    this.settings = {
      ...this.settings,
      ...newSettings
    };

    logger.log('Updated settings locally');
    this.notifyListeners();
  }

  private notifyListeners(): void {
    this.settingsListeners.forEach(listener => {
      try {
        listener(this.settings);
      } catch (error) {
        logger.error('Error in settings listener:', error);
      }
    });
  }

  /**
   * Get current settings
   */
  getSettings(): VisualizationSettings {
    return { ...this.settings };
  }

  /**
   * Subscribe to settings changes
   */
  subscribe(listener: (settings: VisualizationSettings) => void): () => void {
    this.settingsListeners.add(listener);
    return () => {
      this.settingsListeners.delete(listener);
    };
  }

  /**
   * Reset settings to defaults
   */
  resetToDefaults(): void {
    this.updateSettings(DEFAULT_VISUALIZATION_SETTINGS);
  }

  // Essential setting getters
  getNodeSettings(): {
    size: number;
    color: string;
    opacity: number;
    highlightColor: string;
  } {
    return {
      size: this.settings.nodeSize,
      color: this.settings.nodeColor,
      opacity: this.settings.nodeOpacity,
      highlightColor: this.settings.nodeHighlightColor
    };
  }

  getEdgeSettings(): {
    width: number;
    color: string;
    opacity: number;
  } {
    return {
      width: this.settings.edgeWidth,
      color: this.settings.edgeColor,
      opacity: this.settings.edgeOpacity
    };
  }

  getBloomSettings(): {
    enabled: boolean;
    intensity: number;
    threshold: number;
    radius: number;
  } {
    return {
      enabled: this.settings.enableBloom,
      intensity: this.settings.bloomIntensity,
      threshold: this.settings.bloomThreshold,
      radius: this.settings.bloomRadius
    };
  }

  getLabelSettings(): {
    show: boolean;
    size: number;
    color: string;
  } {
    return {
      show: this.settings.showLabels,
      size: this.settings.labelSize,
      color: this.settings.labelColor
    };
  }

  getXRSettings(): {
    controllerVibration: boolean;
    hapticIntensity: number;
  } {
    return {
      controllerVibration: this.settings.xrControllerVibration,
      hapticIntensity: this.settings.xrControllerHapticIntensity
    };
  }

  getPerformanceSettings(): {
    maxFps: number;
    updateThrottle: number;
  } {
    return {
      maxFps: this.settings.maxFps,
      updateThrottle: this.settings.updateThrottle
    };
  }
}

// Export a singleton instance
// Note: This will be initialized with the WebSocket instance in index.ts
export let settingsManager: SettingsManager;

export function initializeSettingsManager(webSocket: WebSocketService): void {
  settingsManager = SettingsManager.getInstance(webSocket);
}

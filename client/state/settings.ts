/**
 * Settings management with simplified visualization configuration
 */

import { VisualizationSettings } from '../core/types';
import { createLogger } from '../core/utils';
import { WebSocketService } from '../websocket/websocketService';

const logger = createLogger('SettingsManager');

export const DEFAULT_VISUALIZATION_SETTINGS: VisualizationSettings = {
    // Node Appearance
    nodeSize: 0.2,
    nodeColor: '#FFB700',
    nodeOpacity: 0.92,
    metalness: 0.85,
    roughness: 0.15,
    clearcoat: 1.0,

    // Edge Appearance
    edgeWidth: 2.0,
    edgeColor: '#FFD700',
    edgeOpacity: 0.6,
    enableArrows: true,
    arrowSize: 0.15,

    // Visual Effects
    enableBloom: true,
    bloomIntensity: 1.8,
    bloomRadius: 0.5,
    enableNodeAnimations: true,
    enableMotionBlur: true,
    motionBlurStrength: 0.4,

    // Labels
    showLabels: true,
    labelSize: 1.0,
    labelColor: '#FFFFFF',

    // Performance
    maxFps: 60,

    // AR Settings (Meta Quest 3)
    // Scene Understanding
    enablePlaneDetection: true,
    enableSceneUnderstanding: true,
    showPlaneOverlay: true,
    planeOpacity: 0.3,
    planeColor: '#4A90E2',
    enableLightEstimation: true,
    
    // Hand Tracking
    enableHandTracking: true,
    handMeshEnabled: true,
    handMeshColor: '#FFD700',
    handMeshOpacity: 0.3,
    handRayEnabled: true,
    handRayColor: '#FFD700',
    handRayWidth: 0.002,
    handPointSize: 0.01,
    
    // Gesture Controls
    gestureSmoothing: 0.9,
    pinchThreshold: 0.015,
    dragThreshold: 0.04,
    rotationThreshold: 0.08,
    
    // Haptics
    enableHaptics: true,
    hapticIntensity: 0.7,
    
    // Room Scale
    roomScale: true,
    snapToFloor: true,
    
    // Passthrough
    passthroughOpacity: 1.0,
    passthroughBrightness: 1.0,
    passthroughContrast: 1.0,
    enablePassthroughPortal: false,
    portalSize: 1.0,
    portalEdgeColor: '#FFD700',
    portalEdgeWidth: 0.02
};

export class SettingsManager {
  private static instance: SettingsManager | null = null;
  private settings: VisualizationSettings;
  private settingsListeners: Set<(settings: VisualizationSettings) => void>;
  private webSocket: WebSocketService | null = null;

  private constructor() {
    this.settings = { ...DEFAULT_VISUALIZATION_SETTINGS };
    this.settingsListeners = new Set();
    logger.log('Initialized with default settings');
  }

  static getInstance(): SettingsManager {
    if (!SettingsManager.instance) {
      SettingsManager.instance = new SettingsManager();
    }
    return SettingsManager.instance;
  }

  /**
   * Initialize WebSocket connection
   */
  initializeWebSocket(webSocket: WebSocketService): void {
    this.webSocket = webSocket;

    // Listen for settings updates from server
    this.webSocket.on('settingsUpdated', (data) => {
      if (data && data.settings) {
        this.settings = data.settings;
        this.notifyListeners();
      }
    });

    logger.log('WebSocket initialized for settings');
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
    if (!this.webSocket) {
      throw new Error('WebSocket not initialized');
    }

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
    
    // Send update to server if WebSocket is available
    if (this.webSocket) {
      this.saveSettings().catch(error => {
        logger.error('Failed to save settings to server:', error);
      });
    }
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
   * Add a settings update listener
   */
  addSettingsListener(listener: (settings: VisualizationSettings) => void): void {
    this.settingsListeners.add(listener);
  }

  /**
   * Remove a settings update listener
   */
  removeSettingsListener(listener: (settings: VisualizationSettings) => void): void {
    this.settingsListeners.delete(listener);
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

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.webSocket) {
      // Remove WebSocket listeners
      this.webSocket.off('settingsUpdated', this.notifyListeners);
      this.webSocket = null;
    }
    // Clear all listeners
    this.settingsListeners.clear();
    SettingsManager.instance = null;
  }

  public getThreeJSSettings() {
    return {
      nodes: {
        size: this.settings.nodeSize,
        color: this.settings.nodeColor,
        opacity: this.settings.nodeOpacity,
        metalness: this.settings.metalness,
        roughness: this.settings.roughness,
        clearcoat: this.settings.clearcoat,
        highlightColor: '#FFFFFF' // Default highlight color
      },
      edges: {
        width: this.settings.edgeWidth,
        color: this.settings.edgeColor,
        opacity: this.settings.edgeOpacity,
        arrows: {
          enabled: this.settings.enableArrows,
          size: this.settings.arrowSize
        }
      },
      bloom: {
        enabled: this.settings.enableBloom,
        intensity: this.settings.bloomIntensity,
        radius: this.settings.bloomRadius
      },
      animations: {
        enabled: this.settings.enableNodeAnimations,
        motionBlur: {
          enabled: this.settings.enableMotionBlur,
          strength: this.settings.motionBlurStrength
        }
      },
      labels: {
        enabled: this.settings.showLabels,
        size: this.settings.labelSize,
        color: this.settings.labelColor
      },
      performance: {
        maxFps: this.settings.maxFps
      }
    };
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
      highlightColor: '#FFFFFF' // Default highlight color
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
      threshold: 0.5, // Default threshold
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
      controllerVibration: false, // Default controller vibration
      hapticIntensity: 0.5 // Default haptic intensity
    };
  }

  getPerformanceSettings(): {
    maxFps: number;
    updateThrottle: number;
  } {
    return {
      maxFps: this.settings.maxFps,
      updateThrottle: 0 // Default update throttle
    };
  }
}

// Export singleton instance and initialization function
export const settingsManager = SettingsManager.getInstance();

export function initializeSettingsManager(webSocket: WebSocketService): void {
  settingsManager.initializeWebSocket(webSocket);
}
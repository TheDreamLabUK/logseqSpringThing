/**
 * Settings management with simplified visualization configuration
 */

import { VisualizationSettings } from '../core/types';
import { DEFAULT_VISUALIZATION_SETTINGS, IS_PRODUCTION } from '../core/constants';
import { createLogger } from '../core/utils';

const logger = createLogger('SettingsManager');

// Settings endpoint
const SETTINGS_URL = IS_PRODUCTION
  ? 'https://www.visionflow.info/settings'
  : 'http://localhost:4000/settings';

export class SettingsManager {
  private static instance: SettingsManager;
  private settings: VisualizationSettings;
  private settingsListeners: Set<(settings: VisualizationSettings) => void>;

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
   * Load settings from the server
   */
  async loadSettings(): Promise<void> {
    try {
      const response = await fetch(SETTINGS_URL);
      if (!response.ok) {
        throw new Error(`Failed to load settings: ${response.statusText}`);
      }
      const settings = await response.json();
      this.updateSettings(settings);
      logger.log('Loaded settings from server');
    } catch (error) {
      logger.error('Error loading settings:', error);
      // Fall back to default settings
      this.updateSettings(DEFAULT_VISUALIZATION_SETTINGS);
    }
  }

  /**
   * Save current settings to the server
   */
  async saveSettings(): Promise<void> {
    try {
      const response = await fetch(SETTINGS_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(this.settings),
      });

      if (!response.ok) {
        throw new Error(`Failed to save settings: ${response.statusText}`);
      }

      logger.log('Settings saved successfully');
    } catch (error) {
      logger.error('Error saving settings:', error);
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

    logger.log('Updated settings');

    // Notify all listeners of the settings change
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
export const settingsManager = SettingsManager.getInstance();

/**
 * Settings management for visualization configuration
 */

import { VisualizationSettings } from '../core/types';
import { DEFAULT_VISUALIZATION_SETTINGS, SETTINGS_URL } from '../core/constants';
import { createLogger } from '../core/utils';

const logger = createLogger('SettingsManager');

export class SettingsManager {
  private static instance: SettingsManager;
  private settings: VisualizationSettings;
  private settingsListeners: Set<(settings: VisualizationSettings) => void>;

  private constructor() {
    this.settings = { ...DEFAULT_VISUALIZATION_SETTINGS };
    this.settingsListeners = new Set();
    logger.log('Initialized with default settings:', this.settings);
  }

  static getInstance(): SettingsManager {
    if (!SettingsManager.instance) {
      SettingsManager.instance = new SettingsManager();
    }
    return SettingsManager.instance;
  }

  /**
   * Load settings from the server's settings.toml file
   */
  async loadSettings(): Promise<void> {
    try {
      const response = await fetch(SETTINGS_URL);
      if (!response.ok) {
        throw new Error(`Failed to load settings: ${response.statusText}`);
      }
      const settings = await response.json();
      this.updateSettings(settings);
      logger.log('Loaded settings from server:', settings);
    } catch (error) {
      logger.error('Error loading settings:', error);
      // Fall back to default settings
      this.updateSettings(DEFAULT_VISUALIZATION_SETTINGS);
    }
  }

  /**
   * Save current settings to the server's settings.toml file
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

    logger.log('Updated settings:', this.settings);

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
    
    // Return unsubscribe function
    return () => {
      this.settingsListeners.delete(listener);
    };
  }

  /**
   * Reset settings to defaults
   */
  async resetToDefaults(): Promise<void> {
    this.updateSettings(DEFAULT_VISUALIZATION_SETTINGS);
    await this.saveSettings();
  }

  // Specific setting getters
  getNodeSize(): number {
    return this.settings.nodeSize;
  }

  getNodeColor(): string {
    return this.settings.nodeColor;
  }

  getNodeMaterialSettings(): {
    metalness: number;
    roughness: number;
    emissiveIntensity: number;
    clearcoat: number;
    clearcoatRoughness: number;
    reflectivity: number;
    envMapIntensity: number;
  } {
    return {
      metalness: this.settings.nodeMaterialMetalness,
      roughness: this.settings.nodeMaterialRoughness,
      emissiveIntensity: this.settings.nodeMaterialEmissiveIntensity,
      clearcoat: this.settings.nodeMaterialClearcoat,
      clearcoatRoughness: this.settings.nodeMaterialClearcoatRoughness,
      reflectivity: this.settings.nodeMaterialReflectivity,
      envMapIntensity: this.settings.nodeMaterialEnvMapIntensity
    };
  }

  getEdgeWidth(): number {
    return this.settings.edgeWidth;
  }

  getEdgeColor(): string {
    return this.settings.edgeColor;
  }

  getBloomEnabled(): boolean {
    return this.settings.enableBloom;
  }

  getBloomSettings(): {
    intensity: number;
    threshold: number;
    radius: number;
  } {
    return {
      intensity: this.settings.bloomIntensity,
      threshold: this.settings.bloomThreshold,
      radius: this.settings.bloomRadius
    };
  }

  getPhysicsSettings(): {
    gravity: number;
    springLength: number;
    springStiffness: number;
    charge: number;
    damping: number;
  } {
    return {
      gravity: this.settings.gravity,
      springLength: this.settings.springLength,
      springStiffness: this.settings.springStiffness,
      charge: this.settings.charge,
      damping: this.settings.damping
    };
  }

  getLabelSettings(): {
    showLabels: boolean;
    labelSize: number;
    labelColor: string;
  } {
    return {
      showLabels: this.settings.showLabels,
      labelSize: this.settings.labelSize,
      labelColor: this.settings.labelColor
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

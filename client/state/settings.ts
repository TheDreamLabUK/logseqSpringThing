/**
 * Settings management with simplified visualization configuration
 */

import { VisualizationSettings, ServerSettings, UpdateSettingsMessage, SettingsUpdatedMessage } from '../core/types';
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
    this.webSocket.on('settingsUpdated', (data: SettingsUpdatedMessage['data']) => {
      if (data && data.settings) {
        this.settings = this.flattenSettings(data.settings);
        this.notifyListeners();
        logger.log('Received settings update from server');
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
      // Convert client settings to server format
      const serverSettings: ServerSettings = {
        nodes: {
          base_size: this.settings.nodeSize,
          base_color: this.settings.nodeColor,
          opacity: this.settings.nodeOpacity,
          metalness: this.settings.metalness,
          roughness: this.settings.roughness,
          clearcoat: this.settings.clearcoat
        },
        edges: {
          base_width: this.settings.edgeWidth,
          color: this.settings.edgeColor,
          opacity: this.settings.edgeOpacity,
          enable_arrows: this.settings.enableArrows,
          arrow_size: this.settings.arrowSize
        },
        bloom: {
          enabled: this.settings.enableBloom,
          strength: this.settings.bloomIntensity,
          radius: this.settings.bloomRadius
        },
        animations: {
          enable_node_animations: this.settings.enableNodeAnimations,
          enable_motion_blur: this.settings.enableMotionBlur,
          motion_blur_strength: this.settings.motionBlurStrength
        },
        labels: {
          enable_labels: this.settings.showLabels,
          desktop_font_size: this.settings.labelSize * 48,
          text_color: this.settings.labelColor
        },
        ar: {
          enable_plane_detection: this.settings.enablePlaneDetection,
          enable_scene_understanding: this.settings.enableSceneUnderstanding,
          show_plane_overlay: this.settings.showPlaneOverlay,
          plane_opacity: this.settings.planeOpacity,
          plane_color: this.settings.planeColor,
          enable_light_estimation: this.settings.enableLightEstimation,
          enable_hand_tracking: this.settings.enableHandTracking,
          hand_mesh_enabled: this.settings.handMeshEnabled,
          hand_mesh_color: this.settings.handMeshColor,
          hand_mesh_opacity: this.settings.handMeshOpacity,
          hand_ray_enabled: this.settings.handRayEnabled,
          hand_ray_color: this.settings.handRayColor,
          hand_ray_width: this.settings.handRayWidth,
          hand_point_size: this.settings.handPointSize,
          gesture_smoothing: this.settings.gestureSmoothing,
          pinch_threshold: this.settings.pinchThreshold,
          drag_threshold: this.settings.dragThreshold,
          rotation_threshold: this.settings.rotationThreshold,
          enable_haptics: this.settings.enableHaptics,
          haptic_intensity: this.settings.hapticIntensity,
          room_scale: this.settings.roomScale,
          snap_to_floor: this.settings.snapToFloor,
          passthrough_opacity: this.settings.passthroughOpacity,
          passthrough_brightness: this.settings.passthroughBrightness,
          passthrough_contrast: this.settings.passthroughContrast,
          enable_passthrough_portal: this.settings.enablePassthroughPortal,
          portal_size: this.settings.portalSize,
          portal_edge_color: this.settings.portalEdgeColor,
          portal_edge_width: this.settings.portalEdgeWidth
        }
      };

      const message: UpdateSettingsMessage = {
        type: 'updateSettings',
        data: {
          settings: serverSettings
        }
      };

      this.webSocket.send(message);
      logger.log('Settings update sent to server');
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
   * Convert server settings to client format
   */
  private flattenSettings(serverSettings: ServerSettings): VisualizationSettings {
    return {
      // Node settings
      nodeSize: serverSettings.nodes.base_size,
      nodeColor: serverSettings.nodes.base_color,
      nodeOpacity: serverSettings.nodes.opacity,
      metalness: serverSettings.nodes.metalness,
      roughness: serverSettings.nodes.roughness,
      clearcoat: serverSettings.nodes.clearcoat,

      // Edge settings
      edgeWidth: serverSettings.edges.base_width,
      edgeColor: serverSettings.edges.color,
      edgeOpacity: serverSettings.edges.opacity,
      enableArrows: serverSettings.edges.enable_arrows,
      arrowSize: serverSettings.edges.arrow_size,

      // Bloom settings
      enableBloom: serverSettings.bloom.enabled,
      bloomIntensity: serverSettings.bloom.strength,
      bloomRadius: serverSettings.bloom.radius,

      // Animation settings
      enableNodeAnimations: serverSettings.animations.enable_node_animations,
      enableMotionBlur: serverSettings.animations.enable_motion_blur,
      motionBlurStrength: serverSettings.animations.motion_blur_strength,

      // Label settings
      showLabels: serverSettings.labels.enable_labels,
      labelSize: serverSettings.labels.desktop_font_size / 48,
      labelColor: serverSettings.labels.text_color,

      // Performance settings
      maxFps: this.settings.maxFps, // Not in server settings

      // AR settings
      enablePlaneDetection: serverSettings.ar.enable_plane_detection,
      enableSceneUnderstanding: serverSettings.ar.enable_scene_understanding,
      showPlaneOverlay: serverSettings.ar.show_plane_overlay,
      planeOpacity: serverSettings.ar.plane_opacity,
      planeColor: serverSettings.ar.plane_color,
      enableLightEstimation: serverSettings.ar.enable_light_estimation,
      enableHandTracking: serverSettings.ar.enable_hand_tracking,
      handMeshEnabled: serverSettings.ar.hand_mesh_enabled,
      handMeshColor: serverSettings.ar.hand_mesh_color,
      handMeshOpacity: serverSettings.ar.hand_mesh_opacity,
      handRayEnabled: serverSettings.ar.hand_ray_enabled,
      handRayColor: serverSettings.ar.hand_ray_color,
      handRayWidth: serverSettings.ar.hand_ray_width,
      handPointSize: serverSettings.ar.hand_point_size,
      gestureSmoothing: serverSettings.ar.gesture_smoothing,
      pinchThreshold: serverSettings.ar.pinch_threshold,
      dragThreshold: serverSettings.ar.drag_threshold,
      rotationThreshold: serverSettings.ar.rotation_threshold,
      enableHaptics: serverSettings.ar.enable_haptics,
      hapticIntensity: serverSettings.ar.haptic_intensity,
      roomScale: serverSettings.ar.room_scale,
      snapToFloor: serverSettings.ar.snap_to_floor,
      passthroughOpacity: serverSettings.ar.passthrough_opacity,
      passthroughBrightness: serverSettings.ar.passthrough_brightness,
      passthroughContrast: serverSettings.ar.passthrough_contrast,
      enablePassthroughPortal: serverSettings.ar.enable_passthrough_portal,
      portalSize: serverSettings.ar.portal_size,
      portalEdgeColor: serverSettings.ar.portal_edge_color,
      portalEdgeWidth: serverSettings.ar.portal_edge_width
    };
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
}

// Create and export the singleton instance
export const settingsManager = SettingsManager.getInstance();

/**
 * Settings management with simplified visualization configuration
 */

import { VisualizationSettings, ServerSettings, UpdateSettingsMessage, SettingsUpdatedMessage } from '../core/types';
import { createLogger } from '../core/utils';
import { WebSocketService } from '../websocket/websocketService';

const logger = createLogger('SettingsManager');

export const DEFAULT_VISUALIZATION_SETTINGS: VisualizationSettings = {
  // Node Appearance
  nodeSize: 1.0,
  nodeColor: '#FFB700',
  nodeOpacity: 0.92,
  metalness: 0.85,
  roughness: 0.15,
  clearcoat: 1.0,
  highlightColor: '#FFFFFF',
  highlightDuration: 500,
  enableHoverEffect: true,
  hoverScale: 1.2,

  // Edge Appearance
  edgeWidth: 2.0,
  edgeColor: '#FFD700',
  edgeOpacity: 0.6,
  edgeWidthRange: [1.0, 3.0],

  // Visual Effects
  enableBloom: false,
  nodeBloomStrength: 0.2,
  edgeBloomStrength: 0.3,
  environmentBloomStrength: 0.5,

  // Labels
  showLabels: true,
  labelColor: '#FFFFFF',

  // Physics
  physicsEnabled: true,
  attractionStrength: 0.015,
  repulsionStrength: 1500.0,
  springStrength: 0.018,
  damping: 0.88,
  maxVelocity: 2.5,
  collisionRadius: 0.25,
  boundsSize: 12.0,
  enableBounds: true,
  iterations: 500,

  // AR Settings
  enableHandTracking: true,
  enableHaptics: true,
  enablePlaneDetection: true,

  // Rendering
  ambientLightIntensity: 0.7,
  directionalLightIntensity: 1.0,
  environmentIntensity: 1.2,
  enableAmbientOcclusion: true,
  enableAntialiasing: true,
  enableShadows: true,
  backgroundColor: '#000000',

  // Integrations
  github: {
    basePath: "default_path",
    owner: "default_owner",
    rateLimitEnabled: true,
    repo: "default_repo",
    token: "default_token"
  },
  openai: {
    apiKey: "default_openai_key",
    baseUrl: "wss://api.openai.com/v1/realtime",
    model: "gpt-4o-realtime-preview-2024-10-01",
    rateLimit: 100,
    timeout: 30
  },
  perplexity: {
    apiKey: "default_perplexity_key",
    apiUrl: "https://api.perplexity.ai/chat/completions",
    frequencyPenalty: 1.0,
    maxTokens: 4096,
    model: "llama-3.1-sonar-small-128k-online",
    presencePenalty: 0.0,
    prompt: `You are an AI assistant for LogSeq knowledge summaries via perplexity API. Style: Informative, analytical, optimistic, critical, conversational, authoritative. Markdown: hierarchical headings (- #), minimal bold, italics for book titles, descriptive links [URL](text), images ![alt](path){:width height}, embeds {{type id}}, lists with '-', block refs [[title]], properties property:: value, code blocks \`\`\`lang\`\`\` and \`inline code\`, \r newlines, public:: true at start. UK spelling, introduce acronyms once, numeric citations, [[reference]] for sources, minimal emojis, parentheses for asides, collapsed:: true. Focus on emerging tech (decentralization, AI, XR), detail, credible sources, implications, examples, future-oriented. Adhere strictly, ensure accuracy, consistency, large context, refine with feedback.`,
    rateLimit: 100,
    temperature: 0.5,
    timeout: 30,
    topP: 0.9
  },
  ragflow: {
    apiKey: "default_ragflow_key",
    baseUrl: "http://ragflow-server/v1/",
    maxRetries: 3,
    timeout: 30
  }
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

  initializeWebSocket(webSocket: WebSocketService): void {
    this.webSocket = webSocket;

    this.webSocket.on('settingsUpdated', (data: SettingsUpdatedMessage['data']) => {
      if (data && data.settings) {
        this.settings = this.flattenSettings(data.settings);
        this.notifyListeners();
        logger.log('Received settings update from server');
      }
    });

    logger.log('WebSocket initialized for settings');
  }

  async loadSettings(): Promise<void> {
    logger.log('Settings will be received through WebSocket');
  }

  async saveSettings(): Promise<void> {
    if (!this.webSocket) {
      throw new Error('WebSocket not initialized');
    }

    try {
      const serverSettings: ServerSettings = {
        nodes: {
          base_size: this.settings.nodeSize,
          base_color: this.settings.nodeColor,
          opacity: this.settings.nodeOpacity,
          metalness: this.settings.metalness,
          roughness: this.settings.roughness,
          clearcoat: this.settings.clearcoat,
          highlight_color: this.settings.highlightColor,
          highlight_duration: this.settings.highlightDuration,
          enable_hover_effect: this.settings.enableHoverEffect,
          hover_scale: this.settings.hoverScale
        },
        edges: {
          base_width: this.settings.edgeWidth,
          color: this.settings.edgeColor,
          opacity: this.settings.edgeOpacity,
          width_range: this.settings.edgeWidthRange
        },
        bloom: {
          enabled: this.settings.enableBloom,
          node_bloom_strength: this.settings.nodeBloomStrength,
          edge_bloom_strength: this.settings.edgeBloomStrength,
          environment_bloom_strength: this.settings.environmentBloomStrength
        },
        labels: {
          enable_labels: this.settings.showLabels,
          text_color: this.settings.labelColor
        },
        physics: {
          enabled: this.settings.physicsEnabled,
          attraction_strength: this.settings.attractionStrength,
          repulsion_strength: this.settings.repulsionStrength,
          spring_strength: this.settings.springStrength,
          damping: this.settings.damping,
          max_velocity: this.settings.maxVelocity,
          collision_radius: this.settings.collisionRadius,
          bounds_size: this.settings.boundsSize,
          enable_bounds: this.settings.enableBounds,
          iterations: this.settings.iterations
        },
        ar: {
          enable_hand_tracking: this.settings.enableHandTracking,
          enable_haptics: this.settings.enableHaptics,
          enable_plane_detection: this.settings.enablePlaneDetection
        },
        rendering: {
          ambient_light_intensity: this.settings.ambientLightIntensity,
          directional_light_intensity: this.settings.directionalLightIntensity,
          environment_intensity: this.settings.environmentIntensity,
          enable_ambient_occlusion: this.settings.enableAmbientOcclusion,
          enable_antialiasing: this.settings.enableAntialiasing,
          enable_shadows: this.settings.enableShadows,
          background_color: this.settings.backgroundColor
        },
        github: {
          base_path: this.settings.github.basePath,
          owner: this.settings.github.owner,
          rate_limit: this.settings.github.rateLimitEnabled,
          repo: this.settings.github.repo,
          token: this.settings.github.token
        },
        openai: {
          api_key: this.settings.openai.apiKey,
          base_url: this.settings.openai.baseUrl,
          model: this.settings.openai.model,
          rate_limit: this.settings.openai.rateLimit,
          timeout: this.settings.openai.timeout
        },
        perplexity: {
          api_key: this.settings.perplexity.apiKey,
          api_url: this.settings.perplexity.apiUrl,
          frequency_penalty: this.settings.perplexity.frequencyPenalty,
          max_tokens: this.settings.perplexity.maxTokens,
          model: this.settings.perplexity.model,
          presence_penalty: this.settings.perplexity.presencePenalty,
          prompt: this.settings.perplexity.prompt,
          rate_limit: this.settings.perplexity.rateLimit,
          temperature: this.settings.perplexity.temperature,
          timeout: this.settings.perplexity.timeout,
          top_p: this.settings.perplexity.topP
        },
        ragflow: {
          api_key: this.settings.ragflow.apiKey,
          base_url: this.settings.ragflow.baseUrl,
          max_retries: this.settings.ragflow.maxRetries,
          timeout: this.settings.ragflow.timeout
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

  updateSettings(newSettings: Partial<VisualizationSettings>): void {
    this.settings = {
      ...this.settings,
      ...newSettings
    };

    logger.log('Updated settings locally');
    this.notifyListeners();
    
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

  private flattenSettings(serverSettings: ServerSettings): VisualizationSettings {
    return {
      // Node settings
      nodeSize: serverSettings.nodes.base_size,
      nodeColor: serverSettings.nodes.base_color,
      nodeOpacity: serverSettings.nodes.opacity,
      metalness: serverSettings.nodes.metalness,
      roughness: serverSettings.nodes.roughness,
      clearcoat: serverSettings.nodes.clearcoat,
      highlightColor: serverSettings.nodes.highlight_color,
      highlightDuration: serverSettings.nodes.highlight_duration,
      enableHoverEffect: serverSettings.nodes.enable_hover_effect,
      hoverScale: serverSettings.nodes.hover_scale,

      // Edge settings
      edgeWidth: serverSettings.edges.base_width,
      edgeColor: serverSettings.edges.color,
      edgeOpacity: serverSettings.edges.opacity,
      edgeWidthRange: serverSettings.edges.width_range,

      // Bloom settings
      enableBloom: serverSettings.bloom.enabled,
      nodeBloomStrength: serverSettings.bloom.node_bloom_strength,
      edgeBloomStrength: serverSettings.bloom.edge_bloom_strength,
      environmentBloomStrength: serverSettings.bloom.environment_bloom_strength,

      // Label settings
      showLabels: serverSettings.labels.enable_labels,
      labelColor: serverSettings.labels.text_color,

      // Physics settings
      physicsEnabled: serverSettings.physics.enabled,
      attractionStrength: serverSettings.physics.attraction_strength,
      repulsionStrength: serverSettings.physics.repulsion_strength,
      springStrength: serverSettings.physics.spring_strength,
      damping: serverSettings.physics.damping,
      maxVelocity: serverSettings.physics.max_velocity,
      collisionRadius: serverSettings.physics.collision_radius,
      boundsSize: serverSettings.physics.bounds_size,
      enableBounds: serverSettings.physics.enable_bounds,
      iterations: serverSettings.physics.iterations,

      // AR settings
      enableHandTracking: serverSettings.ar.enable_hand_tracking,
      enableHaptics: serverSettings.ar.enable_haptics,
      enablePlaneDetection: serverSettings.ar.enable_plane_detection,

      // Rendering settings
      ambientLightIntensity: serverSettings.rendering.ambient_light_intensity,
      directionalLightIntensity: serverSettings.rendering.directional_light_intensity,
      environmentIntensity: serverSettings.rendering.environment_intensity,
      enableAmbientOcclusion: serverSettings.rendering.enable_ambient_occlusion,
      enableAntialiasing: serverSettings.rendering.enable_antialiasing,
      enableShadows: serverSettings.rendering.enable_shadows,
      backgroundColor: serverSettings.rendering.background_color,

      // Integration settings
      github: {
        basePath: serverSettings.github.base_path,
        owner: serverSettings.github.owner,
        rateLimitEnabled: serverSettings.github.rate_limit,
        repo: serverSettings.github.repo,
        token: serverSettings.github.token
      },
      openai: {
        apiKey: serverSettings.openai.api_key,
        baseUrl: serverSettings.openai.base_url,
        model: serverSettings.openai.model,
        rateLimit: serverSettings.openai.rate_limit,
        timeout: serverSettings.openai.timeout
      },
      perplexity: {
        apiKey: serverSettings.perplexity.api_key,
        apiUrl: serverSettings.perplexity.api_url,
        frequencyPenalty: serverSettings.perplexity.frequency_penalty,
        maxTokens: serverSettings.perplexity.max_tokens,
        model: serverSettings.perplexity.model,
        presencePenalty: serverSettings.perplexity.presence_penalty,
        prompt: serverSettings.perplexity.prompt,
        rateLimit: serverSettings.perplexity.rate_limit,
        temperature: serverSettings.perplexity.temperature,
        timeout: serverSettings.perplexity.timeout,
        topP: serverSettings.perplexity.top_p
      },
      ragflow: {
        apiKey: serverSettings.ragflow.api_key,
        baseUrl: serverSettings.ragflow.base_url,
        maxRetries: serverSettings.ragflow.max_retries,
        timeout: serverSettings.ragflow.timeout
      }
    };
  }

  addSettingsListener(listener: (settings: VisualizationSettings) => void): void {
    this.settingsListeners.add(listener);
  }

  removeSettingsListener(listener: (settings: VisualizationSettings) => void): void {
    this.settingsListeners.delete(listener);
  }

  getSettings(): VisualizationSettings {
    return { ...this.settings };
  }

  subscribe(listener: (settings: VisualizationSettings) => void): () => void {
    this.settingsListeners.add(listener);
    return () => {
      this.settingsListeners.delete(listener);
    };
  }

  resetToDefaults(): void {
    this.updateSettings(DEFAULT_VISUALIZATION_SETTINGS);
  }

  dispose(): void {
    if (this.webSocket) {
      this.webSocket.off('settingsUpdated', this.notifyListeners);
      this.webSocket = null;
    }
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
        highlightColor: this.settings.highlightColor
      },
      edges: {
        width: this.settings.edgeWidth,
        color: this.settings.edgeColor,
        opacity: this.settings.edgeOpacity,
        widthRange: this.settings.edgeWidthRange
      },
      bloom: {
        enabled: this.settings.enableBloom,
        nodeStrength: this.settings.nodeBloomStrength,
        edgeStrength: this.settings.edgeBloomStrength,
        environmentStrength: this.settings.environmentBloomStrength
      },
      labels: {
        enabled: this.settings.showLabels,
        color: this.settings.labelColor
      },
      rendering: {
        ambientLightIntensity: this.settings.ambientLightIntensity,
        directionalLightIntensity: this.settings.directionalLightIntensity,
        environmentIntensity: this.settings.environmentIntensity,
        enableAmbientOcclusion: this.settings.enableAmbientOcclusion,
        enableAntialiasing: this.settings.enableAntialiasing,
        enableShadows: this.settings.enableShadows,
        backgroundColor: this.settings.backgroundColor
      }
    };
  }
}

export const settingsManager = SettingsManager.getInstance();

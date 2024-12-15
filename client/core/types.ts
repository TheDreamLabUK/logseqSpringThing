// Core types for the application

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface NodeData {
  position: Vector3;
  velocity: Vector3;
}

export interface Node {
  id: string;
  data: NodeData;
  color?: string;
}

export interface Edge {
  source: string;
  target: string;
  weight?: number;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

// Integration Settings Types
export interface GithubSettings {
  basePath: string;
  owner: string;
  rateLimitEnabled: boolean;
  repo: string;
  token: string;
}

export interface OpenAISettings {
  apiKey: string;
  baseUrl: string;
  model: string;
  rateLimit: number;
  timeout: number;
}

export interface PerplexitySettings {
  apiKey: string;
  apiUrl: string;
  frequencyPenalty: number;
  maxTokens: number;
  model: string;
  presencePenalty: number;
  prompt: string;
  rateLimit: number;
  temperature: number;
  timeout: number;
  topP: number;
}

export interface RagFlowSettings {
  apiKey: string;
  baseUrl: string;
  maxRetries: number;
  timeout: number;
}

// Client-side visualization settings
export interface VisualizationSettings {
  // Node Appearance
  nodeSize: number;
  nodeColor: string;
  nodeOpacity: number;
  metalness: number;
  roughness: number;
  clearcoat: number;
  highlightColor: string;
  highlightDuration: number;
  enableHoverEffect: boolean;
  hoverScale: number;

  // Edge Appearance
  edgeWidth: number;
  edgeColor: string;
  edgeOpacity: number;
  edgeWidthRange: [number, number];

  // Visual Effects
  enableBloom: boolean;
  nodeBloomStrength: number;
  edgeBloomStrength: number;
  environmentBloomStrength: number;

  // Labels
  showLabels: boolean;
  labelColor: string;

  // Physics
  physicsEnabled: boolean;
  attractionStrength: number;
  repulsionStrength: number;
  springStrength: number;
  damping: number;
  maxVelocity: number;
  collisionRadius: number;
  boundsSize: number;
  enableBounds: boolean;
  iterations: number;

  // AR Settings
  enableHandTracking: boolean;
  enableHaptics: boolean;
  enablePlaneDetection: boolean;

  // Rendering
  ambientLightIntensity: number;
  directionalLightIntensity: number;
  environmentIntensity: number;
  enableAmbientOcclusion: boolean;
  enableAntialiasing: boolean;
  enableShadows: boolean;
  backgroundColor: string;

  // Integrations
  github: GithubSettings;
  openai: OpenAISettings;
  perplexity: PerplexitySettings;
  ragflow: RagFlowSettings;
}

// Server-side settings format
export interface ServerSettings {
  nodes: {
    base_size: number;
    base_color: string;
    opacity: number;
    metalness: number;
    roughness: number;
    clearcoat: number;
    highlight_color: string;
    highlight_duration: number;
    enable_hover_effect: boolean;
    hover_scale: number;
  };
  edges: {
    base_width: number;
    color: string;
    opacity: number;
    width_range: [number, number];
  };
  bloom: {
    enabled: boolean;
    node_bloom_strength: number;
    edge_bloom_strength: number;
    environment_bloom_strength: number;
  };
  labels: {
    enable_labels: boolean;
    text_color: string;
  };
  physics: {
    enabled: boolean;
    attraction_strength: number;
    repulsion_strength: number;
    spring_strength: number;
    damping: number;
    max_velocity: number;
    collision_radius: number;
    bounds_size: number;
    enable_bounds: boolean;
    iterations: number;
  };
  ar: {
    enable_hand_tracking: boolean;
    enable_haptics: boolean;
    enable_plane_detection: boolean;
  };
  rendering: {
    ambient_light_intensity: number;
    directional_light_intensity: number;
    environment_intensity: number;
    enable_ambient_occlusion: boolean;
    enable_antialiasing: boolean;
    enable_shadows: boolean;
    background_color: string;
  };
  github: {
    base_path: string;
    owner: string;
    rate_limit: boolean;
    repo: string;
    token: string;
  };
  openai: {
    api_key: string;
    base_url: string;
    model: string;
    rate_limit: number;
    timeout: number;
  };
  perplexity: {
    api_key: string;
    api_url: string;
    frequency_penalty: number;
    max_tokens: number;
    model: string;
    presence_penalty: number;
    prompt: string;
    rate_limit: number;
    temperature: number;
    timeout: number;
    top_p: number;
  };
  ragflow: {
    api_key: string;
    base_url: string;
    max_retries: number;
    timeout: number;
  };
}

// WebSocket message types
export interface UpdateSettingsMessage {
  type: 'updateSettings';
  data: {
    settings: ServerSettings;
  };
}

export interface SettingsUpdatedMessage {
  type: 'settingsUpdated';
  data: {
    settings: ServerSettings;
  };
}

export interface GraphUpdateMessage {
  type: 'graphUpdate';
  data: {
    nodes: Node[];
    edges: Edge[];
  };
}

export interface PositionUpdateMessage {
  type: 'positionUpdate';
  data: Float32Array;
}

export type WebSocketMessage = 
  | UpdateSettingsMessage 
  | SettingsUpdatedMessage 
  | GraphUpdateMessage 
  | PositionUpdateMessage;

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
  metadata?: any;
}

export interface Edge {
  source: string;
  target: string;
  weight?: number;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: any;
}

// Platform types
export type Platform = 'desktop' | 'quest' | 'browser';

export interface PlatformCapabilities {
  xrSupported: boolean;
  webglSupported: boolean;
  websocketSupported: boolean;
  webxr: boolean;
  handTracking: boolean;
  planeDetection: boolean;
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
  // Node appearance
  nodeSize: number;
  nodeColor: string;
  nodeOpacity: number;
  metalness: number;
  roughness: number;
  clearcoat: number;
  enableInstancing: boolean;
  materialType: string;
  sizeRange: [number, number];
  sizeByConnections: boolean;
  highlightColor: string;
  highlightDuration: number;
  enableHoverEffect: boolean;
  hoverScale: number;

  // Edge appearance
  edgeWidth: number;
  edgeColor: string;
  edgeOpacity: number;
  edgeWidthRange: [number, number];
  enableArrows: boolean;
  arrowSize: number;

  // Physics settings
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

  // Lighting and environment
  ambientLightIntensity: number;
  directionalLightIntensity: number;
  environmentIntensity: number;
  enableAmbientOcclusion: boolean;
  enableAntialiasing: boolean;
  enableShadows: boolean;
  backgroundColor: string;

  // Visual effects
  enableBloom: boolean;
  bloomIntensity: number;
  bloomRadius: number;
  nodeBloomStrength: number;
  edgeBloomStrength: number;
  environmentBloomStrength: number;
  enableNodeAnimations: boolean;
  enableMotionBlur: boolean;
  motionBlurStrength: number;

  // Labels
  showLabels: boolean;
  labelSize: number;
  labelColor: string;

  // Performance
  maxFps: number;

  // AR Settings
  enablePlaneDetection: boolean;
  enableSceneUnderstanding: boolean;
  showPlaneOverlay: boolean;
  planeOpacity: number;
  planeColor: string;
  enableLightEstimation: boolean;
  enableHandTracking: boolean;
  handMeshEnabled: boolean;
  handMeshColor: string;
  handMeshOpacity: number;
  handRayEnabled: boolean;
  handRayColor: string;
  handRayWidth: number;
  handPointSize: number;
  gestureSmoothing: number;
  pinchThreshold: number;
  dragThreshold: number;
  rotationThreshold: number;
  enableHaptics: boolean;
  hapticIntensity: number;
  roomScale: boolean;
  snapToFloor: boolean;
  passthroughOpacity: number;
  passthroughBrightness: number;
  passthroughContrast: number;
  enablePassthroughPortal: boolean;
  portalSize: number;
  portalEdgeColor: string;
  portalEdgeWidth: number;
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
    enable_instancing: boolean;
    material_type: string;
    size_range: [number, number];
    size_by_connections: boolean;
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
    enable_arrows: boolean;
    arrow_size: number;
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
  rendering: {
    ambient_light_intensity: number;
    directional_light_intensity: number;
    environment_intensity: number;
    enable_ambient_occlusion: boolean;
    enable_antialiasing: boolean;
    enable_shadows: boolean;
    background_color: string;
  };
  bloom: {
    enabled: boolean;
    strength: number;
    radius: number;
    node_bloom_strength: number;
    edge_bloom_strength: number;
    environment_bloom_strength: number;
  };
  animations: {
    enable_node_animations: boolean;
    enable_motion_blur: boolean;
    motion_blur_strength: number;
  };
  labels: {
    enable_labels: boolean;
    desktop_font_size: number;
    text_color: string;
  };
  ar: {
    enable_plane_detection: boolean;
    enable_scene_understanding: boolean;
    show_plane_overlay: boolean;
    plane_opacity: number;
    plane_color: string;
    enable_light_estimation: boolean;
    enable_hand_tracking: boolean;
    hand_mesh_enabled: boolean;
    hand_mesh_color: string;
    hand_mesh_opacity: number;
    hand_ray_enabled: boolean;
    hand_ray_color: string;
    hand_ray_width: number;
    hand_point_size: number;
    gesture_smoothing: number;
    pinch_threshold: number;
    drag_threshold: number;
    rotation_threshold: number;
    enable_haptics: boolean;
    haptic_intensity: number;
    room_scale: boolean;
    snap_to_floor: boolean;
    passthrough_opacity: number;
    passthrough_brightness: number;
    passthrough_contrast: number;
    enable_passthrough_portal: boolean;
    portal_size: number;
    portal_edge_color: string;
    portal_edge_width: number;
  };
}

// WebSocket message types
export type MessageType = 
  | 'initialData'
  | 'graphUpdate'
  | 'positionUpdate'
  | 'binaryPositionUpdate'
  | 'updateSettings'
  | 'settingsUpdated'
  | 'requestInitialData'
  | 'enableBinaryUpdates'
  | 'connect'
  | 'disconnect'
  | 'ping'
  | 'pong';

// Base message types
export interface WebSocketMessage {
  type: MessageType;
  data?: any;
}

export interface RawWebSocketMessage {
  type: MessageType;
  data?: any;
}

// Initial data messages
export interface InitialDataMessage extends WebSocketMessage {
  type: 'initialData';
  data: {
    graph: GraphData;
  };
}

export interface RawInitialDataMessage extends RawWebSocketMessage {
  type: 'initialData';
  data: {
    graph: any;
  };
}

// Binary position update messages
export interface BinaryPositionUpdateMessage extends WebSocketMessage {
  type: 'binaryPositionUpdate';
  data: {
    nodes: {
      nodeId: string;
      data: NodeData;
    }[];
  };
}

export interface RawBinaryPositionUpdateMessage extends RawWebSocketMessage {
  type: 'binaryPositionUpdate';
  data: {
    nodes: {
      nodeId: string;
      data: any;
    }[];
  };
}

// Other message types
export interface RequestInitialDataMessage extends WebSocketMessage {
  type: 'requestInitialData';
}

export interface EnableBinaryUpdatesMessage extends WebSocketMessage {
  type: 'enableBinaryUpdates';
}

export interface PingMessage extends WebSocketMessage {
  type: 'ping';
}

export interface PongMessage extends WebSocketMessage {
  type: 'pong';
}

// Settings messages
export interface UpdateSettingsMessage extends WebSocketMessage {
  type: 'updateSettings';
  data: {
    settings: ServerSettings;
  };
}

export interface SettingsUpdatedMessage extends WebSocketMessage {
  type: 'settingsUpdated';
  data: {
    settings: ServerSettings;
  };
}

// Logger interface
export interface Logger {
  log: (message: string, ...args: any[]) => void;
  error: (message: string, ...args: any[]) => void;
  warn: (message: string, ...args: any[]) => void;
  debug: (message: string, ...args: any[]) => void;
  info: (message: string, ...args: any[]) => void;
}

// Helper functions
export function transformGraphData(data: any): GraphData {
  return {
    nodes: data.nodes.map((node: any) => transformNodeData(node)),
    edges: data.edges,
    metadata: data.metadata
  };
}

export function transformNodeData(node: any): Node {
  return {
    id: node.id,
    data: {
      position: node.data.position,
      velocity: node.data.velocity || { x: 0, y: 0, z: 0 }
    },
    color: node.color,
    metadata: node.metadata
  };
}

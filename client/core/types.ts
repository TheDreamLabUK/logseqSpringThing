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
  animations: {
    enable_motion_blur: boolean;
    enable_node_animations: boolean;
    motion_blur_strength: number;
  };
  ar: {
    drag_threshold: number;
    enable_hand_tracking: boolean;
    enable_haptics: boolean;
    enable_light_estimation: boolean;
    enable_passthrough_portal: boolean;
    enable_plane_detection: boolean;
    enable_scene_understanding: boolean;
    gesture_smoothing: number;
    hand_mesh_color: string;
    hand_mesh_enabled: boolean;
    hand_mesh_opacity: number;
    hand_point_size: number;
    hand_ray_color: string;
    hand_ray_enabled: boolean;
    hand_ray_width: number;
    haptic_intensity: number;
    passthrough_brightness: number;
    passthrough_contrast: number;
    passthrough_opacity: number;
    pinch_threshold: number;
    plane_color: string;
    plane_opacity: number;
    portal_edge_color: string;
    portal_edge_width: number;
    portal_size: number;
    room_scale: boolean;
    rotation_threshold: number;
    show_plane_overlay: boolean;
    snap_to_floor: boolean;
  };
  audio: {
    enable_ambient_sounds: boolean;
    enable_interaction_sounds: boolean;
    enable_spatial_audio: boolean;
  };
  bloom: {
    edge_bloom_strength: number;
    enabled: boolean;
    environment_bloom_strength: number;
    node_bloom_strength: number;
    radius: number;
    strength: number;
  };
  edges: {
    arrow_size: number;
    base_width: number;
    color: string;
    enable_arrows: boolean;
    opacity: number;
    width_range: [number, number];
  };
  labels: {
    desktop_font_size: number;
    enable_labels: boolean;
    text_color: string;
  };
  nodes: {
    base_color: string;
    base_size: number;
    clearcoat: number;
    enable_hover_effect: boolean;
    enable_instancing: boolean;
    highlight_color: string;
    highlight_duration: number;
    hover_scale: number;
    material_type: string;
    metalness: number;
    opacity: number;
    roughness: number;
    size_by_connections: boolean;
    size_range: [number, number];
  };
  physics: {
    attraction_strength: number;
    bounds_size: number;
    collision_radius: number;
    damping: number;
    enable_bounds: boolean;
    enabled: boolean;
    iterations: number;
    max_velocity: number;
    repulsion_strength: number;
    spring_strength: number;
  };
  rendering: {
    ambient_light_intensity: number;
    background_color: string;
    directional_light_intensity: number;
    enable_ambient_occlusion: boolean;
    enable_antialiasing: boolean;
    enable_shadows: boolean;
    environment_intensity: number;
  };
}

// Client-side visualization settings (mapped from server settings)
export interface VisualizationSettings {
  // Node Appearance
  nodeSize: number;  // maps from nodes.base_size
  nodeColor: string;  // maps from nodes.base_color
  nodeOpacity: number;  // maps from nodes.opacity
  metalness: number;
  roughness: number;
  clearcoat: number;
  enableInstancing: boolean;  // maps from nodes.enable_instancing
  materialType: string;  // maps from nodes.material_type
  sizeRange: [number, number];  // maps from nodes.size_range
  sizeByConnections: boolean;  // maps from nodes.size_by_connections
  highlightColor: string;  // maps from nodes.highlight_color
  highlightDuration: number;  // maps from nodes.highlight_duration
  enableHoverEffect: boolean;  // maps from nodes.enable_hover_effect
  hoverScale: number;  // maps from nodes.hover_scale

  // Edge Appearance
  edgeWidth: number;  // maps from edges.base_width
  edgeColor: string;  // maps from edges.color
  edgeOpacity: number;  // maps from edges.opacity
  edgeWidthRange: [number, number];  // maps from edges.width_range
  enableArrows: boolean;  // maps from edges.enable_arrows
  arrowSize: number;  // maps from edges.arrow_size

  // Physics Settings
  physicsEnabled: boolean;  // maps from physics.enabled
  attractionStrength: number;  // maps from physics.attraction_strength
  repulsionStrength: number;  // maps from physics.repulsion_strength
  springStrength: number;  // maps from physics.spring_strength
  damping: number;  // maps from physics.damping
  maxVelocity: number;  // maps from physics.max_velocity
  collisionRadius: number;  // maps from physics.collision_radius
  boundsSize: number;  // maps from physics.bounds_size
  enableBounds: boolean;  // maps from physics.enable_bounds
  iterations: number;  // maps from physics.iterations

  // Rendering Settings
  ambientLightIntensity: number;  // maps from rendering.ambient_light_intensity
  directionalLightIntensity: number;  // maps from rendering.directional_light_intensity
  environmentIntensity: number;  // maps from rendering.environment_intensity
  enableAmbientOcclusion: boolean;  // maps from rendering.enable_ambient_occlusion
  enableAntialiasing: boolean;  // maps from rendering.enable_antialiasing
  enableShadows: boolean;  // maps from rendering.enable_shadows
  backgroundColor: string;  // maps from rendering.background_color

  // Visual Effects
  enableBloom: boolean;  // maps from bloom.enabled
  bloomIntensity: number;  // maps from bloom.strength
  bloomRadius: number;  // maps from bloom.radius
  nodeBloomStrength: number;  // maps from bloom.node_bloom_strength
  edgeBloomStrength: number;  // maps from bloom.edge_bloom_strength
  environmentBloomStrength: number;  // maps from bloom.environment_bloom_strength
  enableNodeAnimations: boolean;  // maps from animations.enable_node_animations
  enableMotionBlur: boolean;  // maps from animations.enable_motion_blur
  motionBlurStrength: number;  // maps from animations.motion_blur_strength

  // Labels
  showLabels: boolean;  // maps from labels.enable_labels
  labelSize: number;  // maps from labels.desktop_font_size / 48
  labelColor: string;  // maps from labels.text_color

  // AR Settings
  enablePlaneDetection: boolean;  // maps from ar.enable_plane_detection
  enableSceneUnderstanding: boolean;  // maps from ar.enable_scene_understanding
  showPlaneOverlay: boolean;  // maps from ar.show_plane_overlay
  planeOpacity: number;  // maps from ar.plane_opacity
  planeColor: string;  // maps from ar.plane_color
  enableLightEstimation: boolean;  // maps from ar.enable_light_estimation
  enableHandTracking: boolean;  // maps from ar.enable_hand_tracking
  handMeshEnabled: boolean;  // maps from ar.hand_mesh_enabled
  handMeshColor: string;  // maps from ar.hand_mesh_color
  handMeshOpacity: number;  // maps from ar.hand_mesh_opacity
  handRayEnabled: boolean;  // maps from ar.hand_ray_enabled
  handRayColor: string;  // maps from ar.hand_ray_color
  handRayWidth: number;  // maps from ar.hand_ray_width
  handPointSize: number;  // maps from ar.hand_point_size
  gestureSmoothing: number;  // maps from ar.gesture_smoothing
  pinchThreshold: number;  // maps from ar.pinch_threshold
  dragThreshold: number;  // maps from ar.drag_threshold
  rotationThreshold: number;  // maps from ar.rotation_threshold
  enableHaptics: boolean;  // maps from ar.enable_haptics
  hapticIntensity: number;  // maps from ar.haptic_intensity
  roomScale: boolean;  // maps from ar.room_scale
  snapToFloor: boolean;  // maps from ar.snap_to_floor
  passthroughOpacity: number;  // maps from ar.passthrough_opacity
  passthroughBrightness: number;  // maps from ar.passthrough_brightness
  passthroughContrast: number;  // maps from ar.passthrough_contrast
  enablePassthroughPortal: boolean;  // maps from ar.enable_passthrough_portal
  portalSize: number;  // maps from ar.portal_size
  portalEdgeColor: string;  // maps from ar.portal_edge_color
  portalEdgeWidth: number;  // maps from ar.portal_edge_width
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

export type BinaryNodeUpdate = {
  nodeId: string;
  data: NodeData;
};

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
    nodes: BinaryNodeUpdate[];
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

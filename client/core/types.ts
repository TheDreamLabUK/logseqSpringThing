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

// Settings interfaces in camelCase
export interface AnimationSettings {
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  motionBlurStrength: number;
  selectionWaveEnabled: boolean;
  pulseEnabled: boolean;
  rippleEnabled: boolean;
  edgeAnimationEnabled: boolean;
  flowParticlesEnabled: boolean;
}

export interface ARSettings {
  dragThreshold: number;
  enableHandTracking: boolean;
  enableHaptics: boolean;
  enableLightEstimation: boolean;
  enablePassthroughPortal: boolean;
  enablePlaneDetection: boolean;
  enableSceneUnderstanding: boolean;
  gestureSsmoothing: number;
  handMeshColor: string;
  handMeshEnabled: boolean;
  handMeshOpacity: number;
  handPointSize: number;
  handRayColor: string;
  handRayEnabled: boolean;
  handRayWidth: number;
  hapticIntensity: number;
  passthroughBrightness: number;
  passthroughContrast: number;
  passthroughOpacity: number;
  pinchThreshold: number;
  planeColor: string;
  planeOpacity: number;
  portalEdgeColor: string;
  portalEdgeWidth: number;
  portalSize: number;
  roomScale: boolean;
  rotationThreshold: number;
  showPlaneOverlay: boolean;
  snapToFloor: boolean;
}

export interface AudioSettings {
  enableAmbientSounds: boolean;
  enableInteractionSounds: boolean;
  enableSpatialAudio: boolean;
}

export interface BloomSettings {
  edgeBloomStrength: number;
  enabled: boolean;
  environmentBloomStrength: number;
  nodeBloomStrength: number;
  radius: number;
  strength: number;
}

export interface ClientDebugSettings {
  enableDataDebug: boolean;
  enableWebsocketDebug: boolean;
  enabled: boolean;
  logBinaryHeaders: boolean;
  logFullJson: boolean;
}

export interface DefaultSettings {
  apiClientTimeout: number;
  enableMetrics: boolean;
  enableRequestLogging: boolean;
  logFormat: string;
  logLevel: string;
  maxConcurrentRequests: number;
  maxPayloadSize: number;
  maxRetries: number;
  metricsPort: number;
  retryDelay: number;
}

export interface EdgeSettings {
  arrowSize: number;
  baseWidth: number;
  color: string;
  enableArrows: boolean;
  opacity: number;
  widthRange: [number, number];
}

export interface GithubSettings {
  basePath: string;
  owner: string;
  rateLimit: boolean;
  repo: string;
  token: string;
}

export interface LabelSettings {
  desktopFontSize: number;
  enableLabels: boolean;
  textColor: string;
}

export interface NetworkSettings {
  bindAddress: string;
  domain: string;
  enableHttp2: boolean;
  enableRateLimiting: boolean;
  enableTls: boolean;
  maxRequestSize: number;
  minTlsVersion: string;
  port: number;
  rateLimitRequests: number;
  rateLimitWindow: number;
  tunnelId: string;
}

export interface NodeSettings {
  baseColor: string;
  baseSize: number;
  clearcoat: number;
  enableHoverEffect: boolean;
  enableInstancing: boolean;
  highlightColor: string;
  highlightDuration: number;
  hoverScale: number;
  materialType: string;
  metalness: number;
  opacity: number;
  roughness: number;
  sizeByConnections: boolean;
  sizeRange: [number, number];
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
  prompt: string;
  rateLimit: number;
  presencePenalty: number;
  temperature: number;
  timeout: number;
  topP: number;
}

export interface PhysicsSettings {
  attractionStrength: number;
  boundsSize: number;
  collisionRadius: number;
  damping: number;
  enableBounds: boolean;
  enabled: boolean;
  iterations: number;
  maxVelocity: number;
  repulsionStrength: number;
  springStrength: number;
}

export interface RagflowSettings {
  apiKey: string;
  baseUrl: string;
  maxRetries: number;
  timeout: number;
}

export interface RenderingSettings {
  ambientLightIntensity: number;
  backgroundColor: string;
  directionalLightIntensity: number;
  enableAmbientOcclusion: boolean;
  enableAntialiasing: boolean;
  enableShadows: boolean;
  environmentIntensity: number;
}

export interface SecuritySettings {
  allowedOrigins: string[];
  auditLogPath: string;
  cookieHttponly: boolean;
  cookieSamesite: string;
  cookieSecure: boolean;
  csrfTokenTimeout: number;
  enableAuditLogging: boolean;
  enableRequestValidation: boolean;
  sessionTimeout: number;
}

export interface ServerDebugSettings {
  enableDataDebug: boolean;
  enableWebsocketDebug: boolean;
  enabled: boolean;
  logBinaryHeaders: boolean;
  logFullJson: boolean;
}

export interface WebsocketSettings {
  binaryChunkSize: number;
  compressionEnabled: boolean;
  compressionThreshold: number;
  heartbeatInterval: number;
  heartbeatTimeout: number;
  maxConnections: number;
  maxMessageSize: number;
}

export interface Settings {
  animations: AnimationSettings;
  ar: ARSettings;
  audio: AudioSettings;
  bloom: BloomSettings;
  clientDebug: ClientDebugSettings;
  default: DefaultSettings;
  edges: EdgeSettings;
  github: GithubSettings;
  labels: LabelSettings;
  network: NetworkSettings;
  nodes: NodeSettings;
  openai: OpenAISettings;
  perplexity: PerplexitySettings;
  physics: PhysicsSettings;
  ragflow: RagflowSettings;
  rendering: RenderingSettings;
  security: SecuritySettings;
  serverDebug: ServerDebugSettings;
  websocket: WebsocketSettings;
}

export type SettingCategory = keyof Settings;
export type SettingKey<T extends SettingCategory> = keyof Settings[T];
export type SettingValue = string | number | boolean | number[] | string[];

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
    settings: Settings;
  };
}

export interface SettingsUpdatedMessage extends WebSocketMessage {
  type: 'settingsUpdated';
  data: {
    settings: Settings;
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

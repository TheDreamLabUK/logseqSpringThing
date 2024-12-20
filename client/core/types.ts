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

export interface EdgeSettings {
  arrowSize: number;
  baseWidth: number;
  color: string;
  enableArrows: boolean;
  opacity: number;
  widthRange: [number, number];
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

export interface RenderingSettings {
  ambientLightIntensity: number;
  backgroundColor: string;
  directionalLightIntensity: number;
  enableAmbientOcclusion: boolean;
  enableAntialiasing: boolean;
  enableShadows: boolean;
  environmentIntensity: number;
}

export interface WebsocketSettings {
  binaryChunkSize: number;
  compressionEnabled: boolean;
  compressionThreshold: number;
  heartbeatInterval: number;
  heartbeatTimeout: number;
  maxConnections: number;
  maxMessageSize: number;
  reconnectAttempts: number;
  reconnectDelay: number;
  updateRate: number;
}

export interface Settings {
  animations: AnimationSettings;
  ar: ARSettings;
  audio: AudioSettings;
  bloom: BloomSettings;
  clientDebug: ClientDebugSettings;
  default: DefaultSettings;
  edges: EdgeSettings;
  labels: LabelSettings;
  network: NetworkSettings;
  nodes: NodeSettings;
  physics: PhysicsSettings;
  rendering: RenderingSettings;
  security: SecuritySettings;
  serverDebug: ServerDebugSettings;
  websocket: WebsocketSettings;
}

export type SettingCategory = keyof Settings;
export type SettingKey<T extends SettingCategory> = keyof Settings[T];
export type SettingValue = string | number | boolean | number[] | string[];

// WebSocket specific error types
export enum WebSocketErrorType {
  CONNECTION_FAILED = 'CONNECTION_FAILED',
  CONNECTION_LOST = 'CONNECTION_LOST',
  MAX_RETRIES_EXCEEDED = 'MAX_RETRIES_EXCEEDED',
  MESSAGE_PARSE_ERROR = 'MESSAGE_PARSE_ERROR',
  SEND_FAILED = 'SEND_FAILED',
  INVALID_MESSAGE = 'INVALID_MESSAGE',
  TIMEOUT = 'TIMEOUT'
}

export interface WebSocketError extends Error {
  type: WebSocketErrorType;
  code?: number;
  details?: any;
}

// WebSocket connection status
export enum WebSocketStatus {
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  DISCONNECTED = 'DISCONNECTED',
  RECONNECTING = 'RECONNECTING',
  FAILED = 'FAILED'
}

// WebSocket message types
export type MessageType = 
  | 'initialData'
  | 'binaryPositionUpdate'
  | 'requestInitialData'
  | 'enableBinaryUpdates'
  | 'ping'
  | 'pong'
  | 'settingsUpdated'
  | 'graphUpdated'
  | 'connectionStatus'
  | 'updatePositions'
  | 'simulationModeSet';

// Handler types
export type MessageHandler = (data: any) => void;
export type ErrorHandler = (error: WebSocketError) => void;
export type ConnectionHandler = (status: WebSocketStatus, details?: any) => void;

// Base WebSocket message interface
export interface BaseWebSocketMessage {
  type: MessageType;
}

export interface InitialDataMessage extends BaseWebSocketMessage {
  type: 'initialData';
  data: {
    nodes: Node[];
    edges: Edge[];
  };
}

export interface BinaryPositionUpdateMessage extends BaseWebSocketMessage {
  type: 'binaryPositionUpdate';
  data: {
    nodes: {
      nodeId: string;
      data: NodeData;
    }[];
  };
}

export interface RequestInitialDataMessage extends BaseWebSocketMessage {
  type: 'requestInitialData';
}

export interface EnableBinaryUpdatesMessage extends BaseWebSocketMessage {
  type: 'enableBinaryUpdates';
  data: {
    enabled: boolean;
  };
}

export interface PingMessage extends BaseWebSocketMessage {
  type: 'ping';
  timestamp: number;
}

export interface PongMessage extends BaseWebSocketMessage {
  type: 'pong';
  timestamp: number;
}

export interface ConnectionStatusMessage extends BaseWebSocketMessage {
  type: 'connectionStatus';
  status: WebSocketStatus;
  details?: any;
}

export type WebSocketMessage =
  | InitialDataMessage
  | BinaryPositionUpdateMessage
  | RequestInitialDataMessage
  | EnableBinaryUpdatesMessage
  | PingMessage
  | PongMessage
  | ConnectionStatusMessage;

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

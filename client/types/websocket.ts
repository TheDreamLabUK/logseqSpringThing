// Message Types (matching server's ServerMessage enum)
export type MessageType = 
  | 'graphUpdate'
  | 'error'
  | 'positionUpdateComplete'
  | 'settingsUpdated'
  | 'simulationModeSet'
  | 'fisheyeSettingsUpdated'
  | 'initialData'
  | 'gpuState'
  | 'layoutState'
  | 'audioData'
  | 'updateSettings'
  | 'openaiResponse'
  | 'ragflowResponse'
  | 'completion'
  | 'ping'
  | 'pong';

// Binary Protocol Types
export interface BinaryMessage {
  data: ArrayBuffer;        // Raw binary data in format:
                           // [x,y,z,vx,vy,vz](24) per node
                           // Node index in array matches index in original graph data
  positions: NodePosition[];  // Processed position data
  nodeCount: number;        // Number of nodes in the update
}

export interface NodePosition {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}

// Graph Data (establishes node order for binary updates)
export interface GraphData {
  nodes: Node[];  // Order of nodes here determines binary update indices
  edges: Edge[];
  metadata?: Record<string, any>;
}

export interface Node {
  id: string;
  label?: string;
  position?: [number, number, number];
  velocity?: [number, number, number];
  size?: number;
  color?: string;
  type?: string;
  metadata?: Record<string, any>;
  userData?: Record<string, any>;
  weight?: number;
  group?: string;
}

export interface Edge {
  id: string;  // Added to match core Edge type
  source: string;
  target: string;
  weight?: number;
  width?: number;
  color?: string;
  type?: string;
  metadata?: Record<string, any>;
  userData?: Record<string, any>;
  directed?: boolean;
}

// Graph Update Message
export interface GraphUpdateMessage extends BaseMessage {
  type: 'graphUpdate';
  graphData: GraphData;
}

// Base Message Interface
export interface BaseMessage {
  type: MessageType;
  [key: string]: any;
}

// Settings Interfaces
export interface FisheyeSettings {
  enabled: boolean;
  strength: number;
  focusPoint: [number, number, number];
  radius: number;
}

export interface MaterialSettings {
  nodeSize: number;
  nodeColor: string;
  edgeWidth: number;
  edgeColor: string;
  highlightColor: string;
  opacity: number;
}

export interface PhysicsSettings {
  gravity: number;
  springLength: number;
  springStrength: number;
  repulsion: number;
  damping: number;
  timeStep: number;
}

export interface BloomSettings {
  enabled: boolean;
  strength: number;
  radius: number;
  threshold: number;
}

// Message Type Interfaces (matching server's ServerMessage variants)
export interface FisheyeUpdateMessage extends BaseMessage {
  type: 'fisheyeSettingsUpdated';
  enabled: boolean;
  strength: number;
  focusPoint: [number, number, number];
  radius: number;
}

export interface ErrorMessage extends BaseMessage {
  type: 'error';
  message: string;
  details?: string;
  code?: string;
}

export interface AudioMessage extends BaseMessage {
  type: 'audioData';
  audioData: Blob;
}

export interface RagflowResponse extends BaseMessage {
  type: 'ragflowResponse';
  answer: string;
  audio?: string;
}

export interface SimulationModeMessage extends BaseMessage {
  type: 'simulationModeSet';
  mode: string;
  gpuEnabled: boolean;
}

export interface SettingsUpdateMessage extends BaseMessage {
  type: 'updateSettings';
  settings: {
    material?: Partial<MaterialSettings>;
    physics?: Partial<PhysicsSettings>;
    bloom?: Partial<BloomSettings>;
    fisheye?: Partial<FisheyeSettings>;
  };
}

export interface SettingsUpdatedMessage extends BaseMessage {
  type: 'settingsUpdated';
  settings: {
    material?: MaterialSettings;
    physics?: PhysicsSettings;
    bloom?: BloomSettings;
    fisheye?: FisheyeSettings;
  };
}

// Heartbeat Messages
export interface PingMessage extends BaseMessage {
  type: 'ping';
}

export interface PongMessage extends BaseMessage {
  type: 'pong';
}

// WebSocket Configuration
export interface WebSocketConfig {
  messageRateLimit: number;
  messageTimeWindow: number;
  maxMessageSize: number;
  maxAudioSize: number;
  maxQueueSize: number;
  maxRetries: number;
  retryDelay: number;
}

// Event System Types
export type WebSocketEventMap = {
  open: void;
  close: void;
  error: ErrorMessage;
  message: BaseMessage;
  graphUpdate: GraphUpdateMessage;
  serverSettings: Record<string, any>;
  ragflowAnswer: string;
  openaiResponse: string;
  simulationModeSet: string;
  completion: string;
  positionUpdateComplete: string;
  gpuPositions: BinaryMessage;
  maxReconnectAttemptsReached: void;
};

export type WebSocketEventCallback<T> = (data: T) => void;

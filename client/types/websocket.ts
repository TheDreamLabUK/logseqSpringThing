// Message Types
export type MessageType = 
  | 'initial_data'
  | 'graphUpdate'
  | 'audioData'
  | 'answer'
  | 'error'
  | 'ragflowResponse'
  | 'openaiResponse'
  | 'simulationModeSet'
  | 'fisheye_settings_updated'
  | 'completion'
  | 'position_update_complete'
  | 'graphData'
  | 'visualSettings'
  | 'materialSettings'
  | 'physicsSettings'
  | 'bloomSettings'
  | 'fisheyeSettings'
  | 'updateSettings'
  | 'settings_updated'
  | 'chatMessage'
  | 'setTTSMethod'
  | 'updateNodePosition'
  | 'updateNodeVelocity';

// Binary Protocol Types
export interface PositionUpdate {
  id: string;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}

export interface BinaryMessage {
  isInitialLayout: boolean;
  positions: PositionUpdate[];
}

// WebSocket Message Interfaces
export interface BaseMessage {
  type: MessageType;
  [key: string]: any;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>;
}

export interface Node {
  id: string;
  label?: string;
  position?: [number, number, number];
  velocity?: [number, number, number];
  [key: string]: any;
}

export interface Edge {
  source: string;
  target: string;
  [key: string]: any;
}

// Update GraphUpdateMessage to handle both camelCase and snake_case
export interface GraphUpdateMessage extends BaseMessage {
  type: 'graphUpdate' | 'graphData';
  graphData?: GraphData;  // camelCase version
  graph_data?: GraphData; // snake_case version from server
}

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

export interface FisheyeUpdateMessage extends BaseMessage {
  type: 'fisheye_settings_updated';
  fisheye_enabled: boolean;
  fisheye_strength: number;
  fisheye_focus_x: number;
  fisheye_focus_y: number;
  fisheye_focus_z: number;
  fisheye_radius: number;
}

export interface ErrorMessage extends BaseMessage {
  type: 'error';
  message: string;
  details?: string;
}

export interface AudioMessage extends BaseMessage {
  type: 'audioData';
  audio_data: Blob;
}

export interface RagflowResponse extends BaseMessage {
  type: 'ragflowResponse';
  answer: string;
  audio?: string;
}

export interface SimulationModeMessage extends BaseMessage {
  type: 'simulationModeSet';
  mode: string;
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
  type: 'settings_updated';
  settings: {
    material?: MaterialSettings;
    physics?: PhysicsSettings;
    bloom?: BloomSettings;
    fisheye?: FisheyeSettings;
  };
}

// WebSocket Service Configuration Types
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

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
  | 'simulation_mode_set'
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
  | 'updateNodeVelocity'
  | 'layout_state'
  | 'gpu_state';

// Binary Protocol Types
export interface BinaryMessage {
  data: ArrayBuffer;        // Raw binary data in format:
                           // [isInitial(4)] + [x,y,z,vx,vy,vz](24) per node
                           // Node index in array matches index in original graph data
  isInitialLayout: boolean; // First 4 bytes flag
  nodeCount: number;        // Number of nodes in the update
}

// Node Position Update
export interface NodePositionUpdate {
  nodeIndex: number;      // Index in the nodes array
  position: [number, number, number];
  velocity?: [number, number, number];
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

// Message Type Interfaces
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
  code?: string;
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
  type: 'simulation_mode_set';
  mode: string;
  gpu_enabled?: boolean;
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

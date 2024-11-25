import type { Node, Edge } from './core';
import type { VisualizationConfig, BloomConfig, FisheyeConfig } from './components';
import type { BaseMessage } from './websocket';

// Visualization Store Types
export interface VisualizationState {
  nodes: Node[];
  edges: Edge[];
  metadata: Record<string, any>;
  selectedNode: string | null;
  hoveredNode: string | null;
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  error: string | null;
  isLoading: boolean;
}

// Binary Update Store Types
export interface BinaryUpdateState {
  positions: Float32Array | null;
  velocities: Float32Array | null;
  isProcessing: boolean;
  updateCount: number;
  lastUpdateTime: number;
}

// WebSocket Store Types
export interface WebSocketState {
  isConnected: boolean;
  reconnectAttempts: number;
  lastError: string | null;
  messageQueue: BaseMessage[];
  graphData: {
    nodes: Node[];
    edges: Edge[];
    metadata: Record<string, any>;
  } | null;
}

// Settings Store Types
export interface SettingsState {
  visualization: VisualizationConfig;
  bloom: BloomConfig;
  fisheye: FisheyeConfig;
  isDirty: boolean;
}

// Store Getters Types
export interface VisualizationGetters {
  getNodes: () => Node[];
  getEdges: () => Edge[];
  getMetadata: () => Record<string, any>;
  getSelectedNode: () => string | null;
  getHoveredNode: () => string | null;
  getCameraPosition: () => [number, number, number];
  getCameraTarget: () => [number, number, number];
  getError: () => string | null;
  getIsLoading: () => boolean;
}

export interface BinaryUpdateGetters {
  getPositionBuffer: () => Float32Array | null;
  getVelocityBuffer: () => Float32Array | null;
  getIsProcessing: () => boolean;
  getUpdateCount: () => number;
  getLastUpdateTime: () => number;
}

export interface WebSocketGetters {
  getIsConnected: () => boolean;
  getReconnectAttempts: () => number;
  getLastError: () => string | null;
  getMessageQueue: () => BaseMessage[];
  getGraphData: () => {
    nodes: Node[];
    edges: Edge[];
    metadata: Record<string, any>;
  } | null;
}

export interface SettingsGetters {
  getVisualizationSettings: () => VisualizationConfig;
  getBloomSettings: () => BloomConfig;
  getFisheyeSettings: () => FisheyeConfig;
  hasUnsavedChanges: () => boolean;
}

// Store Actions Types
export interface VisualizationActions {
  setGraphData(nodes: Node[], edges: Edge[], metadata?: Record<string, any>): void;
  updateNodePosition(nodeId: string, position: [number, number, number]): void;
  selectNode(nodeId: string | null): void;
  setHoveredNode(nodeId: string | null): void;
  setCameraPosition(position: [number, number, number]): void;
  setCameraTarget(target: [number, number, number]): void;
  setError(error: string | null): void;
  setLoading(loading: boolean): void;
  reset(): void;
}

export interface BinaryUpdateActions {
  processPositionUpdate(buffer: ArrayBuffer): void;
  reset(): void;
}

export interface WebSocketActions {
  setConnected(connected: boolean): void;
  incrementReconnectAttempts(): void;
  setError(error: string | null): void;
  queueMessage(message: BaseMessage): void;
  clearMessageQueue(): void;
  setGraphData(data: { nodes: Node[]; edges: Edge[]; metadata: Record<string, any> }): void;
  reset(): void;
}

export interface SettingsActions {
  updateVisualizationSettings(settings: Partial<VisualizationConfig>): void;
  updateBloomSettings(settings: Partial<BloomConfig>): void;
  updateFisheyeSettings(settings: Partial<FisheyeConfig>): void;
  applyServerSettings(settings: {
    visualization?: Partial<VisualizationConfig>;
    bloom?: Partial<BloomConfig>;
    fisheye?: Partial<FisheyeConfig>;
  }): void;
  resetToDefaults(): void;
  markSaved(): void;
}

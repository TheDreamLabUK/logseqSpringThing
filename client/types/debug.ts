/**
 * Debug panel types
 */

export interface DebugMetrics {
  cacheHitRate: number;
  updateInterval: number;
  fps: number;
  nodeCount: number;
  edgeCount: number;
  positionUpdates: number;
  messageCount: number;
  queueSize: number;
}

export interface SampleNodeData {
  id: string;
  position?: [number, number, number];
  velocity?: [number, number, number];
  edgeCount: number;
  weight?: number;
  group?: string;
}

export interface WebSocketStatus {
  connected: boolean;
  lastMessageTime: number;
  messageCount: number;
  queueSize: number;
  pendingUpdates: number;
}

export interface BinaryUpdateStatus {
  lastUpdateTime: number;
  activePositions: number;
  isInitialLayout: boolean;
  pendingCount: number;
}

export interface DebugPanelState {
  isVisible: boolean;
  isExpanded: boolean;
  metrics: DebugMetrics;
  sampleNode: SampleNodeData | null;
  wsStatus: WebSocketStatus;
  binaryStatus: BinaryUpdateStatus;
}

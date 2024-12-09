import type { Node, Edge } from './core';
import type { 
  BaseMessage, 
  MaterialSettings, 
  PhysicsSettings, 
  BloomSettings, 
  FisheyeSettings 
} from './websocket';

// Re-export Node and Edge types
export type { Node, Edge };

// Binary message format types (no state storage needed)
export type BinaryMessageHeader = {
  timeStep: number;
  nodeCount: number;
};

export type BinaryPositionUpdate = {
  nodeId: string;
  position: [number, number, number];
};

// WebSocket Store State
export interface WebSocketState {
  isConnected: boolean;
  reconnectAttempts: number;
  messageQueue: (BaseMessage | ArrayBuffer)[];
  lastError: string | null;
  graphData: {
    nodes: Node[];
    edges: Edge[];
    metadata: Record<string, any>;
  } | null;
}

// Visualization Store State
export interface VisualizationState {
  nodes: Node[];
  edges: Edge[];
  metadata: Record<string, any>;
  selectedNode: Node | null;
  hoveredNode: Node | null;
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  isLoading: boolean;
  error: string | null;
  renderSettings: {
    nodeSize: number;
    nodeColor: string;
    edgeWidth: number;
    edgeColor: string;
    highlightColor: string;
    opacity: number;
    bloom: {
      enabled: boolean;
      strength: number;
      radius: number;
      threshold: number;
    };
    fisheye: {
      enabled: boolean;
      strength: number;
      focusPoint: [number, number, number];
      radius: number;
    };
  };
  physicsSettings: {
    enabled: boolean;
    gravity: number;
    springLength: number;
    springStrength: number;
    repulsion: number;
    damping: number;
    timeStep: number;
  };
}

// Settings Store State
export interface SettingsState {
  visualization: MaterialSettings;
  physics: PhysicsSettings;
  bloom: BloomSettings;
  fisheye: FisheyeSettings;
  audio: {
    enabled: boolean;
    volume: number;
    useOpenAI: boolean;
    ttsEnabled: boolean;
  };
  performance: {
    gpuAcceleration: boolean;
    maxFPS: number;
    quality: 'low' | 'medium' | 'high';
    autoAdjust: boolean;
  };
  debug: {
    showStats: boolean;
    logLevel: 'error' | 'warn' | 'info' | 'debug';
    showGrid: boolean;
    showAxes: boolean;
  };
}

// Store Actions
export interface WebSocketActions {
  connect(): Promise<void>;
  disconnect(): void;
  send(message: BaseMessage | ArrayBuffer): void;
  handleMessage(message: BaseMessage): void;
  handleBinaryMessage(data: ArrayBuffer): void;
}

export interface VisualizationActions {
  updateNodePositions(updates: Array<[string, [number, number, number]]>): void;
  setGraphData(nodes: Node[], edges: Edge[], metadata: Record<string, any>): void;
  selectNode(node: Node | null): void;
  hoverNode(node: Node | null): void;
  updateCamera(position: [number, number, number], target: [number, number, number]): void;
  updateRenderSettings(settings: Partial<VisualizationState['renderSettings']>): void;
  updatePhysicsSettings(settings: Partial<VisualizationState['physicsSettings']>): void;
  startAnimation(): void;
  stopAnimation(): void;
  updatePerformanceMetrics(): void;
}

export interface SettingsActions {
  updateVisualization(settings: Partial<MaterialSettings>): void;
  updatePhysics(settings: Partial<PhysicsSettings>): void;
  updateBloom(settings: Partial<BloomSettings>): void;
  updateFisheye(settings: Partial<FisheyeSettings>): void;
  updateAudio(settings: Partial<SettingsState['audio']>): void;
  updatePerformance(settings: Partial<SettingsState['performance']>): void;
  updateDebug(settings: Partial<SettingsState['debug']>): void;
  resetToDefaults(): void;
  applyServerSettings(settings: {
    visualization?: Partial<MaterialSettings>;
    physics?: Partial<PhysicsSettings>;
    bloom?: Partial<BloomSettings>;
    fisheye?: Partial<FisheyeSettings>;
  }): void;
}

// Store Getters
export interface WebSocketGetters {
  isConnected: boolean;
  hasError: boolean;
  queueLength: number;
  graphData: WebSocketState['graphData'];
}

export interface VisualizationGetters {
  selectedNode: Node | null;
  hoveredNode: Node | null;
  nodeCount: number;
  edgeCount: number;
  fps: number;
  isPerformant: boolean;
}

export interface SettingsGetters {
  isGPUAccelerated: boolean;
  currentQuality: SettingsState['performance']['quality'];
  debugEnabled: boolean;
  audioEnabled: boolean;
}

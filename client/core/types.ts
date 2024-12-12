/**
 * Core types for graph visualization
 */

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface Node {
  id: string;
  label: string;
  position: Vector3;
  color?: string;
  size?: number;
}

export interface Edge {
  source: string;
  target: string;
  weight?: number;
  color?: string;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>;
}

export interface VisualizationSettings {
  // Node appearance
  nodeSize: number;
  nodeColor: string;
  nodeOpacity: number;
  nodeHighlightColor: string;
  
  // Edge appearance
  edgeWidth: number;
  edgeColor: string;
  edgeOpacity: number;
  
  // Visual effects
  enableBloom: boolean;
  bloomIntensity: number;
  bloomThreshold: number;
  bloomRadius: number;
  
  // Performance
  maxFps: number;
  updateThrottle: number;

  // Labels
  showLabels: boolean;
  labelSize: number;
  labelColor: string;

  // XR specific
  xrControllerVibration: boolean;
  xrControllerHapticIntensity: number;
}

// WebSocket message types
export type MessageType = 
  | 'initialData'
  | 'graphUpdate'
  | 'binaryPositionUpdate'
  | 'settingsUpdate'
  | 'error'
  | 'ping';

export interface WebSocketMessage {
  type: MessageType;
  data: any;
}

// Platform detection types
export type Platform = 'browser' | 'quest';

export interface PlatformCapabilities {
  xrSupported: boolean;
  webglSupported: boolean;
  websocketSupported: boolean;
}

// Message queue types
export interface QueuedMessage {
  data: ArrayBuffer;
  timestamp: number;
}

/**
 * Core types for the LogseqXR visualization system
 */

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface Transform {
  position: Vector3;
  rotation: Vector3;
  scale: Vector3;
}

export interface Node {
  id: string;
  label: string;
  position: Vector3;
  velocity: Vector3;
  mass: number;
  color?: string;
  size?: number;
  metadata?: Record<string, any>;
}

export interface Edge {
  source: string;
  target: string;
  weight?: number;
  color?: string;
  metadata?: Record<string, any>;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>;
}

export interface Viewport {
  width: number;
  height: number;
  devicePixelRatio: number;
}

export interface VisualizationSettings {
  // Node appearance
  nodeSize: number;
  nodeColor: string;
  nodeOpacity: number;
  nodeHighlightColor: string;
  
  // Node material properties
  nodeMaterialMetalness: number;  // 0-1
  nodeMaterialRoughness: number;  // 0-1
  nodeMaterialEmissiveIntensity: number;  // 0-1
  nodeMaterialClearcoat: number;  // 0-1
  nodeMaterialClearcoatRoughness: number;  // 0-1
  nodeMaterialReflectivity: number;  // 0-1
  nodeMaterialEnvMapIntensity: number;  // 0-1
  
  // Edge appearance
  edgeWidth: number;
  edgeColor: string;
  edgeOpacity: number;
  
  // Physics settings
  gravity: number;
  springLength: number;
  springStiffness: number;
  charge: number;
  damping: number;
  
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
  | 'ping';  // Changed from 'heartbeat' to 'ping'

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

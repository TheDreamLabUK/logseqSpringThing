/**
 * Core types for graph visualization
 */

// Base types
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

// Binary protocol types
export interface BinaryHeader {
  version: number;
  nodeCount: number;
}

export interface BinaryNodeData {
  position: Float32Array; // x, y, z
  velocity: Float32Array; // vx, vy, vz
}

// Raw types (matching server format)
export interface RawNodeData {
  position: [number, number, number];
  velocity: [number, number, number];
  mass: number;
  flags: number;
}

export interface RawNode {
  id: string;
  label: string;
  data: RawNodeData;
  metadata?: Record<string, string>;
  nodeType?: string;
  size?: number;
  color?: string;
  weight?: number;
  group?: string;
  userData?: Record<string, string>;
}

export interface RawGraphData {
  nodes: RawNode[];
  edges: Edge[];
  metadata?: Record<string, any>;
}

// Transformed types (used in client)
export interface NodeData {
  position: Vector3;
  velocity: Vector3;
  mass: number;
  flags: number;
}

export interface Node {
  id: string;
  label: string;
  data: NodeData;
  metadata?: Record<string, string>;
  nodeType?: string;
  size?: number;
  color?: string;
  weight?: number;
  group?: string;
  userData?: Record<string, string>;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>;
}

// Performance types
export interface UpdateBatch {
  timestamp: number;
  updates: Float32Array;
}

export interface EdgeUpdateBatch {
  edgeIndices: Set<number>;
  timestamp: number;
}

// Shared types
export interface Edge {
  source: string;
  target: string;
  weight: number;
  edgeType?: string;
  metadata?: Record<string, string>;
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

// Transform functions
export function arrayToVector3(arr: [number, number, number]): Vector3 {
  return { x: arr[0], y: arr[1], z: arr[2] };
}

export function vector3ToArray(vec: Vector3): [number, number, number] {
  return [vec.x, vec.y, vec.z];
}

export function float32ArrayToVector3(arr: Float32Array, offset: number = 0): Vector3 {
  return {
    x: arr[offset],
    y: arr[offset + 1],
    z: arr[offset + 2]
  };
}

export function vector3ToFloat32Array(vec: Vector3, arr: Float32Array, offset: number = 0): void {
  arr[offset] = vec.x;
  arr[offset + 1] = vec.y;
  arr[offset + 2] = vec.z;
}

export function transformNodeData(raw: RawNodeData): NodeData {
  return {
    position: arrayToVector3(raw.position),
    velocity: arrayToVector3(raw.velocity),
    mass: raw.mass,
    flags: raw.flags
  };
}

export function transformNode(raw: RawNode): Node {
  return {
    ...raw,
    data: transformNodeData(raw.data)
  };
}

export function transformGraphData(raw: RawGraphData): GraphData {
  return {
    nodes: raw.nodes.map(transformNode),
    edges: raw.edges,
    metadata: raw.metadata
  };
}

// Binary data validation
export function validateBinaryHeader(data: ArrayBuffer): BinaryHeader | null {
  if (data.byteLength < 8) return null; // Minimum size for version and count
  const view = new Float32Array(data);
  return {
    version: view[0],
    nodeCount: Math.floor((view.length - 1) / 6) // (length - version) / floats per node
  };
}

// WebSocket message types
export type MessageType = 
  | 'initialData'
  | 'requestInitialData'
  | 'binaryPositionUpdate'
  | 'settingsUpdated'
  | 'updateSettings'
  | 'enableBinaryUpdates'
  | 'ping'
  | 'pong';

// Raw message types (from server)
export interface RawInitialDataMessage {
  type: 'initialData';
  data: {
    graph: RawGraphData;
  };
}

export interface RawBinaryNodeUpdate {
  nodeId: string;
  data: RawNodeData;
}

export interface RawBinaryPositionUpdateMessage {
  type: 'binaryPositionUpdate';
  data: {
    nodes: RawBinaryNodeUpdate[];
  };
}

// Transformed message types (for client use)
export interface InitialDataMessage {
  type: 'initialData';
  data: {
    graph: GraphData;
  };
}

export interface BinaryNodeUpdate {
  nodeId: string;
  data: NodeData;
}

export interface BinaryPositionUpdateMessage {
  type: 'binaryPositionUpdate';
  data: {
    nodes: BinaryNodeUpdate[];
  };
}

// Other message types
export interface RequestInitialDataMessage {
  type: 'requestInitialData';
}

export interface EnableBinaryUpdatesMessage {
  type: 'enableBinaryUpdates';
}

export interface SettingsUpdateMessage {
  type: 'settingsUpdated';
data: {
    settings: VisualizationSettings;
  };
}

export interface UpdateSettingsMessage {
  type: 'updateSettings';
data: {
    settings: Partial<VisualizationSettings>;
  };
}

export interface PingMessage {
  type: 'ping';
}

export interface PongMessage {
  type: 'pong';
}

// Union types for messages
export type RawWebSocketMessage =
  | RawInitialDataMessage
  | RawBinaryPositionUpdateMessage
  | SettingsUpdateMessage
  | UpdateSettingsMessage
  | RequestInitialDataMessage
  | EnableBinaryUpdatesMessage
  | PingMessage
  | PongMessage;

export type WebSocketMessage =
  | InitialDataMessage
  | BinaryPositionUpdateMessage
  | SettingsUpdateMessage
  | UpdateSettingsMessage
  | RequestInitialDataMessage
  | EnableBinaryUpdatesMessage
  | PingMessage
  | PongMessage;

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

// Debug types
export interface NetworkDebugMessage {
  direction: 'in' | 'out';
  type: 'binary' | 'json';
  timestamp: number;
  data: any;
}

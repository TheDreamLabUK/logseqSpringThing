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

export interface BinaryNodeUpdate {
  nodeId: string;
  data: NodeData;
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

// Platform types
export type Platform = 'browser' | 'quest';

export interface PlatformCapabilities {
  xrSupported: boolean;
  webglSupported: boolean;
  websocketSupported: boolean;
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

export interface WebSocketMessage {
  type: MessageType;
  data?: any;
}

export interface RawWebSocketMessage {
  type: MessageType;
  data?: any;
}

export interface InitialDataMessage {
  type: 'initialData';
  data: {
    graph: GraphData;
  };
}

export interface RawInitialDataMessage {
  type: 'initialData';
  data: {
    graph: RawGraphData;
  };
}

export interface BinaryPositionUpdateMessage {
  type: 'binaryPositionUpdate';
  data: {
    nodes: BinaryNodeUpdate[];
  };
}

export interface RawBinaryPositionUpdateMessage {
  type: 'binaryPositionUpdate';
  data: {
    nodes: RawBinaryNodeUpdate[];
  };
}

export interface RawBinaryNodeUpdate {
  nodeId: string;
  data: RawNodeData;
}

export interface RequestInitialDataMessage {
  type: 'requestInitialData';
}

export interface EnableBinaryUpdatesMessage {
  type: 'enableBinaryUpdates';
}

export interface PingMessage {
  type: 'ping';
}

export interface UpdateSettingsMessage {
  type: 'updateSettings';
  data: {
    settings: ServerSettings;
  };
}

export interface SettingsUpdatedMessage {
  type: 'settingsUpdated';
  data: {
    settings: ServerSettings;
  };
}

// Settings types
export interface VisualizationSettings {
    // Node Appearance
    nodeSize: number;
    nodeColor: string;
    nodeOpacity: number;
    metalness: number;
    roughness: number;
    clearcoat: number;

    // Edge Appearance
    edgeWidth: number;
    edgeColor: string;
    edgeOpacity: number;
    enableArrows: boolean;
    arrowSize: number;

    // Visual Effects
    enableBloom: boolean;
    bloomIntensity: number;
    bloomRadius: number;
    enableNodeAnimations: boolean;
    enableMotionBlur: boolean;
    motionBlurStrength: number;

    // Labels
    showLabels: boolean;
    labelSize: number;
    labelColor: string;

    // Performance
    maxFps: number;

    // AR Settings
    enablePlaneDetection: boolean;
    enableSceneUnderstanding: boolean;
    showPlaneOverlay: boolean;
    planeOpacity: number;
    planeColor: string;
    enableLightEstimation: boolean;
    enableHandTracking: boolean;
    handMeshEnabled: boolean;
    handMeshColor: string;
    handMeshOpacity: number;
    handRayEnabled: boolean;
    handRayColor: string;
    handRayWidth: number;
    handPointSize: number;
    gestureSmoothing: number;
    pinchThreshold: number;
    dragThreshold: number;
    rotationThreshold: number;
    enableHaptics: boolean;
    hapticIntensity: number;
    roomScale: boolean;
    snapToFloor: boolean;
    passthroughOpacity: number;
    passthroughBrightness: number;
    passthroughContrast: number;
    enablePassthroughPortal: boolean;
    portalSize: number;
    portalEdgeColor: string;
    portalEdgeWidth: number;
}

// Server-side settings interface
export interface ServerSettings {
    nodes: {
        base_size: number;
        base_color: string;
        opacity: number;
        metalness: number;
        roughness: number;
        clearcoat: number;
    };
    edges: {
        base_width: number;
        color: string;
        opacity: number;
        enable_arrows: boolean;
        arrow_size: number;
    };
    bloom: {
        enabled: boolean;
        strength: number;
        radius: number;
    };
    animations: {
        enable_node_animations: boolean;
        enable_motion_blur: boolean;
        motion_blur_strength: number;
    };
    labels: {
        enable_labels: boolean;
        desktop_font_size: number;
        text_color: string;
    };
    ar: {
        enable_plane_detection: boolean;
        enable_scene_understanding: boolean;
        show_plane_overlay: boolean;
        plane_opacity: number;
        plane_color: string;
        enable_light_estimation: boolean;
        enable_hand_tracking: boolean;
        hand_mesh_enabled: boolean;
        hand_mesh_color: string;
        hand_mesh_opacity: number;
        hand_ray_enabled: boolean;
        hand_ray_color: string;
        hand_ray_width: number;
        hand_point_size: number;
        gesture_smoothing: number;
        pinch_threshold: number;
        drag_threshold: number;
        rotation_threshold: number;
        enable_haptics: boolean;
        haptic_intensity: number;
        room_scale: boolean;
        snap_to_floor: boolean;
        passthrough_opacity: number;
        passthrough_brightness: number;
        passthrough_contrast: number;
        enable_passthrough_portal: boolean;
        portal_size: number;
        portal_edge_color: string;
        portal_edge_width: number;
    };
}

// Helper functions
export function transformNodeData(raw: RawNodeData): NodeData {
  return {
    position: {
      x: raw.position[0],
      y: raw.position[1],
      z: raw.position[2]
    },
    velocity: {
      x: raw.velocity[0],
      y: raw.velocity[1],
      z: raw.velocity[2]
    },
    mass: raw.mass,
    flags: raw.flags
  };
}

export function transformGraphData(raw: RawGraphData): GraphData {
  return {
    nodes: raw.nodes.map(node => ({
      ...node,
      data: transformNodeData(node.data)
    })),
    edges: raw.edges,
    metadata: raw.metadata
  };
}

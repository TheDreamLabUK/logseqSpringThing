/**
 * Application constants
 */

// Environment detection
export const IS_PRODUCTION = window.location.hostname === 'www.visionflow.info';
export const IS_DEVELOPMENT = !IS_PRODUCTION;

// WebSocket URLs
export const WS_URL = IS_PRODUCTION
  ? 'wss://www.visionflow.info/wss'
  : 'ws://localhost:4000/ws';

// WebSocket configuration
export const WS_RECONNECT_INTERVAL = 5000;
export const WS_MESSAGE_QUEUE_SIZE = 100;

// Performance configuration
export const THROTTLE_INTERVAL = 16; // ~60fps

// Visualization constants
export const NODE_SIZE = 2.5;
export const NODE_SEGMENTS = 16;
export const EDGE_RADIUS = 0.25;
export const EDGE_SEGMENTS = 8;

// Font configuration
export const FONT_URL = '/fonts/Roboto-Regular.woff2';

// Colors
export const NODE_COLOR = 0x4CAF50;  // Material Design Green
export const NODE_HIGHLIGHT_COLOR = 0xff4444;  // Material Design Red
export const EDGE_COLOR = 0xE0E0E0;  // Material Design Grey 300
export const BACKGROUND_COLOR = 0x212121;  // Material Design Grey 900
export const LABEL_COLOR = 0xFFFFFF;  // White

// Default settings
export const DEFAULT_VISUALIZATION_SETTINGS = {
  // Node appearance
  nodeSize: NODE_SIZE,
  nodeColor: '#4CAF50',
  nodeOpacity: 0.7,
  nodeHighlightColor: '#ff4444',
  
  // Edge appearance
  edgeWidth: EDGE_RADIUS * 2,
  edgeColor: '#E0E0E0',
  edgeOpacity: 0.7,
  
  // Visual effects
  enableBloom: true,
  bloomIntensity: 1.5,
  bloomThreshold: 0.3,
  bloomRadius: 0.75,
  
  // Performance
  maxFps: 60,
  updateThrottle: THROTTLE_INTERVAL,

  // Labels
  showLabels: true,
  labelSize: 1.0,
  labelColor: '#FFFFFF',

  // XR specific
  xrControllerVibration: true,
  xrControllerHapticIntensity: 1.0
};

// Default bloom settings
export const DEFAULT_BLOOM_SETTINGS = {
  threshold: DEFAULT_VISUALIZATION_SETTINGS.bloomThreshold,
  strength: DEFAULT_VISUALIZATION_SETTINGS.bloomIntensity,
  radius: DEFAULT_VISUALIZATION_SETTINGS.bloomRadius,
};

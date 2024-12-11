/**
 * Application constants
 */

// Environment detection
export const IS_PRODUCTION = window.location.hostname === 'www.visionflow.info';
export const IS_DEVELOPMENT = !IS_PRODUCTION;

// Base URLs
export const BASE_URL = IS_PRODUCTION 
  ? 'https://www.visionflow.info'
  : 'http://localhost:4000';

// WebSocket URLs - production uses /wss path as configured in nginx
export const WS_URL = IS_PRODUCTION
  ? 'wss://www.visionflow.info/wss'
  : 'ws://localhost:4000/ws';

// Settings endpoint - no /api prefix needed
export const SETTINGS_URL = `${BASE_URL}/settings`;

// WebSocket configuration
export const WS_RECONNECT_INTERVAL = 5000;
export const WS_MESSAGE_QUEUE_SIZE = 100;

// Font configuration - served from public directory
export const FONT_URL = '/fonts/Roboto-Regular.woff2';

// Camera configuration - Adjusted to better view the actual node positions
export const CAMERA_FOV = 75;
export const CAMERA_NEAR = 0.1;
export const CAMERA_FAR = 1000;
// Position camera to better view the nodes based on actual data
// Centered between the two nodes: (-8.27, 8.11, 2.82)
export const CAMERA_POSITION = { x: -8, y: 20, z: 35 };

// Performance configuration
export const THROTTLE_INTERVAL = 16; // ~60fps

// Visualization constants - Increased node size for better visibility
export const NODE_SIZE = 2.0;
export const EDGE_WIDTH = 0.2;
export const LABEL_SIZE = 1.0;

// Position scale factors
export const POSITION_SCALE = 1.0;
export const VELOCITY_SCALE = 0.1;

// Node size scaling - Increased for better visibility
export const MIN_NODE_SIZE = 1.0;
export const MAX_NODE_SIZE = 3.0;
export const SERVER_MIN_SIZE = 20;  // From server data
export const SERVER_MAX_SIZE = 30;  // From server data

// Default settings
export const DEFAULT_VISUALIZATION_SETTINGS = {
  // Node appearance
  nodeSize: NODE_SIZE,
  nodeColor: '#FFA500', // Orange from server settings
  nodeOpacity: 0.95,
  nodeHighlightColor: '#ff4444',
  
  // Node material properties
  nodeMaterialMetalness: 0.7,
  nodeMaterialRoughness: 0.2,
  nodeMaterialEmissiveIntensity: 0.4,
  nodeMaterialClearcoat: 1.0,
  nodeMaterialClearcoatRoughness: 0.1,
  nodeMaterialReflectivity: 1.0,
  nodeMaterialEnvMapIntensity: 1.5,
  
  // Edge appearance
  edgeWidth: EDGE_WIDTH,
  edgeColor: '#FFD700', // Gold from server settings
  edgeOpacity: 0.4,
  
  // Physics settings
  gravity: 0.1,
  springLength: 100,
  springStiffness: 0.08,
  charge: -30,
  damping: 0.9,
  
  // Visual effects
  enableBloom: true,
  bloomIntensity: 1.5,
  bloomThreshold: 0.2,
  bloomRadius: 0.4,
  
  // Performance
  maxFps: 60,
  updateThrottle: THROTTLE_INTERVAL,
  
  // Labels
  showLabels: true,
  labelSize: LABEL_SIZE,
  labelColor: '#ffffff',
  
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

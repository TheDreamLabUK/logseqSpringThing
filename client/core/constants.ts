/**
 * Application constants
 */

// Environment detection
export const IS_PRODUCTION = ['www.visionflow.info', 'visionflow.info'].includes(window.location.hostname);
export const IS_DEVELOPMENT = !IS_PRODUCTION;

// API configuration
export const API_BASE = '';  // Empty string means use relative URLs

// API paths
export const API_PATHS = {
    SETTINGS: 'settings',
    WEBSOCKET: 'wss',
    GRAPH: 'graph',
    FILES: 'files'
};

// API endpoints
export const API_ENDPOINTS = {
    GRAPH_DATA: '/api/graph/data',
    GRAPH_UPDATE: '/api/graph/update',
    GRAPH_PAGINATED: '/api/graph/data/paginated',
    SETTINGS: '/api/settings',
    SETTINGS_UPDATE: '/api/settings/update',
    SETTINGS_CATEGORY: (category: string) => `/api/settings/${category}`,
    SETTINGS_ITEM: (category: string, setting: string) => `/api/settings/${category}/${setting}`,
    VISUALIZATION_SETTINGS: '/api/settings/visualization',
    WEBSOCKET_CONTROL: '/api/settings/websocket',
    FILES: '/api/files'
} as const;

export type ApiEndpoints = typeof API_ENDPOINTS[keyof typeof API_ENDPOINTS];

// Settings categories matching server's snake_case
export const SETTINGS_CATEGORIES = {
    // System settings
    NETWORK: 'system.network',
    WEBSOCKET: 'system.websocket',
    SECURITY: 'system.security',
    DEBUG: 'system.debug',
    
    // Visualization settings
    ANIMATIONS: 'visualization.animations',
    AR: 'visualization.ar',
    AUDIO: 'visualization.audio',
    BLOOM: 'visualization.bloom',
    EDGES: 'visualization.edges',
    HOLOGRAM: 'visualization.hologram',
    LABELS: 'visualization.labels',
    NODES: 'visualization.nodes',
    PHYSICS: 'visualization.physics',
    RENDERING: 'visualization.rendering',
    
    // Default settings
    DEFAULT: 'default'
};

// WebSocket configuration
export const WS_MESSAGE_QUEUE_SIZE = 1000;

// Binary protocol configuration
export const FLOATS_PER_NODE = 6;  // x, y, z, vx, vy, vz
export const VERSION_OFFSET = 0;    // No version header
export const BINARY_CHUNK_SIZE = 1000; // Number of nodes to process in one chunk
export const NODE_POSITION_SIZE = 24;  // 6 floats * 4 bytes (position + velocity)

// Performance configuration
export const THROTTLE_INTERVAL = 16; // ~60fps
export const EDGE_UPDATE_BATCH_INTERVAL = 16; // Batch edge updates at ~60fps

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

// Debug configuration
export const DEBUG = {
    NETWORK_PANEL: {
        MAX_MESSAGES: 50,
        ENABLED: IS_DEVELOPMENT
    }
};

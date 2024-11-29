// Binary protocol scale factors for network transmission
export const POSITION_SCALE = 10000; // Increased from 1000 to handle larger boundaries (up to ±600 units)
export const VELOCITY_SCALE = 20000; // Increased from 10000 to handle higher velocities (up to 20 units)

// WebSocket configuration defaults
export const DEFAULT_RECONNECT_ATTEMPTS = 3;
export const DEFAULT_RECONNECT_DELAY = 5000;
export const DEFAULT_MESSAGE_RATE_LIMIT = 60;
export const DEFAULT_MESSAGE_TIME_WINDOW = 1000;
export const DEFAULT_MAX_MESSAGE_SIZE = 5 * 1024 * 1024; // 5MB
export const DEFAULT_MAX_AUDIO_SIZE = 10 * 1024 * 1024; // 10MB
export const DEFAULT_MAX_QUEUE_SIZE = 1000;

// Connection timeouts (matching server)
export const CONNECTION_TIMEOUT = 10000;      // 10 seconds to establish connection
export const HEARTBEAT_INTERVAL = 30000;      // 30 seconds between pings
export const HEARTBEAT_TIMEOUT = 5000;        // 5 seconds to receive pong

// Binary protocol constants
export const BINARY_UPDATE_HEADER_SIZE = 4;   // Float32 for initial layout flag
export const BINARY_UPDATE_NODE_SIZE = 24;    // 6 float32s per node (position + velocity)
export const FLOAT32_SIZE = 4;                // Size of Float32 in bytes

// Validation constants
export const MAX_VALID_POSITION = 1000;       // Maximum valid position value
export const MAX_VALID_VELOCITY = 50;         // Maximum valid velocity value
export const MIN_VALID_POSITION = -1000;      // Minimum valid position value
export const MIN_VALID_VELOCITY = -50;        // Minimum valid velocity value

// Performance thresholds
export const MAX_PERFORMANCE_SAMPLES = 100;    // Number of samples to keep for performance metrics
export const PERFORMANCE_RESET_INTERVAL = 60000; // Reset performance metrics every minute
export const MAX_MESSAGE_PROCESSING_TIME = 100;  // Maximum acceptable message processing time (ms)
export const MAX_POSITION_UPDATE_TIME = 16;      // Maximum acceptable position update time (ms)

// Debug flags
export const ENABLE_BINARY_DEBUG = true;         // Enable detailed binary update logging
export const ENABLE_POSITION_VALIDATION = true;  // Enable position/velocity validation
export const ENABLE_PERFORMANCE_LOGGING = true;  // Enable performance metric logging

// Message type constants (matching server)
export const MESSAGE_TYPES = {
    GRAPH_UPDATE: 'graphUpdate',
    GRAPH_DATA: 'graphData',
    ERROR: 'error',
    POSITION_UPDATE_COMPLETE: 'position_update_complete',
    SETTINGS_UPDATED: 'settings_updated',
    SIMULATION_MODE_SET: 'simulation_mode_set',
    FISHEYE_SETTINGS_UPDATED: 'fisheye_settings_updated',
    INITIAL_DATA: 'initial_data',
    GPU_STATE: 'gpu_state',
    LAYOUT_STATE: 'layout_state'
} as const;

// Error codes (matching server)
export const ERROR_CODES = {
    CONNECTION_FAILED: 'CONNECTION_FAILED',
    MESSAGE_TOO_LARGE: 'MESSAGE_TOO_LARGE',
    INVALID_MESSAGE: 'INVALID_MESSAGE',
    INVALID_POSITION: 'INVALID_POSITION',
    INVALID_VELOCITY: 'INVALID_VELOCITY',
    MAX_RETRIES_EXCEEDED: 'MAX_RETRIES_EXCEEDED',
    HEARTBEAT_TIMEOUT: 'HEARTBEAT_TIMEOUT'
} as const;

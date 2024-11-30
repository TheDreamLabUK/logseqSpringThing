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
export const HEARTBEAT_INTERVAL = 15000;      // 15 seconds between pings (matching server)
export const HEARTBEAT_TIMEOUT = 60000;       // 60 seconds to receive pong (matching server)

// Binary protocol constants
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

// Server message types (matching server's ServerMessage enum)
export const SERVER_MESSAGE_TYPES = {
    // Direct server message types from ServerMessage enum
    GRAPH_UPDATE: 'graphUpdate',
    ERROR: 'error',
    POSITION_UPDATE_COMPLETE: 'position_update_complete',
    SETTINGS_UPDATED: 'settings_updated',
    SIMULATION_MODE_SET: 'simulation_mode_set',
    FISHEYE_SETTINGS_UPDATED: 'fisheye_settings_updated',
    
    // Additional client-side message types
    INITIAL_DATA: 'initial_data',
    GPU_STATE: 'gpu_state',
    LAYOUT_STATE: 'layout_state',
    OPENAI_RESPONSE: 'openaiResponse',
    RAGFLOW_RESPONSE: 'ragflowResponse',
    COMPLETION: 'completion',
    UPDATE_SETTINGS: 'updateSettings'  // Added for settings updates
} as const;

// Error codes (matching server error structure)
export const ERROR_CODES = {
    // Connection errors
    CONNECTION_FAILED: 'CONNECTION_FAILED',
    MAX_RETRIES_EXCEEDED: 'MAX_RETRIES_EXCEEDED',
    HEARTBEAT_TIMEOUT: 'HEARTBEAT_TIMEOUT',
    
    // Message errors
    MESSAGE_TOO_LARGE: 'MESSAGE_TOO_LARGE',
    INVALID_MESSAGE: 'INVALID_MESSAGE',
    
    // Data validation errors
    INVALID_POSITION: 'INVALID_POSITION',
    INVALID_VELOCITY: 'INVALID_VELOCITY',
    
    // Graph errors
    INVALID_NODE: 'INVALID_NODE',
    INVALID_EDGE: 'INVALID_EDGE',
    
    // State errors
    INVALID_STATE: 'INVALID_STATE',
    SIMULATION_ERROR: 'SIMULATION_ERROR'
} as const;

// Message field names (matching server struct fields)
export const MESSAGE_FIELDS = {
    // GraphUpdate fields
    GRAPH_DATA: 'graph_data',
    
    // Error fields
    MESSAGE: 'message',
    CODE: 'code',
    DETAILS: 'details',
    
    // PositionUpdateComplete fields
    STATUS: 'status',
    
    // SimulationModeSet fields
    MODE: 'mode',
    GPU_ENABLED: 'gpu_enabled',
    
    // FisheyeSettingsUpdated fields
    ENABLED: 'enabled',
    STRENGTH: 'strength',
    FOCUS_POINT: 'focus_point',
    RADIUS: 'radius',

    // Settings fields
    SETTINGS: 'settings',
    MATERIAL: 'material',
    BLOOM: 'bloom',
    FISHEYE: 'fisheye'
} as const;

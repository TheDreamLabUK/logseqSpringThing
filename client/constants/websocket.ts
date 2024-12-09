// Binary protocol scale factors for network transmission
export const POSITION_SCALE = 10000; // Increased from 1000 to handle larger boundaries (up to ±600 units)
export const VELOCITY_SCALE = 20000; // Increased from 10000 to handle higher velocities (up to 20 units)

// WebSocket configuration defaults
export const DEFAULT_RECONNECT_ATTEMPTS = 3;
export const DEFAULT_RECONNECT_DELAY = 5000;
export const DEFAULT_MESSAGE_RATE_LIMIT = 5; // Reduced to 5 messages per second
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
export const BINARY_HEADER_SIZE = 4;          // Size of binary message header in bytes
export const MAX_BINARY_MESSAGE_SIZE = 100 * 1024 * 1024; // 100MB maximum binary message size

// Validation constants
export const MAX_VALID_POSITION = 1000;       // Maximum valid position value
export const MAX_VALID_VELOCITY = 50;         // Maximum valid velocity value
export const MIN_VALID_POSITION = -1000;      // Minimum valid position value
export const MIN_VALID_VELOCITY = -50;        // Minimum valid velocity value
export const MAX_NODE_COUNT = 1000000;        // Maximum number of nodes to prevent memory issues
export const MAX_EDGE_COUNT = 5000000;        // Maximum number of edges to prevent memory issues

// Performance thresholds
export const MAX_PERFORMANCE_SAMPLES = 100;    // Number of samples to keep for performance metrics
export const PERFORMANCE_RESET_INTERVAL = 60000; // Reset performance metrics every minute
export const MAX_MESSAGE_PROCESSING_TIME = 100;  // Maximum acceptable message processing time (ms)
export const MAX_POSITION_UPDATE_TIME = 200;     // Increased to 200ms (5 FPS)
export const MEMORY_WARNING_THRESHOLD = 100 * 1024 * 1024; // 100MB memory usage warning threshold
export const MAX_UPDATES_PER_SECOND = 5;      // Reduced to 5 updates per second
export const UPDATE_THROTTLE_MS = 200;        // Set to 200ms for 5 FPS
export const POSITION_CHANGE_THRESHOLD = 0.01; // Minimum position change to trigger update
export const VELOCITY_CHANGE_THRESHOLD = 0.001; // Minimum velocity change to trigger update
export const THRESHOLD_ADJUSTMENT_FACTOR = 1.5; // Factor to adjust thresholds under high load

// Debug flags
export const ENABLE_BINARY_DEBUG = true;         // Enable detailed binary update logging
export const ENABLE_POSITION_VALIDATION = true;  // Enable position/velocity validation
export const ENABLE_PERFORMANCE_LOGGING = true;  // Enable performance metric logging
export const ENABLE_MEMORY_MONITORING = true;    // Enable memory usage monitoring
export const ENABLE_THRESHOLD_ADJUSTMENT = true; // Enable dynamic threshold adjustment

// Server message types (matching server's ServerMessage enum)
export const SERVER_MESSAGE_TYPES = {
    // Direct server message types from ServerMessage enum
    GRAPH_UPDATE: 'graphUpdate',
    ERROR: 'error',
    POSITION_UPDATE_COMPLETE: 'positionUpdateComplete',
    SETTINGS_UPDATED: 'settingsUpdated',
    SIMULATION_MODE_SET: 'simulationModeSet',
    FISHEYE_SETTINGS_UPDATED: 'fisheyeSettingsUpdated',
    BINARY_POSITION_UPDATE: 'binaryPositionUpdate',
    
    // Additional client-side message types
    INITIAL_DATA: 'initialData',
    GPU_STATE: 'gpuState',
    LAYOUT_STATE: 'layoutState',
    OPENAI_RESPONSE: 'openaiResponse',
    RAGFLOW_RESPONSE: 'ragflowResponse',
    COMPLETION: 'completion',
    UPDATE_SETTINGS: 'updateSettings'
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
    BINARY_MESSAGE_TOO_LARGE: 'BINARY_MESSAGE_TOO_LARGE',
    INVALID_BINARY_FORMAT: 'INVALID_BINARY_FORMAT',
    BUFFER_ALIGNMENT_ERROR: 'BUFFER_ALIGNMENT_ERROR',
    
    // Data validation errors
    INVALID_POSITION: 'INVALID_POSITION',
    INVALID_VELOCITY: 'INVALID_VELOCITY',
    NODE_COUNT_MISMATCH: 'NODE_COUNT_MISMATCH',
    INVALID_NODE_COUNT: 'INVALID_NODE_COUNT',
    
    // Performance errors
    PROCESSING_TIME_EXCEEDED: 'PROCESSING_TIME_EXCEEDED',
    MEMORY_LIMIT_EXCEEDED: 'MEMORY_LIMIT_EXCEEDED',
    UPDATE_RATE_EXCEEDED: 'UPDATE_RATE_EXCEEDED',
    
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
    GRAPH_DATA: 'graphData',
    
    // Error fields
    MESSAGE: 'message',
    CODE: 'code',
    DETAILS: 'details',
    
    // PositionUpdateComplete fields
    STATUS: 'status',
    
    // SimulationModeSet fields
    MODE: 'mode',
    GPU_ENABLED: 'gpuEnabled',
    
    // FisheyeSettingsUpdated fields
    ENABLED: 'enabled',
    STRENGTH: 'strength',
    FOCUS_POINT: 'focusPoint',
    RADIUS: 'radius',

    // Settings fields
    SETTINGS: 'settings',
    MATERIAL: 'material',
    BLOOM: 'bloom',
    FISHEYE: 'fisheye',
    
    // Performance fields
    PROCESSING_TIME: 'processingTime',
    MEMORY_USAGE: 'memoryUsage',
    UPDATE_FREQUENCY: 'updateFrequency'
} as const;

// Performance metric names
export const PERFORMANCE_METRICS = {
    PROCESSING_TIME: 'processingTime',
    AVERAGE_PROCESSING_TIME: 'averageProcessingTime',
    MEMORY_USAGE: 'memoryUsage',
    UPDATE_FREQUENCY: 'updateFrequency',
    INVALID_UPDATE_RATE: 'invalidUpdateRate',
    CHANGED_NODES: 'changedNodes'
} as const;

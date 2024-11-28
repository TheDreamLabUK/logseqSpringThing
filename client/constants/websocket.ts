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

// Binary protocol constants
export const BINARY_UPDATE_HEADER_SIZE = 4; // Float32 for initial layout flag
export const BINARY_UPDATE_NODE_SIZE = 24;  // 6 float32s per node (position + velocity)

// Validation constants
export const MAX_VALID_POSITION = 1000;  // Maximum valid position value
export const MAX_VALID_VELOCITY = 50;    // Maximum valid velocity value

// Debug flags
export const ENABLE_BINARY_DEBUG = true;  // Enable detailed binary update logging
export const ENABLE_POSITION_VALIDATION = true;  // Enable position/velocity validation

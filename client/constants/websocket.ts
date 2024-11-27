// Binary protocol scale factors for network transmission
export const POSITION_SCALE = 1000; // Quantize positions to millimeters
export const VELOCITY_SCALE = 10000; // Quantize velocities to 0.0001 units

// WebSocket configuration defaults
export const DEFAULT_RECONNECT_ATTEMPTS = 3;
export const DEFAULT_RECONNECT_DELAY = 5000;
export const DEFAULT_MESSAGE_RATE_LIMIT = 60;
export const DEFAULT_MESSAGE_TIME_WINDOW = 1000;
export const DEFAULT_MAX_MESSAGE_SIZE = 5 * 1024 * 1024; // 5MB
export const DEFAULT_MAX_AUDIO_SIZE = 10 * 1024 * 1024; // 10MB
export const DEFAULT_MAX_QUEUE_SIZE = 1000;

export const BINARY_PROTOCOL = {
    VERSION: 1,
    NODE_POSITION_SIZE: 4 * 3, // 3 float32 values (x, y, z)
    MAX_MESSAGE_SIZE: 1024 * 1024, // 1MB
    MAX_CONNECTIONS: 100,
    HEARTBEAT_INTERVAL: 30000, // 30 seconds
    MAX_CLIENT_TIMEOUT: 90000, // 90 seconds
    POSITION_UPDATE_INTERVAL: 16 // ~60fps
} as const;

export const HEADER_SIZE = 8; // Version (4 bytes) + Message Type (4 bytes) 
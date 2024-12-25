// WebSocket protocol version
pub const BINARY_PROTOCOL_VERSION: i32 = 1;

// WebSocket timing constants (in seconds)
pub const HEARTBEAT_INTERVAL: u64 = 30;  // Match Cloudflared interval
pub const MAX_CLIENT_TIMEOUT: u64 = 60;  // Match Cloudflared timeout

// Binary protocol constants
pub const FLOATS_PER_NODE: usize = 6;  // x, y, z, vx, vy, vz
pub const VERSION_HEADER_SIZE: usize = std::mem::size_of::<i32>();
pub const FLOAT_SIZE: usize = std::mem::size_of::<f32>();
pub const NODE_DATA_SIZE: usize = FLOAT_SIZE * FLOATS_PER_NODE;

// Connection limits
pub const MAX_CONNECTIONS: usize = 100;
pub const MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;  // 32MB

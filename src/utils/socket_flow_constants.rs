// WebSocket protocol version
pub const BINARY_PROTOCOL_VERSION: i32 = 1;

// WebSocket constants - matching nginx configuration
pub const HEARTBEAT_INTERVAL: u64 = 30; // seconds - matches nginx proxy_connect_timeout
pub const CLIENT_TIMEOUT: u64 = 60; // seconds - double heartbeat interval for safety
pub const MAX_CLIENT_TIMEOUT: u64 = 3600; // seconds - matches nginx proxy_read_timeout
pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB
pub const BINARY_CHUNK_SIZE: usize = 64 * 1024; // 64KB

// Update rate constants
pub const POSITION_UPDATE_RATE: u32 = 5; // Hz (matching client's MAX_UPDATES_PER_SECOND)
pub const METADATA_UPDATE_RATE: u32 = 1; // Hz

// Binary message constants
pub const NODE_POSITION_SIZE: usize = 24; // 6 f32s per node (position + velocity)

// Connection limits
pub const MAX_CONNECTIONS: usize = 100;
pub const MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;  // 32MB

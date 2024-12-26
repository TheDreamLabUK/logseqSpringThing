// Node and graph constants
pub const NODE_SIZE: f32 = 1.0;
pub const EDGE_WIDTH: f32 = 0.1;
pub const MIN_DISTANCE: f32 = 2.0;
pub const MAX_DISTANCE: f32 = 10.0;

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

// Compression constants
pub const COMPRESSION_THRESHOLD: usize = 1024; // 1KB
pub const ENABLE_COMPRESSION: bool = true;

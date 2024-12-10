// Node and graph constants
pub const NODE_SIZE: f32 = 1.0;
pub const EDGE_WIDTH: f32 = 0.1;
pub const MIN_DISTANCE: f32 = 2.0;
pub const MAX_DISTANCE: f32 = 10.0;

// WebSocket constants
pub const HEARTBEAT_INTERVAL: u64 = 5; // seconds
pub const CLIENT_TIMEOUT: u64 = 10; // seconds
pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB
pub const BINARY_CHUNK_SIZE: usize = 64 * 1024; // 64KB

// Update rate constants
pub const POSITION_UPDATE_RATE: u32 = 30; // Hz
pub const METADATA_UPDATE_RATE: u32 = 1; // Hz

// Compression constants
pub const COMPRESSION_THRESHOLD: usize = 1024; // 1KB
pub const ENABLE_COMPRESSION: bool = true;

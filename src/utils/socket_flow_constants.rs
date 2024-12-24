// Node and graph constants
pub const NODE_SIZE: f32 = 1.0;
pub const EDGE_WIDTH: f32 = 0.1;
pub const MIN_DISTANCE: f32 = 2.0;
pub const MAX_DISTANCE: f32 = 10.0;

// WebSocket constants
// Note: Connection keep-alive is handled by WebSocket protocol-level ping/pong frames
// automatically by the actix-web-actors framework on the server and browser WebSocket API
// on the client. No custom heartbeat implementation is needed.
pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB

// Update rate constants - can be changed via control panel
pub const POSITION_UPDATE_RATE: u32 = 60; // Hz (default 60fps)
pub const METADATA_UPDATE_RATE: u32 = 1; // Hz

// Binary message constants
pub const NODE_POSITION_SIZE: usize = 24; // 6 f32s (x,y,z,vx,vy,vz) * 4 bytes

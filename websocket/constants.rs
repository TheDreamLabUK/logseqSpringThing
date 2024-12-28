pub const BINARY_PROTOCOL_VERSION: u32 = 1;
pub const NODE_POSITION_SIZE: usize = 12; // 3 f32s (x, y, z) * 4 bytes
pub const MAX_MESSAGE_SIZE: usize = 1024 * 1024; // 1MB
pub const MAX_CONNECTIONS: usize = 100;
pub const HEARTBEAT_INTERVAL: u64 = 30000; // 30 seconds
pub const MAX_CLIENT_TIMEOUT: u64 = 90000; // 90 seconds
pub const POSITION_UPDATE_INTERVAL: u64 = 16; // ~60fps

#[derive(Debug, Clone, Copy)]
pub enum MessageType {
    PositionUpdate = 1,
    ForceUpdate = 2,
    StateUpdate = 3,
    GpuComputeStatus = 4,
    CompressedData = 5,
}

impl From<u32> for MessageType {
    fn from(value: u32) -> Self {
        match value {
            1 => MessageType::PositionUpdate,
            2 => MessageType::ForceUpdate,
            3 => MessageType::StateUpdate,
            4 => MessageType::GpuComputeStatus,
            5 => MessageType::CompressedData,
            _ => panic!("Invalid message type: {}", value),
        }
    }
} 
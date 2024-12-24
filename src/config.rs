// Previous content remains the same until WebSocketSettings...

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct WebSocketSettings {
    pub binary_chunk_size: usize,
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    pub max_connections: usize,
    pub max_message_size: usize,
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub update_rate: u32,
}

impl Default for WebSocketSettings {
    fn default() -> Self {
        Self {
            binary_chunk_size: 65536,
            compression_enabled: true,
            compression_threshold: 1024,
            max_connections: 1000,
            max_message_size: 100485760,
            reconnect_attempts: 3,
            reconnect_delay: 5000,
            update_rate: 90,
        }
    }
}

// Rest of the file remains the same...

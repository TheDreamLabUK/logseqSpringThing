use log::trace;
use serde_json::Value;

// Enhanced debug logging macro for websocket data
macro_rules! ws_debug {
    ($debug_mode:expr, $message:expr, $data:expr) => {
        if $debug_mode {
            let timestamp = chrono::Utc::now().to_rfc3339();
            trace!("[WebSocket Debug {}] {}", timestamp, $message);

            // Log detailed data if available
            if let Some(data) = $data {
                match data {
                    // For binary data
                    WsDebugData::Binary { data, is_initial, node_count } => {
                        trace!("[WebSocket Binary Data] Header:");
                        trace!("  Is Initial: {}", is_initial);
                        trace!("  Node Count: {}", node_count);
                        trace!("  Total Size: {} bytes", data.len());
                        
                        // Show hex dump of first 32 bytes
                        if !data.is_empty() {
                            let hex_dump: String = data.iter()
                                .take(32)
                                .map(|b| format!("{:02x}", b))
                                .collect::<Vec<_>>()
                                .join(" ");
                            trace!("  First 32 bytes: {}", hex_dump);
                        }

                        // Show sample node data
                        if node_count > 0 {
                            for i in 0..std::cmp::min(3, node_count) {
                                let offset = 4 + i * 24; // Skip header
                                if offset + 24 <= data.len() {
                                    let mut pos = [0.0f32; 3];
                                    let mut vel = [0.0f32; 3];
                                    for j in 0..3 {
                                        pos[j] = f32::from_le_bytes([
                                            data[offset + j*4],
                                            data[offset + j*4 + 1],
                                            data[offset + j*4 + 2],
                                            data[offset + j*4 + 3]
                                        ]);
                                        vel[j] = f32::from_le_bytes([
                                            data[offset + (j+3)*4],
                                            data[offset + (j+3)*4 + 1],
                                            data[offset + (j+3)*4 + 2],
                                            data[offset + (j+3)*4 + 3]
                                        ]);
                                    }
                                    trace!("  Node {}: pos=({:.3},{:.3},{:.3}), vel=({:.3},{:.3},{:.3})",
                                        i, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
                                }
                            }
                        }
                    },
                    // For JSON data
                    WsDebugData::Json(json_data) => {
                        trace!("[WebSocket JSON Data]:");
                        trace!("  Type: {}", json_data["type"].as_str().unwrap_or("unknown"));
                        trace!("  Size: {} bytes", json_data.to_string().len());
                        trace!("  Content: {}", serde_json::to_string_pretty(&json_data).unwrap_or_default());
                    },
                    // For raw text data
                    WsDebugData::Text(text) => {
                        trace!("[WebSocket Text Data]: {}", text);
                    }
                }
            }
        }
    };
}

// Data types for debug logging
pub enum WsDebugData<'a> {
    Binary {
        data: &'a [u8],
        is_initial: bool,
        node_count: usize,
    },
    Json(Value),
    Text(String),
}

pub(crate) use ws_debug;

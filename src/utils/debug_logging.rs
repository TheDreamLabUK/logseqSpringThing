use serde_json::Value;

// Enhanced debug logging macro for websocket data

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


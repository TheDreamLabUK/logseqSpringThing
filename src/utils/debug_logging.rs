use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use log::debug;

// Global debug state
pub static DEBUG_MODE: AtomicBool = AtomicBool::new(false);
pub static WEBSOCKET_DEBUG: AtomicBool = AtomicBool::new(false);
pub static DATA_DEBUG: AtomicBool = AtomicBool::new(false);

// Initialize debug settings
pub fn init_debug_settings(debug_mode: bool, websocket_debug: bool, data_debug: bool) {
    DEBUG_MODE.store(debug_mode, Ordering::SeqCst);
    WEBSOCKET_DEBUG.store(websocket_debug, Ordering::SeqCst);
    DATA_DEBUG.store(data_debug, Ordering::SeqCst);
}

// Data types for debug logging
#[derive(Debug)]
pub enum WsDebugData<'a> {
    Binary {
        data: &'a [u8],
        is_initial: bool,
        node_count: usize,
    },
    Json(Value),
    Text(String),
}

impl<'a> std::fmt::Display for WsDebugData<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WsDebugData::Binary { data, is_initial, node_count } => {
                write!(f, "Binary message: {} bytes, initial: {}, nodes: {}", 
                    data.len(), is_initial, node_count)
            },
            WsDebugData::Json(value) => {
                write!(f, "JSON message: {}", value)
            },
            WsDebugData::Text(text) => {
                write!(f, "Text message: {}", text)
            }
        }
    }
}

// Logging macros with different levels
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        use log::error;
        error!($($arg)*);
    }}
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{
        use log::warn;
        if $crate::utils::debug_logging::DEBUG_MODE.load(std::sync::atomic::Ordering::SeqCst) {
            warn!($($arg)*);
        }
    }}
}

#[macro_export]
macro_rules! log_websocket {
    ($($arg:tt)*) => {{
        use log::debug;
        if $crate::utils::debug_logging::WEBSOCKET_DEBUG.load(std::sync::atomic::Ordering::SeqCst) {
            debug!("[WS] {}", format!($($arg)*));
        }
    }}
}

#[macro_export]
macro_rules! log_data {
    ($($arg:tt)*) => {{
        use log::debug;
        if $crate::utils::debug_logging::DATA_DEBUG.load(std::sync::atomic::Ordering::SeqCst) {
            debug!("[DATA] {}", format!($($arg)*));
        }
    }}
}

// Helper functions for common debug scenarios
pub fn log_ws_message(data: WsDebugData) {
    if !WEBSOCKET_DEBUG.load(Ordering::SeqCst) {
        return;
    }

    match data {
        WsDebugData::Binary { data, is_initial, node_count } => {
            debug!(
                "WebSocket Binary Message:\n  Size: {} bytes\n  Initial: {}\n  Node Count: {}\n  Header: {:?}",
                data.len(),
                is_initial,
                node_count,
                &data[..std::cmp::min(data.len(), 32)]
            );
        },
        WsDebugData::Json(value) => {
            if let Ok(pretty) = serde_json::to_string_pretty(&value) {
                debug!("WebSocket JSON Message:\n{}", pretty);
            } else {
                debug!("WebSocket JSON Message: {}", value);
            }
        },
        WsDebugData::Text(text) => {
            debug!("WebSocket Text Message: {}", text);
        }
    }
}

pub fn log_data_operation(operation: &str, details: &str) {
    if !DATA_DEBUG.load(Ordering::SeqCst) {
        return;
    }
    debug!("Data Operation - {}: {}", operation, details);
}

pub fn log_binary_headers(data: &[u8], context: &str) {
    if !DEBUG_MODE.load(Ordering::SeqCst) {
        return;
    }
    debug!(
        "Binary Headers [{}]:\n  Size: {} bytes\n  Header: {:?}",
        context,
        data.len(),
        &data[..std::cmp::min(data.len(), 32)]
    );
}

pub fn log_json_data(context: &str, value: &Value) {
    if !DEBUG_MODE.load(Ordering::SeqCst) {
        return;
    }
    if let Ok(pretty) = serde_json::to_string_pretty(value) {
        debug!("JSON Data [{}]:\n{}", context, pretty);
    } else {
        debug!("JSON Data [{}]: {}", context, value);
    }
}

// Test that debug settings are working
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_settings() {
        init_debug_settings(true, true, false);
        assert!(DEBUG_MODE.load(Ordering::SeqCst));
        assert!(WEBSOCKET_DEBUG.load(Ordering::SeqCst));
        assert!(!DATA_DEBUG.load(Ordering::SeqCst));
    }
}

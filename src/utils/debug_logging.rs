use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use log::{error, warn, debug};

// Global debug state
static DEBUG_MODE: AtomicBool = AtomicBool::new(false);
static WEBSOCKET_DEBUG: AtomicBool = AtomicBool::new(false);
static DATA_DEBUG: AtomicBool = AtomicBool::new(false);

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

// Logging macros with different levels
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        error!($($arg)*);
    }
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        if DEBUG_MODE.load(Ordering::SeqCst) {
            warn!($($arg)*);
        }
    }
}

#[macro_export]
macro_rules! log_websocket {
    ($($arg:tt)*) => {
        if WEBSOCKET_DEBUG.load(Ordering::SeqCst) {
            debug!("[WS] {}", format!($($arg)*));
        }
    }
}

#[macro_export]
macro_rules! log_data {
    ($($arg:tt)*) => {
        if DATA_DEBUG.load(Ordering::SeqCst) {
            debug!("[DATA] {}", format!($($arg)*));
        }
    }
}

// Helper functions for common debug scenarios
pub fn log_ws_message(data: WsDebugData) {
    if WEBSOCKET_DEBUG.load(Ordering::SeqCst) {
        match data {
            WsDebugData::Binary { data, is_initial, node_count } => {
                debug!("[WS] Binary message: {} bytes, initial: {}, nodes: {}", 
                    data.len(), is_initial, node_count);
            },
            WsDebugData::Json(value) => {
                debug!("[WS] JSON message: {}", value);
            },
            WsDebugData::Text(text) => {
                debug!("[WS] Text message: {}", text);
            }
        }
    }
}

pub fn log_data_operation(operation: &str, details: &str) {
    if DATA_DEBUG.load(Ordering::SeqCst) {
        debug!("[DATA] {} - {}", operation, details);
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

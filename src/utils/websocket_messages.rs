use actix::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Constants for binary protocol
pub const POSITION_SCALE: f32 = 10000.0;  // Increased to match client
pub const VELOCITY_SCALE: f32 = 20000.0;  // Increased to match client
pub const BINARY_HEADER_SIZE: usize = 4;  // Float32 for initial layout flag
pub const BINARY_NODE_SIZE: usize = 24;   // 6 float32s per node (position + velocity)

// Maximum valid values for validation
pub const MAX_VALID_POSITION: f32 = 1000.0;
pub const MAX_VALID_VELOCITY: f32 = 50.0;

/// Trait for handling WebSocket messages
pub trait MessageHandler {}

/// Message types for WebSocket communication
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendText(pub String);

#[derive(Message)]
#[rtype(result = "()")]
pub struct SendBinary(pub Vec<u8>);

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIMessage(pub String);

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnected;

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnectionFailed;

/// Helper function to validate position values
pub fn validate_position(x: f32, y: f32, z: f32) -> bool {
    x.abs() <= MAX_VALID_POSITION && 
    y.abs() <= MAX_VALID_POSITION && 
    z.abs() <= MAX_VALID_POSITION
}

/// Helper function to validate velocity values
pub fn validate_velocity(vx: f32, vy: f32, vz: f32) -> bool {
    vx.abs() <= MAX_VALID_VELOCITY && 
    vy.abs() <= MAX_VALID_VELOCITY && 
    vz.abs() <= MAX_VALID_VELOCITY
}

/// Helper function to quantize position for network transmission
pub fn quantize_position(pos: f32) -> i32 {
    (pos * POSITION_SCALE).round() as i32
}

/// Helper function to quantize velocity for network transmission
pub fn quantize_velocity(vel: f32) -> i32 {
    (vel * VELOCITY_SCALE).round() as i32
}

/// Helper function to dequantize position from network transmission
pub fn dequantize_position(pos: i32) -> f32 {
    pos as f32 / POSITION_SCALE
}

/// Helper function to dequantize velocity from network transmission
pub fn dequantize_velocity(vel: i32) -> f32 {
    vel as f32 / VELOCITY_SCALE
}

/// Helper function to create binary position update message
pub fn create_binary_update(positions: &[(f32, f32, f32, f32, f32, f32)], is_initial: bool) -> Vec<u8> {
    let mut binary_data = Vec::with_capacity(BINARY_HEADER_SIZE + positions.len() * BINARY_NODE_SIZE);
    
    // Add initial layout flag
    let flag = if is_initial { 1.0f32 } else { 0.0f32 };
    binary_data.extend_from_slice(&flag.to_le_bytes());
    
    // Add quantized positions and velocities
    for &(x, y, z, vx, vy, vz) in positions {
        // Validate values before quantizing
        if !validate_position(x, y, z) || !validate_velocity(vx, vy, vz) {
            continue;
        }
        
        // Quantize and add position
        binary_data.extend_from_slice(&quantize_position(x).to_le_bytes());
        binary_data.extend_from_slice(&quantize_position(y).to_le_bytes());
        binary_data.extend_from_slice(&quantize_position(z).to_le_bytes());
        
        // Quantize and add velocity
        binary_data.extend_from_slice(&quantize_velocity(vx).to_le_bytes());
        binary_data.extend_from_slice(&quantize_velocity(vy).to_le_bytes());
        binary_data.extend_from_slice(&quantize_velocity(vz).to_le_bytes());
    }
    
    binary_data
}

/// Helper function to parse binary position update message
pub fn parse_binary_update(data: &[u8]) -> Option<(bool, Vec<(f32, f32, f32, f32, f32, f32)>)> {
    if data.len() < BINARY_HEADER_SIZE {
        return None;
    }
    
    // Read initial layout flag
    let mut flag_bytes = [0u8; 4];
    flag_bytes.copy_from_slice(&data[0..4]);
    let is_initial = f32::from_le_bytes(flag_bytes) >= 1.0;
    
    // Calculate number of positions
    let num_positions = (data.len() - BINARY_HEADER_SIZE) / BINARY_NODE_SIZE;
    let mut positions = Vec::with_capacity(num_positions);
    
    // Parse positions and velocities
    let mut offset = BINARY_HEADER_SIZE;
    while offset + BINARY_NODE_SIZE <= data.len() {
        let mut buf = [0i32; 6];
        for i in 0..6 {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&data[offset + i * 4..offset + (i + 1) * 4]);
            buf[i] = i32::from_le_bytes(bytes);
        }
        
        let x = dequantize_position(buf[0]);
        let y = dequantize_position(buf[1]);
        let z = dequantize_position(buf[2]);
        let vx = dequantize_velocity(buf[3]);
        let vy = dequantize_velocity(buf[4]);
        let vz = dequantize_velocity(buf[5]);
        
        // Validate dequantized values
        if validate_position(x, y, z) && validate_velocity(vx, vy, vz) {
            positions.push((x, y, z, vx, vy, vz));
        }
        
        offset += BINARY_NODE_SIZE;
    }
    
    Some((is_initial, positions))
}

/// Server message types for WebSocket communication
#[derive(Serialize, Debug)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "graphUpdate")]
    GraphUpdate {
        graph_data: Value,
    },
    #[serde(rename = "settingsUpdated")]
    SettingsUpdated {
        settings: Value,
    },
    #[serde(rename = "simulationModeSet")]
    SimulationModeSet {
        mode: String,
        gpu_enabled: bool,
    },
    #[serde(rename = "fisheyeSettingsUpdated")]
    FisheyeSettingsUpdated {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        code: Option<String>,
    },
}

/// Client message types for WebSocket communication
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "updateNodePosition")]
    UpdateNodePosition {
        node_id: String,
        position: [f32; 3],
    },
    #[serde(rename = "setSimulationMode")]
    SetSimulationMode {
        mode: String,
    },
    #[serde(rename = "updateFisheyeSettings")]
    UpdateFisheyeSettings {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_quantization() {
        let pos = 123.456;
        let quantized = quantize_position(pos);
        let dequantized = dequantize_position(quantized);
        assert!((pos - dequantized).abs() < 0.001);
    }

    #[test]
    fn test_velocity_quantization() {
        let vel = 1.2345;
        let quantized = quantize_velocity(vel);
        let dequantized = dequantize_velocity(quantized);
        assert!((vel - dequantized).abs() < 0.0001);
    }

    #[test]
    fn test_binary_update_roundtrip() {
        let positions = vec![
            (1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
            (-1.0, -2.0, -3.0, -0.1, -0.2, -0.3),
        ];
        let binary = create_binary_update(&positions, true);
        let parsed = parse_binary_update(&binary).unwrap();
        
        assert!(parsed.0); // is_initial
        assert_eq!(parsed.1.len(), positions.len());
        
        for (orig, parsed) in positions.iter().zip(parsed.1.iter()) {
            assert!((orig.0 - parsed.0).abs() < 0.001);
            assert!((orig.1 - parsed.1).abs() < 0.001);
            assert!((orig.2 - parsed.2).abs() < 0.001);
            assert!((orig.3 - parsed.3).abs() < 0.0001);
            assert!((orig.4 - parsed.4).abs() < 0.0001);
            assert!((orig.5 - parsed.5).abs() < 0.0001);
        }
    }

    #[test]
    fn test_validation() {
        assert!(validate_position(100.0, -100.0, 500.0));
        assert!(!validate_position(2000.0, 0.0, 0.0));
        
        assert!(validate_velocity(10.0, -10.0, 20.0));
        assert!(!validate_velocity(100.0, 0.0, 0.0));
    }
}

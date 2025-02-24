use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
/// Binary node data structure for efficient transmission and GPU processing
/// 
/// Wire format (28 bytes per node):
/// - position: [f32; 3] (12 bytes)
/// - velocity: [f32; 3] (12 bytes)
/// - id: u32 (4 bytes)
///
/// Note: mass, flags, and padding are server-side only and not transmitted over the wire
/// to optimize bandwidth. They are still available for GPU processing and physics calculations.
pub struct BinaryNodeData {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub mass: u8,      // Server-side only, not transmitted
    pub flags: u8,     // Server-side only, not transmitted
    pub padding: [u8; 2], // Server-side only, not transmitted
}

// Implement DeviceRepr for BinaryNodeData
unsafe impl DeviceRepr for BinaryNodeData {}

// Implement ValidAsZeroBits for BinaryNodeData
unsafe impl ValidAsZeroBits for BinaryNodeData {}

#[derive(Debug, Serialize, Deserialize)]
pub struct PingMessage {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(default = "default_timestamp")]
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PongMessage {
    #[serde(rename = "type")]
    pub type_: String,
    pub timestamp: u64,
}

fn default_timestamp() -> u64 {
    chrono::Utc::now().timestamp_millis() as u64
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    // Core data
    pub id: String,
    pub label: String,
    pub data: BinaryNodeData,

    // Metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    #[serde(skip)]
    pub file_size: u64,

    // Rendering properties
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<HashMap<String, String>>,
}

impl Node {
    pub fn new(id: String) -> Self {
        Self {
            id: id.clone(),
            label: id,
            data: BinaryNodeData {
                position: [0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
                mass: 0, // default mass, will be updated based on file size
                flags: 0,
                padding: [0, 0],
            },
            metadata: HashMap::new(),
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }

    pub fn calculate_mass(file_size: u64) -> u8 {
        // Use log scale to prevent extremely large masses
        // Add 1 to file_size to handle empty files (log(0) is undefined)
        // Scale down by 10000 to keep masses in a reasonable range
        let base_mass = ((file_size + 1) as f32).log10() / 4.0;
        // Ensure minimum mass of 0.1 and maximum of 10.0
        let mass = base_mass.max(0.1).min(10.0);
        (mass * 255.0 / 10.0) as u8
    }

    pub fn set_file_size(&mut self, size: u64) {
        self.file_size = size;
        // Update mass based on new file size
        self.data.mass = Self::calculate_mass(size);
    }

    // Convenience getters/setters for x, y, z coordinates
    pub fn x(&self) -> f32 { self.data.position[0] }
    pub fn y(&self) -> f32 { self.data.position[1] }
    pub fn z(&self) -> f32 { self.data.position[2] }
    pub fn vx(&self) -> f32 { self.data.velocity[0] }
    pub fn vy(&self) -> f32 { self.data.velocity[1] }
    pub fn vz(&self) -> f32 { self.data.velocity[2] }
    
    pub fn set_x(&mut self, val: f32) { self.data.position[0] = val; }
    pub fn set_y(&mut self, val: f32) { self.data.position[1] = val; }
    pub fn set_z(&mut self, val: f32) { self.data.position[2] = val; }
    pub fn set_vx(&mut self, val: f32) { self.data.velocity[0] = val; }
    pub fn set_vy(&mut self, val: f32) { self.data.velocity[1] = val; }
    pub fn set_vz(&mut self, val: f32) { self.data.velocity[2] = val; }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    #[serde(rename = "ping")]
    Ping { timestamp: u64 },
    
    #[serde(rename = "pong")]
    Pong { timestamp: u64 },
}

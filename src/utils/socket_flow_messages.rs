use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
pub struct BinaryNodeData {
    pub position: [f32; 3],  // x, y, z
    pub velocity: [f32; 3],  // vx, vy, vz
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
                position: [0.0; 3],
                velocity: [0.0; 3],
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

    pub fn update_from_binary_data(&mut self, binary_data: &BinaryNodeData) {
        self.data = *binary_data;
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

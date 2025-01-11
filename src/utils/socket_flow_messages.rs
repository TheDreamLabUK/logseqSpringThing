use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NodeData {
    pub position: [f32; 3],  // 12 bytes - matches THREE.Vector3
    pub velocity: [f32; 3],  // 12 bytes - matches THREE.Vector3
    pub mass: u8,            // 1 byte - quantized mass
    pub flags: u8,           // 1 byte - node state flags
    pub padding: [u8; 2],    // 2 bytes - alignment padding
}

// Custom serialization for NodeData to convert arrays to Vector3-compatible objects
#[derive(Serialize, Deserialize)]
#[serde(remote = "NodeData")]
struct NodeDataDef {
    #[serde(getter = "get_position_vec3", setter = "set_position_vec3")]
    position: [f32; 3],
    #[serde(getter = "get_velocity_vec3", setter = "set_velocity_vec3")]
    velocity: [f32; 3],
    mass: u8,
    flags: u8,
    padding: [u8; 2],
}

// Helper functions for Vector3 serialization
fn get_position_vec3(data: &NodeData) -> serde_json::Value {
    serde_json::json!({
        "x": data.position[0],
        "y": data.position[1],
        "z": data.position[2]
    })
}

fn set_position_vec3(data: &mut NodeData, value: serde_json::Value) {
    if let Some(obj) = value.as_object() {
        if let (Some(x), Some(y), Some(z)) = (
            obj.get("x").and_then(|v| v.as_f64()),
            obj.get("y").and_then(|v| v.as_f64()),
            obj.get("z").and_then(|v| v.as_f64())
        ) {
            data.position = [x as f32, y as f32, z as f32];
        }
    }
}

fn get_velocity_vec3(data: &NodeData) -> serde_json::Value {
    serde_json::json!({
        "x": data.velocity[0],
        "y": data.velocity[1],
        "z": data.velocity[2]
    })
}

fn set_velocity_vec3(data: &mut NodeData, value: serde_json::Value) {
    if let Some(obj) = value.as_object() {
        if let (Some(x), Some(y), Some(z)) = (
            obj.get("x").and_then(|v| v.as_f64()),
            obj.get("y").and_then(|v| v.as_f64()),
            obj.get("z").and_then(|v| v.as_f64())
        ) {
            data.velocity = [x as f32, y as f32, z as f32];
        }
    }
}

// Implement serialization for NodeData
impl Serialize for NodeData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NodeData", 5)?;
        state.serialize_field("position", &get_position_vec3(self))?;
        state.serialize_field("velocity", &get_velocity_vec3(self))?;
        state.serialize_field("mass", &self.mass)?;
        state.serialize_field("flags", &self.flags)?;
        state.serialize_field("padding", &self.padding)?;
        state.end()
    }
}

// Implement deserialization for NodeData
impl<'de> Deserialize<'de> for NodeData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            position: serde_json::Value,
            velocity: serde_json::Value,
            mass: u8,
            flags: u8,
            padding: [u8; 2],
        }

        let helper = Helper::deserialize(deserializer)?;
        let mut data = NodeData {
            position: [0.0; 3],
            velocity: [0.0; 3],
            mass: helper.mass,
            flags: helper.flags,
            padding: helper.padding,
        };
        set_position_vec3(&mut data, helper.position);
        set_velocity_vec3(&mut data, helper.velocity);
        Ok(data)
    }
}

// Implement DeviceRepr for NodeData
unsafe impl DeviceRepr for NodeData {}

// Implement ValidAsZeroBits for NodeData
unsafe impl ValidAsZeroBits for NodeData {}

#[derive(Debug, Serialize, Deserialize)]
pub struct PingMessage {
    pub type_: String,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PongMessage {
    pub type_: String,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    // Core data
    pub id: String,
    pub label: String,
    pub data: NodeData,

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
            data: NodeData {
                position: [0.0; 3],
                velocity: [0.0; 3],
                mass: 127, // Default mass
                flags: 0,
                padding: [0; 2],
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

    pub fn update_mass(&mut self) {
        if self.file_size == 0 {
            self.data.mass = 127; // Default mass
            return;
        }
        
        // Scale file size logarithmically to 0-255 range
        let log_size = (self.file_size as f64).log2();
        let max_log = (1024.0 * 1024.0 * 1024.0_f64).log2(); // 1GB
        let normalized = (log_size / max_log).min(1.0);
        self.data.mass = (normalized * 255.0) as u8;
    }

    pub fn update_from_gpu_node(&mut self, gpu_node: &NodeData) {
        self.data = *gpu_node;
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

impl BinaryNodeData {
    pub fn from_node_data(data: &NodeData) -> Self {
        Self {
            position: data.position,
            velocity: data.velocity,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    #[serde(rename = "ping")]
    Ping { timestamp: u64 },
    
    #[serde(rename = "pong")]
    Pong { timestamp: u64 },
}

// Forward declarations to avoid circular dependencies

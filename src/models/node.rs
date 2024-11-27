use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};
use crate::models::position_update::NodePositionVelocity;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub metadata: HashMap<String, String>,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    #[serde(skip)]
    pub vx: f32,
    #[serde(skip)]
    pub vy: f32,
    #[serde(skip)]
    pub vz: f32,
    #[serde(skip)]
    pub file_size: u64, // Used to calculate mass
    #[serde(rename = "type")]
    pub node_type: Option<String>,
    pub size: Option<f32>,
    pub color: Option<String>,
    pub weight: Option<f32>,
    pub group: Option<String>,
    pub user_data: Option<HashMap<String, String>>,
}

impl Node {
    pub fn new(id: String) -> Self {
        Self {
            id: id.clone(),
            label: id,
            metadata: HashMap::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }

    pub fn position(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn velocity(&self) -> [f32; 3] {
        [self.vx, self.vy, self.vz]
    }

    /// Convert file size to quantized mass value (0-255)
    fn calculate_mass(&self) -> u8 {
        // Scale file size logarithmically to 0-255 range
        // Assuming file sizes from 0 to ~1GB
        if self.file_size == 0 {
            return 127; // Default mass for nodes without size
        }
        let log_size = (self.file_size as f64).log2();
        let max_log = (1024.0 * 1024.0 * 1024.0_f64).log2(); // 1GB
        let normalized = (log_size / max_log).min(1.0);
        (normalized * 255.0) as u8
    }

    pub fn to_gpu_node(&self) -> GPUNode {
        GPUNode {
            x: self.x,
            y: self.y,
            z: self.z,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
            mass: self.calculate_mass(),
            flags: 0,
            padding: [0; 2],
        }
    }

    pub fn update_from_gpu_node(&mut self, gpu_node: &GPUNode) {
        self.x = gpu_node.x;
        self.y = gpu_node.y;
        self.z = gpu_node.z;
        self.vx = gpu_node.vx;
        self.vy = gpu_node.vy;
        self.vz = gpu_node.vz;
    }

    pub fn to_position_update(&self) -> NodePositionVelocity {
        NodePositionVelocity {
            x: self.x,
            y: self.y,
            z: self.z,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
        }
    }

    pub fn update_from_position_update(&mut self, update: &NodePositionVelocity) {
        self.x = update.x;
        self.y = update.y;
        self.z = update.z;
        self.vx = update.vx;
        self.vy = update.vy;
        self.vz = update.vz;
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
            metadata: HashMap::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }
}

/// GPU-compatible representation of a node, matching WGSL layout.
/// WGSL struct:
/// ```wgsl
/// struct Node {
///     position: vec3<f32>,  // 12 bytes
///     velocity: vec3<f32>,  // 12 bytes
///     mass: u8,            // 1 byte (quantized from file size)
///     flags: u8,           // 1 byte (can be used for node state)
///     padding: vec2<u8>,   // 2 bytes to align to 28 bytes total
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GPUNode {
    // position (vec3<f32>)
    pub x: f32,
    pub y: f32,
    pub z: f32,
    // velocity (vec3<f32>)
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    // Additional fields packed into 4 bytes
    pub mass: u8,    // Quantized mass from file size
    pub flags: u8,   // Node state flags
    pub padding: [u8; 2], // Padding for alignment
}

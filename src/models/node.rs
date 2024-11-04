use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

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
        }
    }
}

/// GPU-compatible representation of a node, matching WGSL layout.
/// WGSL struct:
/// ```wgsl
/// struct Node {
///     position: vec3<f32>,  // 12 bytes
///     velocity: vec3<f32>,  // 12 bytes
///     mass: f32,           // 4 bytes
///     padding1: u32,        // 4 bytes
/// }                        // Total: 32 bytes
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
    // Additional fields
    pub mass: f32,
    pub padding1: u32,  // Final padding to match WGSL struct
}

impl From<&Node> for GPUNode {
    fn from(node: &Node) -> Self {
        Self {
            x: node.x,
            y: node.y,
            z: node.z,
            vx: node.vx,
            vy: node.vy,
            vz: node.vz,
            mass: 1.0,
            padding1: 0,
        }
    }
}

impl Node {
    pub fn to_gpu_node(&self) -> GPUNode {
        GPUNode::from(self)
    }

    pub fn update_from_gpu_node(&mut self, gpu_node: &GPUNode) {
        self.x = gpu_node.x;
        self.y = gpu_node.y;
        self.z = gpu_node.z;
        self.vx = gpu_node.vx;
        self.vy = gpu_node.vy;
        self.vz = gpu_node.vz;
    }
}

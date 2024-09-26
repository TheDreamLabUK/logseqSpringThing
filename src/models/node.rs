// node.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

/// Represents a node in the graph.
#[derive(Default, Serialize, Deserialize, Clone)]
pub struct Node {
    /// Unique identifier for the node.
    pub id: String,
    /// Display label for the node.
    pub label: String,
    /// Additional metadata associated with the node.
    pub metadata: HashMap<String, String>,
    /// Position coordinates for visualisation.
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Velocity for simulation purposes.
    #[serde(skip)]
    pub vx: f32,
    #[serde(skip)]
    pub vy: f32,
    #[serde(skip)]
    pub vz: f32,
}

/// GPU-compatible representation of a node.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GPUNode {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
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

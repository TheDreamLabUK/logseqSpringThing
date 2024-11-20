use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::models::node::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    #[serde(rename = "target_node")]  // Rename for JSON serialization to match client expectations
    pub target: String,
    pub weight: f32,
}

// GPU representation of an edge, must match the shader's Edge struct
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUEdge {
    pub source: u32,      // 4 bytes
    pub target_idx: u32,  // 4 bytes
    pub weight: f32,      // 4 bytes
    pub padding1: u32,    // 4 bytes
    pub padding2: u32,    // 4 bytes
    pub padding3: u32,    // 4 bytes
    pub padding4: u32,    // 4 bytes
    pub padding5: u32,    // 4 bytes
}

impl Edge {
    pub fn new(source: String, target: String, weight: f32) -> Self {
        Self {
            source,
            target,
            weight,
        }
    }

    pub fn to_gpu_edge(&self, nodes: &[Node]) -> GPUEdge {
        // Create a temporary HashMap for efficient lookups
        let node_map: HashMap<_, _> = nodes.iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i as u32))
            .collect();

        let source_idx = node_map.get(&self.source).copied().unwrap_or(0);
        let target_idx = node_map.get(&self.target).copied().unwrap_or(0);

        GPUEdge {
            source: source_idx,
            target_idx,
            weight: self.weight,
            padding1: 0,
            padding2: 0,
            padding3: 0,
            padding4: 0,
            padding5: 0,
        }
    }
}

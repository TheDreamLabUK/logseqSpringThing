use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::models::node::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target_node: String,
    pub weight: f32,
    pub hyperlinks: f32,
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
    pub fn new(source: String, target_node: String, weight: f32, hyperlinks: f32) -> Self {
        Self {
            source,
            target_node,
            weight,
            hyperlinks,
        }
    }

    pub fn to_gpu_edge(&self, nodes: &[Node]) -> GPUEdge {
        // Create a temporary HashMap for efficient lookups
        let node_map: HashMap<_, _> = nodes.iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i as u32))
            .collect();

        let source_idx = node_map.get(&self.source).copied().unwrap_or(0);
        let target_idx = node_map.get(&self.target_node).copied().unwrap_or(0);

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

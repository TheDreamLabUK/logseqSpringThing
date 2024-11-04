use serde::{Serialize, Deserialize};
use bytemuck::{Pod, Zeroable};
use crate::models::node::Node;

/// Represents an edge connecting two nodes in the graph.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Edge {
    /// ID of the source node.
    pub source: String,
    /// ID of the target node.
    pub target_node: String,
    /// Weight of the edge (representing interconnectedness).
    pub weight: f32,
    /// Number of direct hyperlinks between the nodes.
    pub hyperlinks: f32,
}

/// GPU-compatible representation of an edge.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GPUEdge {
    pub source: u32,
    pub target_idx: u32,
    pub weight: f32,
    pub padding1: u32,
}

impl Edge {
    pub fn new(source: String, target_node: String, weight: f32, hyperlinks: f32) -> Self {
        Edge { source, target_node, weight, hyperlinks }
    }

    pub fn to_gpu_edge(&self, nodes: &[Node]) -> GPUEdge {
        let source_index = nodes.iter().position(|n| n.id == self.source).unwrap() as u32;
        let target_index = nodes.iter().position(|n| n.id == self.target_node).unwrap() as u32;
        GPUEdge {
            source: source_index,
            target_idx: target_index,
            weight: self.weight,
            padding1: 0,
        }
    }
}

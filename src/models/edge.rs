// models/edge.rs

use serde::{Serialize, Deserialize};
use bytemuck::{Pod, Zeroable};
use crate::models::node::Node; // Import Node from the node module

/// Represents an edge connecting two nodes in the graph.
#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    /// ID of the source node.
    pub source: String,
    /// ID of the target node.
    pub target: String,
    /// Weight of the edge.
    pub weight: f32,
}

/// GPU-compatible representation of an edge.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GPUEdge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
}

impl Edge {
    /// Creates a new `Edge` instance.
    ///
    /// # Arguments
    ///
    /// * `source` - ID of the source node.
    /// * `target` - ID of the target node.
    /// * `weight` - Weight of the edge.
    ///
    /// # Returns
    ///
    /// A new `Edge` instance.
    pub fn new(source: String, target: String, weight: f32) -> Self {
        Edge { source, target, weight }
    }

    /// Converts an `Edge` to a `GPUEdge` using node indices.
    ///
    /// This function finds the indices of the source and target nodes in the
    /// provided `nodes` vector and creates a `GPUEdge` using those indices.
    ///
    /// # Arguments
    ///
    /// * `nodes` - A slice of `Node` structs representing the nodes in the graph.
    ///
    /// # Returns
    ///
    /// A `GPUEdge` representing the edge with node indices instead of IDs.
    pub fn to_gpu_edge(&self, nodes: &[Node]) -> GPUEdge {
        let source_index = nodes.iter().position(|n| n.id == self.source).unwrap() as u32;
        let target_index = nodes.iter().position(|n| n.id == self.target).unwrap() as u32;
        GPUEdge {
            source: source_index,
            target: target_index,
            weight: self.weight,
        }
    }
}
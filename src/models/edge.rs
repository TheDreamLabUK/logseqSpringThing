// models/edge.rs

use serde::{Serialize, Deserialize};
use bytemuck::{Pod, Zeroable};
use crate::models::node::Node; // Import Node from the node module

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
    pub target_node: u32,
    pub weight: f32,
    pub hyperlinks: f32,
}

impl Edge {
    /// Creates a new `Edge` instance.
    ///
    /// # Arguments
    ///
    /// * `source` - ID of the source node.
    /// * `target_node` - ID of the target node.
    /// * `weight` - Weight of the edge (interconnectedness).
    /// * `hyperlinks` - Number of direct hyperlinks between the nodes.
    ///
    /// # Returns
    ///
    /// A new `Edge` instance.
    pub fn new(source: String, target_node: String, weight: f32, hyperlinks: f32) -> Self {
        Edge { source, target_node, weight, hyperlinks }
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
        let target_index = nodes.iter().position(|n| n.id == self.target_node).unwrap() as u32;
        GPUEdge {
            source: source_index,
            target_node: target_index,
            weight: self.weight,
            hyperlinks: self.hyperlinks,
        }
    }
}
// graph.rs

use super::node::Node;
use super::edge::Edge;
use super::metadata::Metadata;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents the graph data structure containing nodes and edges.
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct GraphData {
    /// List of nodes in the graph.
    pub nodes: Vec<Node>,
    /// List of edges connecting the nodes.
    pub edges: Vec<Edge>,
    /// Metadata associated with the graph.
    pub metadata: HashMap<String, Metadata>,
}

impl GraphData {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

// graph.rs

use super::node::Node;
use super::edge::Edge;
use serde::{Deserialize, Serialize};

/// Represents the graph data structure containing nodes and edges.
#[derive(Default, Serialize, Deserialize, Clone)]
pub struct GraphData {
    /// List of nodes in the graph.
    pub nodes: Vec<Node>,
    /// List of edges connecting the nodes.
    pub edges: Vec<Edge>,
}

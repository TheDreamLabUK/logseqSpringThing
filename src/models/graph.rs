use super::node::Node;
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
}

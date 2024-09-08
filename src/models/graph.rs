use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
pub struct GraphData {
    pub edges: Vec<Edge>,
    pub nodes: Vec<Node>,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Edge {
    // Define Edge fields
}

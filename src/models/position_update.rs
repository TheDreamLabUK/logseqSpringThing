use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::models::node::NodeData;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionUpdate {
    /// Maps node indices to their updated data
    pub nodes: HashMap<usize, NodeData>,
    /// Whether this is the final update in a sequence
    pub is_final: bool,
}

impl PositionUpdate {
    pub fn new(nodes: HashMap<usize, NodeData>, is_final: bool) -> Self {
        Self {
            nodes,
            is_final,
        }
    }
}

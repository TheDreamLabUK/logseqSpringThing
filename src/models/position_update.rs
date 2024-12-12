use serde::{Deserialize, Serialize};
use crate::utils::socket_flow_messages::NodeData;

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionUpdate {
    pub node_id: String,
    pub data: NodeData,
}

impl PositionUpdate {
    pub fn new(node_id: String, data: NodeData) -> Self {
        Self { node_id, data }
    }
}

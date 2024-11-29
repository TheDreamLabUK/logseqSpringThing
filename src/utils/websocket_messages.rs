use serde::{Serialize, Deserialize};

// Binary protocol is kept minimal for efficient position/velocity updates only
// Format: [4 bytes isInitialLayout flag][24 bytes per node (position + velocity)]

// Message types for non-binary communication
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "graphUpdate")]
    GraphUpdate {
        graph_data: GraphData,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        details: Option<String>,
    },
    #[serde(rename = "position_update_complete")]
    PositionUpdateComplete {
        status: String,
        is_initial_layout: bool,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity: Option<[f32; 3]>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
}

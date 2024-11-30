use serde::{Serialize, Deserialize};
use actix::prelude::*;
use serde_json::Value;

// Binary protocol is kept minimal for efficient position/velocity updates only
// Format: [4 bytes isInitialLayout flag][24 bytes per node (position + velocity)]

// Message types for non-binary communication
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "graphUpdate")]
    GraphUpdate {
        graph_data: Value,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        code: Option<String>,
        details: Option<String>,
    },
    #[serde(rename = "position_update_complete")]
    PositionUpdateComplete {
        status: String,
        is_initial_layout: bool,
    },
    #[serde(rename = "settings_updated")]
    SettingsUpdated {
        settings: Value,
    },
    #[serde(rename = "simulation_mode_set")]
    SimulationModeSet {
        mode: String,
        gpu_enabled: bool,
    },
    #[serde(rename = "fisheye_settings_updated")]
    FisheyeSettingsUpdated {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
}

// Message types for WebSocket communication
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastGraph {
    pub graph: GraphData,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastError {
    pub message: String,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIMessage(pub String);

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnected;

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnectionFailed;

#[derive(Message)]
#[rtype(result = "()")]
pub struct SendText(pub String);

#[derive(Message)]
#[rtype(result = "()")]
pub struct SendBinary(pub Vec<u8>);

pub trait MessageHandler {}

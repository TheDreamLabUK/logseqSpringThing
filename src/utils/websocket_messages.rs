use actix::prelude::*;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use crate::models::graph::GraphData;  // Import the proper GraphData struct

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "camelCase")]
pub enum ServerMessage {
    GraphUpdate {
        #[serde(rename = "graphData")]
        graph_data: GraphData,  // Use actual GraphData instead of Value
    },
    Error {
        message: String,
        code: Option<String>,
        details: Option<String>,
    },
    PositionUpdateComplete {
        status: String,
    },
    SettingsUpdated {
        settings: Value,
    },
    SimulationModeSet {
        mode: String,
        gpu_enabled: bool,
    },
    FisheyeSettingsUpdated {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    },
    InitialData {
        #[serde(rename = "graphData")]
        graph_data: GraphData,  // Use actual GraphData here too
        settings: Value,
    },
    GpuState {
        enabled: bool,
        node_count: usize,
        frame_time: f32,
    },
    LayoutState {
        iteration: usize,
        energy: f32,
        stable: bool,
    },
    AudioData {
        audio_data: String,
    },
    OpenAIResponse {
        text: String,
        audio: Option<String>,
    },
    RagflowResponse {
        answer: String,
        audio: Option<String>,
    },
    Completion {
        message: String,
    },
    BinaryPositionUpdate {
        is_initial_layout: bool,
    },
    Ping,
    Pong,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Edge {
    pub source: String,
    pub target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub directed: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodePosition {
    pub id: String,
    pub position: [f32; 3],
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdatePositionsMessage {
    pub nodes: Vec<NodePosition>,
}

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

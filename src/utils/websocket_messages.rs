use serde::{Serialize, Deserialize};
use actix::prelude::*;
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "graphUpdate")]
    GraphUpdate {
        graphData: Value,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        code: Option<String>,
        details: Option<String>,
    },
    #[serde(rename = "positionUpdateComplete")]
    PositionUpdateComplete {
        status: String,
        isInitialLayout: bool,
    },
    #[serde(rename = "settingsUpdated")]
    SettingsUpdated {
        settings: Value,
    },
    #[serde(rename = "simulationModeSet")]
    SimulationModeSet {
        mode: String,
        gpuEnabled: bool,
    },
    #[serde(rename = "fisheyeSettingsUpdated")]
    FisheyeSettingsUpdated {
        enabled: bool,
        strength: f32,
        focusPoint: [f32; 3],
        radius: f32,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: Value,
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
    pub nodeType: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub userData: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edgeType: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub userData: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub directed: Option<bool>,
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

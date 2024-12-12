use serde::{Serialize, Deserialize};
use serde_json::Value;
use crate::models::graph::GraphData;
use crate::models::node::NodeData;

/// Message types matching TypeScript MessageType
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "camelCase")]
pub enum ServerMessage {
    InitialData {
        #[serde(rename = "graphData")]
        graph_data: GraphData,
        settings: Value,
    },
    GraphUpdate {
        #[serde(rename = "graphData")]
        graph_data: GraphData,
    },
    BinaryPositionUpdate {
        node_count: usize,
        version: f32,
        is_initial_layout: bool,
    },
    SettingsUpdated {
        settings: Value,
    },
    Error {
        message: String,
        code: Option<String>,
        details: Option<String>,
    },
    PositionUpdateComplete {
        status: String,
    },
    SimulationModeSet {
        mode: String,
        gpu_enabled: bool,
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
    Ping,
    Pong,
}

/// Node data for JSON messages (camelCase for TypeScript compatibility)
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    pub id: String,
    pub position: [f32; 3],  // Matches THREE.Vector3
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity: Option<[f32; 3]>,  // Optional for client messages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<NodeData>,  // Additional node data
}

/// Position update message from client
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdatePositionsMessage {
    pub nodes: Vec<Node>,
}

/// Client messages matching TypeScript interface
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum ClientMessage {
    EnableBinaryUpdates,
    UpdatePositions(UpdatePositionsMessage),
    RequestInitialData,
    UpdateSettings { settings: Value },
    SetSimulationMode { mode: String },
    Ping,
}

/// Binary message format (packed for efficient transfer)
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BinaryNodeData {
    pub position: [f32; 3],  // 12 bytes
    pub velocity: [f32; 3],  // 12 bytes
}

impl BinaryNodeData {
    pub fn new(position: [f32; 3], velocity: [f32; 3]) -> Self {
        Self {
            position,
            velocity,
        }
    }

    pub fn from_node_data(data: &NodeData) -> Self {
        Self {
            position: data.position,
            velocity: data.velocity,
        }
    }
}

// Ensure binary layout matches expectations
const _: () = assert!(std::mem::size_of::<BinaryNodeData>() == 24);

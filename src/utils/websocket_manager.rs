use actix::prelude::*;
use actix_web::{web, Error as ActixError, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use log::{debug, error};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::models::node::GPUNode;
use crate::models::graph::GraphData;
use crate::utils::websocket_messages::{SendBinary, SendText, ServerMessage};
use crate::AppState;
use crate::services::ragflow_service::RAGFlowService;
use crate::handlers::websocket_handlers::WebSocketSession;

pub struct WebSocketManager {
    pub sessions: Arc<RwLock<Vec<Addr<WebSocketSession>>>>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        WebSocketManager {
            sessions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize the WebSocket manager
    pub async fn initialize(&self, _ragflow_service: &RAGFlowService) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Initializing WebSocket manager");
        // Clear any existing sessions
        let mut sessions = self.sessions.write().await;
        sessions.clear();
        Ok(())
    }

    /// Handle new WebSocket connections
    pub async fn handle_websocket(
        req: HttpRequest,
        stream: web::Payload,
        state: web::Data<AppState>
    ) -> Result<HttpResponse, ActixError> {
        debug!("New WebSocket connection request");
        let session = WebSocketSession::new(state);
        ws::start(session, &req, stream)
    }

    /// Broadcast graph structure updates to all connected clients
    pub async fn broadcast_graph_update(&self, graph: &GraphData) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Broadcasting graph update: {} nodes, {} edges", 
            graph.nodes.len(), graph.edges.len());

        let message = ServerMessage::GraphUpdate {
            graph_data: json!({
                "nodes": graph.nodes,
                "edges": graph.edges,
                "metadata": &graph.metadata
            })
        };

        if let Ok(message_str) = serde_json::to_string(&message) {
            let sessions = self.sessions.read().await;
            for session in sessions.iter() {
                if let Err(e) = session.try_send(SendText(message_str.clone())) {
                    error!("Failed to send graph update to session: {}", e);
                }
            }
            debug!("Graph update broadcast complete: {} recipients", sessions.len());
        }

        Ok(())
    }

    /// Broadcast binary position updates to all connected clients
    pub async fn broadcast_binary(&self, nodes: &[GPUNode], is_initial: bool) {
        debug!("Broadcasting binary update: {} nodes, initial: {}", nodes.len(), is_initial);

        // Create binary message
        let mut binary_data = Vec::with_capacity(4 + nodes.len() * 24);
        
        // Add initial layout flag as float32
        let flag_bytes = (if is_initial { 1.0f32 } else { 0.0f32 }).to_le_bytes();
        binary_data.extend_from_slice(&flag_bytes);

        // Add node positions and velocities
        for node in nodes {
            let update = [
                node.x, node.y, node.z,
                node.vx, node.vy, node.vz,
            ];
            for &value in &update {
                binary_data.extend_from_slice(&value.to_le_bytes());
            }
        }

        // Send to all connected clients
        let sessions = self.sessions.read().await;
        for session in sessions.iter() {
            if let Err(e) = session.try_send(SendBinary(binary_data.clone())) {
                error!("Failed to send binary update to session: {}", e);
            }
        }
        debug!("Binary update broadcast complete: {} recipients", sessions.len());
    }

    /// Broadcast error messages to all connected clients
    pub async fn broadcast_error(&self, message: &str, code: Option<&str>) {
        debug!("Broadcasting error: {}, code: {:?}", message, code);

        let error_message = ServerMessage::Error {
            message: message.to_string(),
            code: code.map(String::from),
        };

        if let Ok(message_str) = serde_json::to_string(&error_message) {
            let sessions = self.sessions.read().await;
            for session in sessions.iter() {
                if let Err(e) = session.try_send(SendText(message_str.clone())) {
                    error!("Failed to send error message to session: {}", e);
                }
            }
            debug!("Error broadcast complete: {} recipients", sessions.len());
        }
    }

    /// Broadcast audio data to all connected clients
    pub async fn broadcast_audio(&self, data: &[u8]) {
        debug!("Broadcasting audio data: {} bytes", data.len());

        let audio_message = json!({
            "type": "audio",
            "data": data,
        });

        if let Ok(message_str) = serde_json::to_string(&audio_message) {
            let sessions = self.sessions.read().await;
            for session in sessions.iter() {
                if let Err(e) = session.try_send(SendText(message_str.clone())) {
                    error!("Failed to send audio data to session: {}", e);
                }
            }
            debug!("Audio broadcast complete: {} recipients", sessions.len());
        }
    }

    pub async fn add_session(&self, addr: Addr<WebSocketSession>) {
        let mut sessions = self.sessions.write().await;
        debug!("Adding new WebSocket session. Total sessions: {}", sessions.len() + 1);
        sessions.push(addr);
    }

    pub async fn remove_session(&self, addr: &Addr<WebSocketSession>) {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|x| x != addr);
        debug!("Removed WebSocket session. Remaining sessions: {}", sessions.len());
    }
}

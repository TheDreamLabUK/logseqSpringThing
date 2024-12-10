use socket_flow::event::{Event, ID};
use socket_flow::server::start_server;
use socket_flow::split::WSWriter;
use socket_flow::message::Message;
use futures::StreamExt;
use log::{error, info, warn};
use std::collections::HashMap;
use std::sync::Arc;

use crate::utils::socket_flow_messages::{ServerMessage, UpdatePositionsMessage};
use crate::models::position_update::NodePositionVelocity;
use crate::utils::socket_flow_constants::{NODE_SIZE, BINARY_CHUNK_SIZE};
use crate::AppState;

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
}

impl SocketFlowServer {
    pub fn new(app_state: Arc<AppState>) -> Self {
        Self { app_state }
    }

    pub async fn start(self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        info!("Server started on address 127.0.0.1:{}", port);
        let mut event_receiver = start_server(port).await?;
        let mut clients: HashMap<ID, WSWriter> = HashMap::new();

        while let Some(event) = event_receiver.next().await {
            match event {
                Event::NewClient(id, client_conn) => {
                    info!("New client {} connected", id);
                    clients.insert(id, client_conn);

                    // Send initial data
                    let graph = self.app_state.graph_service.graph_data.read().await;
                    let settings = self.app_state.settings.read().await;

                    let initial_data = ServerMessage::InitialData {
                        graph_data: (*graph).clone(),
                        settings: serde_json::to_value(&*settings)?,
                    };

                    let message = serde_json::to_string(&initial_data)?;
                    if let Some(writer) = clients.get_mut(&id) {
                        if let Err(e) = writer.send_message(Message::Text(message)).await {
                            error!("Failed to send initial data to client {}: {:?}", id, e);
                        }
                    }
                }
                Event::NewMessage(client_id, message) => {
                    match message {
                        Message::Text(text) => {
                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                                match value.get("type").and_then(|t| t.as_str()) {
                                    Some("updatePositions") => {
                                        if let Ok(update_msg) = serde_json::from_value::<UpdatePositionsMessage>(value) {
                                            // Update positions
                                            let mut graph = self.app_state.graph_service.graph_data.write().await;
                                            for node_pos in &update_msg.nodes {
                                                if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_pos.id) {
                                                    node.x = node_pos.position[0];
                                                    node.y = node_pos.position[1];
                                                    node.z = node_pos.position[2];
                                                    node.position = Some(node_pos.position);
                                                }
                                            }

                                            // Broadcast update
                                            let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                                                x: node.x,
                                                y: node.y,
                                                z: node.z,
                                                vx: 0.0,
                                                vy: 0.0,
                                                vz: 0.0,
                                            }).collect();

                                            let nodes_per_chunk = BINARY_CHUNK_SIZE / NODE_SIZE as usize;

                                            for writer in clients.values_mut() {
                                                for chunk in nodes.chunks(nodes_per_chunk) {
                                                    let mut binary_data = Vec::with_capacity(chunk.len() * NODE_SIZE as usize);
                                                    for node in chunk {
                                                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                                                    }

                                                    if let Err(e) = writer.send_message(Message::Binary(binary_data)).await {
                                                        error!("Failed to send position update: {:?}", e);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Some("simulation_mode") => {
                                        if let Some(mode) = value.get("mode").and_then(|m| m.as_str()) {
                                            if let Some(writer) = clients.get_mut(&client_id) {
                                                let response = ServerMessage::SimulationModeSet {
                                                    mode: mode.to_string(),
                                                    gpu_enabled: matches!(mode, "remote" | "gpu"),
                                                };

                                                if let Ok(message) = serde_json::to_string(&response) {
                                                    if let Err(e) = writer.send_message(Message::Text(message)).await {
                                                        error!("Failed to send simulation mode response: {:?}", e);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        warn!("Unknown message type from client {}", client_id);
                                    }
                                }
                            }
                        }
                        Message::Binary(data) => {
                            if data.len() % NODE_SIZE as usize != 0 {
                                error!("Invalid binary data length");
                                continue;
                            }

                            // Process binary update and broadcast to other clients
                            let nodes_per_chunk = BINARY_CHUNK_SIZE / NODE_SIZE as usize;

                            let graph = self.app_state.graph_service.graph_data.read().await;
                            let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                                x: node.x,
                                y: node.y,
                                z: node.z,
                                vx: 0.0,
                                vy: 0.0,
                                vz: 0.0,
                            }).collect();

                            for writer in clients.values_mut() {
                                for chunk in nodes.chunks(nodes_per_chunk) {
                                    let mut binary_data = Vec::with_capacity(chunk.len() * NODE_SIZE as usize);
                                    for node in chunk {
                                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                                    }

                                    if let Err(e) = writer.send_message(Message::Binary(binary_data)).await {
                                        error!("Failed to send position update: {:?}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                Event::Disconnect(client_id) => {
                    info!("Client {} disconnected", client_id);
                    clients.remove(&client_id);
                }
                Event::Error(client_id, error) => {
                    error!("Error occurred for client {}: {:?}", client_id, error);
                }
            }
        }

        Ok(())
    }
}

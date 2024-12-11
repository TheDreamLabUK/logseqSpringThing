use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use std::sync::Arc;
use log::{error, warn}; // Add log macros
use crate::utils::socket_flow_messages::{ServerMessage, UpdatePositionsMessage};
use crate::models::position_update::NodePositionVelocity;
use crate::utils::socket_flow_constants::{NODE_POSITION_SIZE, BINARY_HEADER_SIZE};
use crate::AppState;
use crate::{log_websocket, log_data};

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    initial_data_sent: bool,
    binary_updates_enabled: bool,
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        log_websocket!("New WebSocket connection established");
        
        // Send initial data when the connection is established
        let app_state = self.app_state.clone();
        let fut = async move {
            let graph = app_state.graph_service.graph_data.read().await;
            let settings = app_state.settings.read().await;

            log_data!("Preparing initial graph data with {} nodes", graph.nodes.len());

            let initial_data = ServerMessage::InitialData {
                graph_data: (*graph).clone(),
                settings: serde_json::to_value(&*settings).unwrap_or_default(),
            };

            serde_json::to_string(&initial_data)
        };

        let fut = fut.into_actor(self).map(|res, actor, ctx| {
            match res {
                Ok(message) => {
                    log_websocket!("Sending initial data to client");
                    // Log full JSON when enabled
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&message) {
                        log_data!("Initial data JSON: {}", serde_json::to_string_pretty(&json).unwrap_or_default());
                    }
                    actor.initial_data_sent = true;
                    ctx.text(message);
                }
                Err(e) => {
                    error!("Failed to serialize initial data: {}", e);
                }
            }
        });
        ctx.spawn(fut);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                log_websocket!("Received ping message");
                ctx.pong(&msg)
            },
            Ok(ws::Message::Text(text)) => {
                log_websocket!("Received text message");
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                    log_data!("Parsed text message: {}", value);
                    match value.get("type").and_then(|t| t.as_str()) {
                        Some("initialData") => {
                            log_websocket!("Processing initialData request");
                            // Client is requesting initial data, send it again
                            let app_state = self.app_state.clone();
                            let fut = async move {
                                let graph = app_state.graph_service.graph_data.read().await;
                                let settings = app_state.settings.read().await;

                                log_data!("Re-preparing initial graph data with {} nodes", graph.nodes.len());

                                let initial_data = ServerMessage::InitialData {
                                    graph_data: (*graph).clone(),
                                    settings: serde_json::to_value(&*settings).unwrap_or_default(),
                                };

                                serde_json::to_string(&initial_data)
                            };

                            let fut = fut.into_actor(self).map(|res, actor, ctx| {
                                match res {
                                    Ok(message) => {
                                        log_websocket!("Re-sending initial data to client");
                                        // Log full JSON when enabled
                                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&message) {
                                            log_data!("Initial data JSON: {}", serde_json::to_string_pretty(&json).unwrap_or_default());
                                        }
                                        actor.initial_data_sent = true;
                                        ctx.text(message);
                                    }
                                    Err(e) => {
                                        error!("Failed to serialize initial data: {}", e);
                                    }
                                }
                            });
                            ctx.spawn(fut);
                        },
                        Some("enableBinaryUpdates") => {
                            if !self.initial_data_sent {
                                log_websocket!("Ignoring binary update request before initial data");
                                return;
                            }
                            log_websocket!("Enabling binary updates");
                            self.binary_updates_enabled = true;
                        },
                        Some("updatePositions") => {
                            // Only process position updates after initial data is sent
                            if !self.initial_data_sent {
                                log_websocket!("Ignoring position update before initial data");
                                return;
                            }

                            log_websocket!("Processing updatePositions message");
                            if let Ok(update_msg) = serde_json::from_value::<UpdatePositionsMessage>(value) {
                                log_data!("Updating positions for {} nodes", update_msg.nodes.len());
                                let app_state = self.app_state.clone();
                                let fut = async move {
                                    let mut graph = app_state.graph_service.graph_data.write().await;
                                    for node_update in update_msg.nodes {
                                        if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_update.id) {
                                            node.position = Some(node_update.position);
                                        }
                                    }

                                    // Prepare binary response
                                    let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                                        x: node.position.map(|p| p[0]).unwrap_or(0.0),
                                        y: node.position.map(|p| p[1]).unwrap_or(0.0),
                                        z: node.position.map(|p| p[2]).unwrap_or(0.0),
                                        vx: 0.0,
                                        vy: 0.0,
                                        vz: 0.0,
                                    }).collect();

                                    log_data!("Preparing binary response for {} nodes", nodes.len());
                                    let mut binary_data = Vec::with_capacity(BINARY_HEADER_SIZE + nodes.len() * NODE_POSITION_SIZE);
                                    
                                    // Add header (version 1.0)
                                    binary_data.extend_from_slice(&1.0f32.to_le_bytes());
                                    
                                    // Add node data
                                    for node in &nodes {
                                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                                    }

                                    binary_data
                                };

                                let fut = fut.into_actor(self).map(|binary_data, actor, ctx| {
                                    if actor.binary_updates_enabled {
                                        // First send a message indicating binary data is coming
                                        if let Ok(message) = serde_json::to_string(&ServerMessage::BinaryPositionUpdate {
                                            is_initial_layout: false,
                                        }) {
                                            ctx.text(message);
                                        }
                                        log_websocket!("Sending binary response: {} bytes", binary_data.len());
                                        ctx.binary(binary_data);
                                    }
                                });
                                ctx.spawn(fut);
                            }
                        }
                        Some("simulation_mode") => {
                            log_websocket!("Processing simulation_mode message");
                            if let Some(mode) = value.get("mode").and_then(|m| m.as_str()) {
                                log_data!("Setting simulation mode to: {}", mode);
                                let response = ServerMessage::SimulationModeSet {
                                    mode: mode.to_string(),
                                    gpu_enabled: matches!(mode, "remote" | "gpu"),
                                };

                                if let Ok(message) = serde_json::to_string(&response) {
                                    log_websocket!("Sending simulation mode response");
                                    ctx.text(message);
                                }
                            }
                        }
                        _ => {
                            warn!("Unknown message type: {:?}", value.get("type"));
                        }
                    }
                }
            }
            Ok(ws::Message::Binary(bin)) => {
                // Only process binary messages after initial data is sent and binary updates enabled
                if !self.initial_data_sent || !self.binary_updates_enabled {
                    log_websocket!("Ignoring binary message before initialization");
                    return;
                }

                log_websocket!("Received binary message: {} bytes", bin.len());
                let expected_size = BINARY_HEADER_SIZE + (bin.len() - BINARY_HEADER_SIZE) / NODE_POSITION_SIZE * NODE_POSITION_SIZE;
                if bin.len() != expected_size {
                    error!("Invalid binary data length: {} bytes (expected {})", bin.len(), expected_size);
                    return;
                }

                let node_count = (bin.len() - BINARY_HEADER_SIZE) / NODE_POSITION_SIZE;
                log_data!("Processing binary data for {} nodes", node_count);

                let app_state = self.app_state.clone();
                let fut = async move {
                    let graph = app_state.graph_service.graph_data.read().await;
                    let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                        x: node.position.map(|p| p[0]).unwrap_or(0.0),
                        y: node.position.map(|p| p[1]).unwrap_or(0.0),
                        z: node.position.map(|p| p[2]).unwrap_or(0.0),
                        vx: 0.0,
                        vy: 0.0,
                        vz: 0.0,
                    }).collect();

                    log_data!("Preparing binary response for {} nodes", nodes.len());
                    let mut binary_data = Vec::with_capacity(BINARY_HEADER_SIZE + nodes.len() * NODE_POSITION_SIZE);
                    
                    // Add header (version 1.0)
                    binary_data.extend_from_slice(&1.0f32.to_le_bytes());
                    
                    // Add node data
                    for node in &nodes {
                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                    }

                    binary_data
                };

                let fut = fut.into_actor(self).map(|binary_data, actor, ctx| {
                    if actor.binary_updates_enabled {
                        // First send a message indicating binary data is coming
                        if let Ok(message) = serde_json::to_string(&ServerMessage::BinaryPositionUpdate {
                            is_initial_layout: false,
                        }) {
                            ctx.text(message);
                        }
                        log_websocket!("Sending binary response: {} bytes", binary_data.len());
                        ctx.binary(binary_data);
                    }
                });
                ctx.spawn(fut);
            }
            Ok(ws::Message::Close(reason)) => {
                log_websocket!("Client disconnected: {:?}", reason);
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

impl SocketFlowServer {
    pub fn new(app_state: Arc<AppState>) -> Self {
        Self { 
            app_state,
            initial_data_sent: false,
            binary_updates_enabled: false,
        }
    }
}

// WebSocket handler for Actix
pub async fn ws_handler(req: HttpRequest, stream: web::Payload, app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    log_websocket!("New WebSocket connection request from {}", req.peer_addr().map(|addr| addr.to_string()).unwrap_or_else(|| "unknown".to_string()));
    let socket_flow = SocketFlowServer::new(Arc::new(app_state.as_ref().clone()));
    ws::start(socket_flow, &req, stream)
}

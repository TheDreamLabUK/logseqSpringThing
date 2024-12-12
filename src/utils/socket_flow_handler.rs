use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use std::sync::Arc;
use log::{error, warn};
use crate::utils::socket_flow_messages::{ServerMessage, ClientMessage, BinaryNodeData};
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
                if let Ok(client_msg) = serde_json::from_str::<ClientMessage>(&text) {
                    match client_msg {
                        ClientMessage::RequestInitialData => {
                            log_websocket!("Processing initialData request");
                            let app_state = self.app_state.clone();
                            let fut = async move {
                                let graph = app_state.graph_service.graph_data.read().await;
                                let settings = app_state.settings.read().await;

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
                        ClientMessage::EnableBinaryUpdates => {
                            if !self.initial_data_sent {
                                log_websocket!("Ignoring binary update request before initial data");
                                return;
                            }
                            log_websocket!("Enabling binary updates");
                            self.binary_updates_enabled = true;
                        },
                        ClientMessage::UpdatePositions(update_msg) => {
                            if !self.initial_data_sent {
                                log_websocket!("Ignoring position update before initial data");
                                return;
                            }

                            log_websocket!("Processing updatePositions message");
                            let app_state = self.app_state.clone();
                            let fut = async move {
                                let mut graph = app_state.graph_service.graph_data.write().await;
                                for node_update in update_msg.nodes {
                                    if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_update.id) {
                                        node.data.position = node_update.position;
                                        if let Some(vel) = node_update.velocity {
                                            node.data.velocity = vel;
                                        }
                                    }
                                }

                                // Prepare binary response
                                let binary_nodes: Vec<BinaryNodeData> = graph.nodes.iter()
                                    .map(|node| BinaryNodeData::from_node_data(&node.data))
                                    .collect();

                                log_data!("Preparing binary response for {} nodes", binary_nodes.len());
                                let mut binary_data = Vec::with_capacity(BINARY_HEADER_SIZE + binary_nodes.len() * NODE_POSITION_SIZE);
                                
                                // Add header (version 1.0)
                                binary_data.extend_from_slice(&1.0f32.to_le_bytes());
                                
                                // Add node data
                                for node in &binary_nodes {
                                    binary_data.extend_from_slice(bytemuck::bytes_of(node));
                                }

                                binary_data
                            };

                            let fut = fut.into_actor(self).map(|binary_data, actor, ctx| {
                                if actor.binary_updates_enabled {
                                    // First send a message indicating binary data is coming
                                    if let Ok(message) = serde_json::to_string(&ServerMessage::BinaryPositionUpdate {
                                        node_count: binary_data.len() / NODE_POSITION_SIZE,
                                        version: 1.0,
                                        is_initial_layout: false,
                                    }) {
                                        ctx.text(message);
                                    }
                                    log_websocket!("Sending binary response: {} bytes", binary_data.len());
                                    ctx.binary(binary_data);
                                }
                            });
                            ctx.spawn(fut);
                        },
                        ClientMessage::SetSimulationMode { mode } => {
                            log_websocket!("Setting simulation mode to: {}", mode);
                            let response = ServerMessage::SimulationModeSet {
                                mode,
                                gpu_enabled: true, // Always use GPU with vec3
                            };

                            if let Ok(message) = serde_json::to_string(&response) {
                                log_websocket!("Sending simulation mode response");
                                ctx.text(message);
                            }
                        },
                        ClientMessage::UpdateSettings { settings } => {
                            // Handle settings update
                            if let Ok(message) = serde_json::to_string(&ServerMessage::SettingsUpdated { settings }) {
                                ctx.text(message);
                            }
                        },
                        ClientMessage::Ping => {
                            if let Ok(message) = serde_json::to_string(&ServerMessage::Pong) {
                                ctx.text(message);
                            }
                        },
                    }
                }
            },
            Ok(ws::Message::Binary(bin)) => {
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
                    let binary_nodes: Vec<BinaryNodeData> = graph.nodes.iter()
                        .map(|node| BinaryNodeData::from_node_data(&node.data))
                        .collect();

                    let mut binary_data = Vec::with_capacity(BINARY_HEADER_SIZE + binary_nodes.len() * NODE_POSITION_SIZE);
                    binary_data.extend_from_slice(&1.0f32.to_le_bytes());
                    
                    for node in &binary_nodes {
                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                    }

                    binary_data
                };

                let fut = fut.into_actor(self).map(|binary_data, actor, ctx| {
                    if actor.binary_updates_enabled {
                        if let Ok(message) = serde_json::to_string(&ServerMessage::BinaryPositionUpdate {
                            node_count: binary_data.len() / NODE_POSITION_SIZE,
                            version: 1.0,
                            is_initial_layout: false,
                        }) {
                            ctx.text(message);
                        }
                        log_websocket!("Sending binary response: {} bytes", binary_data.len());
                        ctx.binary(binary_data);
                    }
                });
                ctx.spawn(fut);
            },
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

pub async fn ws_handler(req: HttpRequest, stream: web::Payload, app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    log_websocket!("New WebSocket connection request from {}", req.peer_addr().map(|addr| addr.to_string()).unwrap_or_else(|| "unknown".to_string()));
    let socket_flow = SocketFlowServer::new(Arc::new(app_state.as_ref().clone()));
    ws::start(socket_flow, &req, stream)
}

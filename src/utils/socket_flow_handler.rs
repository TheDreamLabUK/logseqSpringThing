use actix::{Actor, ActorContext, AsyncContext, StreamHandler, WrapFuture, ActorFutureExt};
use actix_web_actors::ws;
use log::{error, info, warn};
use std::sync::Arc;
use actix_web::{web, Error, HttpRequest, HttpResponse};

use crate::utils::socket_flow_messages::{ServerMessage, UpdatePositionsMessage};
use crate::models::position_update::NodePositionVelocity;
use crate::utils::socket_flow_constants::NODE_SIZE;
use crate::AppState;
use crate::{log_websocket, log_data};
use crate::utils::debug_logging::WsDebugData;

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
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

        let fut = fut.into_actor(self).map(|res, _actor, ctx| {
            match res {
                Ok(message) => {
                    log_websocket!("Sending initial data to client");
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&message) {
                        log_data!("Initial data JSON: {}", json);
                    }
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

                            let fut = fut.into_actor(self).map(|res, _actor, ctx| {
                                match res {
                                    Ok(message) => {
                                        log_websocket!("Re-sending initial data to client");
                                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&message) {
                                            log_data!("Initial data JSON: {}", json);
                                        }
                                        ctx.text(message);
                                    }
                                    Err(e) => {
                                        error!("Failed to serialize initial data: {}", e);
                                    }
                                }
                            });
                            ctx.spawn(fut);
                        },
                        Some("updatePositions") => {
                            log_websocket!("Processing updatePositions message");
                            if let Ok(update_msg) = serde_json::from_value::<UpdatePositionsMessage>(value) {
                                log_data!("Updating positions for {} nodes", update_msg.nodes.len());
                                let app_state = self.app_state.clone();
                                let fut = async move {
                                    let mut graph = app_state.graph_service.graph_data.write().await;
                                    for node_pos in &update_msg.nodes {
                                        if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_pos.id) {
                                            node.x = node_pos.position[0];
                                            node.y = node_pos.position[1];
                                            node.z = node_pos.position[2];
                                            node.position = Some(node_pos.position);
                                        }
                                    }

                                    let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                                        x: node.x,
                                        y: node.y,
                                        z: node.z,
                                        vx: 0.0,
                                        vy: 0.0,
                                        vz: 0.0,
                                    }).collect();

                                    log_data!("Preparing binary response for {} nodes", nodes.len());
                                    let mut binary_data = Vec::new();
                                    for node in &nodes {
                                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                                    }

                                    binary_data
                                };

                                let fut = fut.into_actor(self).map(|binary_data, _actor, ctx| {
                                    log_websocket!("Sending binary response: {} bytes", binary_data.len());
                                    ctx.binary(binary_data);
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
                log_websocket!("Received binary message: {} bytes", bin.len());
                if bin.len() % NODE_SIZE as usize != 0 {
                    error!("Invalid binary data length: {} bytes (not divisible by NODE_SIZE {})", bin.len(), NODE_SIZE);
                    return;
                }

                let node_count = bin.len() / NODE_SIZE as usize;
                log_data!("Processing binary data for {} nodes", node_count);

                let app_state = self.app_state.clone();
                let fut = async move {
                    let graph = app_state.graph_service.graph_data.read().await;
                    let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                        x: node.x,
                        y: node.y,
                        z: node.z,
                        vx: 0.0,
                        vy: 0.0,
                        vz: 0.0,
                    }).collect();

                    log_data!("Preparing binary response for {} nodes", nodes.len());
                    let mut binary_data = Vec::new();
                    for node in &nodes {
                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                    }

                    binary_data
                };

                let fut = fut.into_actor(self).map(|binary_data, _actor, ctx| {
                    log_websocket!("Sending binary response: {} bytes", binary_data.len());
                    ctx.binary(binary_data);
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
        Self { app_state }
    }
}

// WebSocket handler for Actix
pub async fn ws_handler(req: HttpRequest, stream: web::Payload, app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    log_websocket!("New WebSocket connection request from {}", req.peer_addr().map(|addr| addr.to_string()).unwrap_or_else(|| "unknown".to_string()));
    let socket_flow = SocketFlowServer::new(Arc::new(app_state.as_ref().clone()));
    ws::start(socket_flow, &req, stream)
}

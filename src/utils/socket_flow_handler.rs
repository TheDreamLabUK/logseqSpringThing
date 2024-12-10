use actix::{Actor, ActorContext, AsyncContext, StreamHandler, WrapFuture};
use actix_web_actors::ws;
use log::{error, info, warn};
use std::sync::Arc;
use actix_web::{web, Error, HttpRequest, HttpResponse};

use crate::utils::socket_flow_messages::{ServerMessage, UpdatePositionsMessage};
use crate::models::position_update::NodePositionVelocity;
use crate::utils::socket_flow_constants::NODE_SIZE;
use crate::AppState;

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Send initial data when the connection is established
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

        ctx.spawn(Box::pin(async move {
            if let Ok(message) = fut.await.unwrap_or_default() {
                ctx.text(message);
            }
        }.into_actor(self)));
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                    match value.get("type").and_then(|t| t.as_str()) {
                        Some("updatePositions") => {
                            if let Ok(update_msg) = serde_json::from_value::<UpdatePositionsMessage>(value) {
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

                                    let mut binary_data = Vec::new();
                                    for node in &nodes {
                                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                                    }

                                    binary_data
                                };

                                ctx.spawn(Box::pin(async move {
                                    let binary_data = fut.await;
                                    ctx.binary(binary_data);
                                }.into_actor(self)));
                            }
                        }
                        Some("simulation_mode") => {
                            if let Some(mode) = value.get("mode").and_then(|m| m.as_str()) {
                                let response = ServerMessage::SimulationModeSet {
                                    mode: mode.to_string(),
                                    gpu_enabled: matches!(mode, "remote" | "gpu"),
                                };

                                if let Ok(message) = serde_json::to_string(&response) {
                                    ctx.text(message);
                                }
                            }
                        }
                        _ => {
                            warn!("Unknown message type");
                        }
                    }
                }
            }
            Ok(ws::Message::Binary(bin)) => {
                if bin.len() % NODE_SIZE as usize != 0 {
                    error!("Invalid binary data length");
                    return;
                }

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

                    let mut binary_data = Vec::new();
                    for node in &nodes {
                        binary_data.extend_from_slice(bytemuck::bytes_of(node));
                    }

                    binary_data
                };

                ctx.spawn(Box::pin(async move {
                    let binary_data = fut.await;
                    ctx.binary(binary_data);
                }.into_actor(self)));
            }
            Ok(ws::Message::Close(reason)) => {
                info!("Client disconnected: {:?}", reason);
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
    let socket_flow = SocketFlowServer::new(Arc::new(app_state.as_ref().clone()));
    ws::start(socket_flow, &req, stream)
}

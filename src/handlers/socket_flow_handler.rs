use std::sync::Arc;
use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use tokio::sync::RwLock;
use log::{debug, info, warn, error};
use flate2::{write::ZlibEncoder, read::ZlibDecoder, Compression};
use std::io::{Write, Read};

use crate::app_state::AppState;
use crate::utils::socket_flow_messages::{PingMessage, PongMessage};
use crate::utils::binary_protocol::{self, MessageType, NodeData};

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    settings: Arc<RwLock<crate::config::Settings>>,
    last_ping: Option<u64>,
    update_interval: std::time::Duration,
}

impl SocketFlowServer {
    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<crate::config::Settings>>) -> Self {
        // Calculate update interval from settings
        let update_rate = settings
            .try_read()
            .map(|s| s.websocket.binary_update_rate)
            .unwrap_or(30);
        
        let update_interval = std::time::Duration::from_millis((1000.0 / update_rate as f64) as u64);
        
        Self {
            app_state,
            settings,
            last_ping: None,
            update_interval,
        }
    }

    fn handle_ping(&mut self, msg: PingMessage) -> PongMessage {
        self.last_ping = Some(msg.timestamp);
        PongMessage {
            type_: "pong".to_string(),
            timestamp: msg.timestamp,
        }
    }

    fn maybe_compress(&self, data: Vec<u8>) -> Vec<u8> {
        if let Ok(settings) = self.settings.try_read() {
            if settings.websocket.compression_enabled && data.len() >= settings.websocket.compression_threshold {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
                if encoder.write_all(&data).is_ok() {
                    if let Ok(compressed) = encoder.finish() {
                        if compressed.len() < data.len() {
                            debug!("Compressed binary message: {} -> {} bytes", data.len(), compressed.len());
                            return compressed;
                        }
                    }
                }
            }
        }
        data
    }

    fn maybe_decompress(&self, data: Vec<u8>) -> Result<Vec<u8>, String> {
        if let Ok(settings) = self.settings.try_read() {
            if settings.websocket.compression_enabled {
                let mut decoder = ZlibDecoder::new(data.as_slice());
                let mut decompressed = Vec::new();
                match decoder.read_to_end(&mut decompressed) {
                    Ok(_) => {
                        if decompressed.len() > data.len() {
                            debug!("Decompressed binary message: {} -> {} bytes", data.len(), decompressed.len());
                            return Ok(decompressed);
                        }
                    }
                    Err(e) => {
                        // If decompression fails, assume the data wasn't compressed
                        debug!("Decompression failed (data likely uncompressed): {}", e);
                    }
                }
            }
        }
        Ok(data)
    }
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocket] Client connected");
        
        // Clone Arc references for the interval closure
        let app_state = self.app_state.clone();
        
        ctx.run_interval(self.update_interval, move |actor, ctx| {
            // Get current node positions and velocities
            let app_state_clone = app_state.clone();
            
            // Spawn a future to get positions
            let fut = async move {
                let raw_nodes = app_state_clone.graph_service.get_node_positions().await;
                
                // Convert to binary protocol NodeData format
                let nodes: Vec<NodeData> = raw_nodes.into_iter()
                    .map(|node| NodeData {
                        id: node.id,
                        position: node.data.position,
                        velocity: node.data.velocity,
                    })
                    .collect();
                
                // Encode using binary protocol
                binary_protocol::encode_node_data(&nodes, MessageType::FullStateUpdate)
            };
            
            // Convert the future to an actix future and handle it
            let fut = fut.into_actor(actor);
            ctx.spawn(fut.map(|binary_data, actor, ctx| {
                // Compress if enabled and threshold met
                let final_data = actor.maybe_compress(binary_data);
                ctx.binary(final_data);
            }));
        });
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("[WebSocket] Client disconnected");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                ctx.pong(&msg);
            }
            Ok(ws::Message::Text(text)) => {
                if let Ok(ping_msg) = serde_json::from_str::<PingMessage>(&text) {
                    let pong = self.handle_ping(ping_msg);
                    if let Ok(response) = serde_json::to_string(&pong) {
                        ctx.text(response);
                    }
                }
            }
            Ok(ws::Message::Binary(data)) => {
                // Try to decompress the data first
                match self.maybe_decompress(data.to_vec()) {
                    Ok(decompressed_data) => {
                        // Decode binary message
                        match binary_protocol::decode_node_data(&decompressed_data) {
                            Ok((msg_type, nodes)) => {
                                match msg_type {
                                    MessageType::PositionUpdate | MessageType::FullStateUpdate => {
                                        debug!("[WebSocket] Received {} node updates", nodes.len());
                                        // Here you could update the node positions in your state
                                        // For now we just acknowledge receipt
                                    },
                                    MessageType::VelocityUpdate => {
                                        debug!("[WebSocket] Received velocity updates for {} nodes", nodes.len());
                                        // Handle velocity updates if needed
                                    }
                                }
                            },
                            Err(e) => {
                                error!("[WebSocket] Failed to decode binary message: {}", e);
                            }
                        }
                    },
                    Err(e) => {
                        error!("[WebSocket] Failed to process binary message: {}", e);
                    }
                }
            }
            Ok(ws::Message::Close(reason)) => {
                info!("[WebSocket] Client disconnected: {:?}", reason);
                ctx.close(reason);
                ctx.stop();
            }
            _ => {}
        }
    }
}

pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
    settings: web::Data<Arc<RwLock<crate::config::Settings>>>,
) -> Result<HttpResponse, Error> {
    // Check for WebSocket upgrade headers
    let upgrade = req.headers().get("upgrade")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_lowercase());

    let connection = req.headers().get("connection")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_lowercase());

    match (upgrade, connection) {
        (Some(upgrade), Some(connection)) if upgrade == "websocket" && connection.contains("upgrade") => {
            // Valid WebSocket upgrade request
            debug!("Valid WebSocket upgrade request received");
        }
        _ => {
            debug!("Attempted WebSocket connection without proper upgrade headers");
            debug!("Upgrade header: {:?}", req.headers().get("upgrade"));
            debug!("Connection header: {:?}", req.headers().get("connection"));
            return Ok(HttpResponse::BadRequest()
                .reason("WebSocket upgrade headers required")
                .finish());
        }
    }

    let ws = SocketFlowServer::new(
        app_state.into_inner(),
        settings.get_ref().clone()
    );
    
    match ws::start(ws, &req, stream) {
        Ok(response) => {
            info!("[WebSocket] Client connected");
            Ok(response)
        }
        Err(e) => {
            warn!("WebSocket connection failed: {}", e);
            Err(e)
        }
    }
}

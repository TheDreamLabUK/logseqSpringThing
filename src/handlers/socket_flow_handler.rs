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
        info!("[WebSocket] Client connected from {:?}", ctx.address());
        
        // Send initial connection success message
        let init_msg = serde_json::json!({
            "type": "connection_established",
            "timestamp": chrono::Utc::now().timestamp_millis()
        });
        
        if let Ok(msg) = serde_json::to_string(&init_msg) {
            ctx.text(msg);
        }
        
        // Clone Arc references for the interval closure
        let app_state = self.app_state.clone();
        let settings = self.settings.clone();
        
        // Start position update interval
        ctx.run_interval(self.update_interval, move |actor, ctx| {
            // Get current node positions and velocities
            let app_state_clone = app_state.clone();
            let _settings_clone = settings.clone();
            
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
                
                // Only send update if there are nodes
                if !nodes.is_empty() {
                    // Encode using binary protocol
                    binary_protocol::encode_node_data(&nodes, MessageType::FullStateUpdate)
                } else {
                    Vec::new()
                }
            };
            
            // Convert the future to an actix future and handle it
            let fut = fut.into_actor(actor);
            ctx.spawn(fut.map(|binary_data, actor, ctx| {
                if !binary_data.is_empty() {
                    // Compress if enabled and threshold met
                    let final_data = actor.maybe_compress(binary_data);
                    ctx.binary(final_data);
                }
            }));
        });
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("[WebSocket] Client disconnected");
        // Cleanup any resources if needed
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                debug!("[WebSocket] Received ping");
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                debug!("[WebSocket] Received pong");
                // Update last pong time if needed
            }
            Ok(ws::Message::Text(text)) => {
                debug!("[WebSocket] Received text message: {}", text);
                match serde_json::from_str(&text) {
                    Ok(ping_msg @ PingMessage { .. }) => {
                        let pong = self.handle_ping(ping_msg);
                        if let Ok(response) = serde_json::to_string(&pong) {
                            ctx.text(response);
                        }
                    }
                    Err(e) => {
                        warn!("[WebSocket] Failed to parse text message: {}", e);
                    }
                }
            }
            Ok(ws::Message::Binary(data)) => {
                debug!("[WebSocket] Received binary message of size: {}", data.len());
                
                // Try to decompress the data first
                match self.maybe_decompress(data.to_vec()) {
                    Ok(decompressed_data) => {
                        // Decode binary message
                        match binary_protocol::decode_node_data(&decompressed_data) {
                            Ok((msg_type, nodes)) => {
                                match msg_type {
                                    MessageType::PositionUpdate | MessageType::FullStateUpdate => {
                                        debug!("[WebSocket] Received {} node updates", nodes.len());
                                        // Acknowledge receipt with a small response
                                        let ack = serde_json::json!({
                                            "type": "update_received",
                                            "count": nodes.len(),
                                            "timestamp": chrono::Utc::now().timestamp_millis()
                                        });
                                        if let Ok(response) = serde_json::to_string(&ack) {
                                            ctx.text(response);
                                        }
                                    },
                                    MessageType::VelocityUpdate => {
                                        debug!("[WebSocket] Received velocity updates for {} nodes", nodes.len());
                                    }
                                }
                            },
                            Err(e) => {
                                error!("[WebSocket] Failed to decode binary message: {}", e);
                                // Send error message back to client
                                let error_msg = serde_json::json!({
                                    "type": "error",
                                    "message": format!("Failed to decode binary message: {}", e)
                                });
                                if let Ok(response) = serde_json::to_string(&error_msg) {
                                    ctx.text(response);
                                }
                            }
                        }
                    },
                    Err(e) => {
                        error!("[WebSocket] Failed to process binary message: {}", e);
                        // Send error message back to client
                        let error_msg = serde_json::json!({
                            "type": "error",
                            "message": format!("Failed to process binary message: {}", e)
                        });
                        if let Ok(response) = serde_json::to_string(&error_msg) {
                            ctx.text(response);
                        }
                    }
                }
            }
            Ok(ws::Message::Close(reason)) => {
                info!("[WebSocket] Client initiated close: {:?}", reason);
                ctx.close(reason);
                ctx.stop();
            }
            Ok(ws::Message::Continuation(_)) => {
                warn!("[WebSocket] Received unexpected continuation frame");
            }
            Ok(ws::Message::Nop) => {
                debug!("[WebSocket] Received Nop");
            }
            Err(e) => {
                error!("[WebSocket] Error in WebSocket connection: {}", e);
                ctx.stop();
            }
        }
    }
}

pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
    settings: web::Data<Arc<RwLock<crate::config::Settings>>>,
) -> Result<HttpResponse, Error> {
    // Log request details for debugging
    debug!("WebSocket connection attempt from {:?}", req.peer_addr());
    debug!("Request URI: {}", req.uri());
    debug!("Headers:");
    for (name, value) in req.headers() {
        debug!("  {}: {:?}", name, value);
    }

    // Check for WebSocket upgrade - accept any case for headers
    let upgrade_header = req.headers().get("upgrade")
        .or_else(|| req.headers().get("Upgrade"))
        .or_else(|| req.headers().get("UPGRADE"));

    let connection_header = req.headers().get("connection")
        .or_else(|| req.headers().get("Connection"))
        .or_else(|| req.headers().get("CONNECTION"));

    let is_websocket = upgrade_header
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_lowercase().contains("websocket"))
        .unwrap_or(false);

    let has_connection = connection_header
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_lowercase().contains("upgrade"))
        .unwrap_or(false);

    if !is_websocket || !has_connection {
        warn!("Invalid WebSocket connection attempt");
        debug!("Upgrade header: {:?}", upgrade_header);
        debug!("Connection header: {:?}", connection_header);
        return Ok(HttpResponse::BadRequest()
            .append_header(("Content-Type", "text/plain"))
            .body("WebSocket upgrade required"));
    }

    info!("Valid WebSocket upgrade request received");

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

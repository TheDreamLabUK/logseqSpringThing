use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use flate2::{write::ZlibEncoder, Compression};
use log::{debug, error, info, warn};
use std::io::Write;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::app_state::AppState;
use crate::utils::binary_protocol;
use crate::utils::socket_flow_messages::{BinaryNodeData, PingMessage, PongMessage};

// Constants for throttling debug logs
const DEBUG_LOG_SAMPLE_RATE: usize = 10; // Only log 1 in 10 updates

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    settings: Arc<RwLock<crate::config::Settings>>,
    last_ping: Option<u64>,
    update_counter: usize, // Counter for throttling debug logs
    update_interval: std::time::Duration,
}

impl SocketFlowServer {
    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<crate::config::Settings>>) -> Self {
        // Calculate update interval from settings
        let update_rate = settings
            .try_read()
            .map(|s| s.system.websocket.binary_update_rate)
            .unwrap_or(30);

        let update_interval =
            std::time::Duration::from_millis((1000.0 / update_rate as f64) as u64);

        Self {
            app_state,
            settings,
            last_ping: None,
            update_counter: 0,
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
            if settings.system.websocket.compression_enabled
                && data.len() >= settings.system.websocket.compression_threshold
            {
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
    
    // Helper method to determine if we should log this update (for throttling)
    fn should_log_update(&mut self) -> bool {
        self.update_counter = (self.update_counter + 1) % DEBUG_LOG_SAMPLE_RATE;
        self.update_counter == 0
    }
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocket] Client connected successfully");

        // Send simple connection established message
        let response = serde_json::json!({
            "type": "connection_established",
            "timestamp": chrono::Utc::now().timestamp_millis()
        });

        if let Ok(msg_str) = serde_json::to_string(&response) {
            ctx.text(msg_str);
        }

        // Send a "loading" message to indicate the client should display a loading indicator
        let loading_msg = serde_json::json!({
            "type": "loading",
            "message": "Calculating initial layout..."
        });
        ctx.text(serde_json::to_string(&loading_msg).unwrap_or_default());
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("[WebSocket] Client disconnected");
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
            }
            Ok(ws::Message::Text(text)) => {
                info!("Received text message: {}", text);
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(msg) => {
                        match msg.get("type").and_then(|t| t.as_str()) {
                            Some("ping") => {
                                if let Ok(ping_msg) =
                                    serde_json::from_value::<PingMessage>(msg.clone())
                                {
                                    let pong = self.handle_ping(ping_msg);
                                    if let Ok(response) = serde_json::to_string(&pong) {
                                        ctx.text(response);
                                    }
                                }
                            }
                            Some("requestInitialData") => {
                                info!("Received request for position updates");

                                // No need to check for initial_data_sent, just handle the request
                                let app_state = self.app_state.clone();
                                
                                ctx.run_interval(self.update_interval, move |act: &mut SocketFlowServer, ctx| {
                                    let app_state_clone = app_state.clone();
                                    let settings_clone = act.settings.clone();

                                    let fut = async move {
                                        let raw_nodes = app_state_clone
                                            .graph_service
                                            .get_node_positions()
                                            .await;

                                        let node_count = raw_nodes.len();
                                        if node_count == 0 {
                                            debug!("[WebSocket] No nodes to send! Empty graph data."); return None;
                                        }

                                        // Check if detailed debugging should be enabled
                                        let detailed_debug = if let Ok(settings) = settings_clone.try_read() {
                                            settings.system.debug.enabled && 
                                            settings.system.debug.enable_websocket_debug
                                        } else {
                                            false
                                        };

                                        if detailed_debug {
                                            debug!("Raw nodes count: {}, showing first 5 nodes IDs:", raw_nodes.len());
                                            for (i, node) in raw_nodes.iter().take(5).enumerate() {
                                                debug!("  Node {}: id={} (numeric), metadata_id={} (filename)", 
                                                    i, node.id, node.metadata_id);
                                            }
                                        }

                                        let mut nodes = Vec::with_capacity(raw_nodes.len());
                                        for node in raw_nodes {
                                            if let Ok(node_id) = node.id.parse::<u32>() {
                                                // The node.id should now be numeric, so parsing should always succeed
                                                nodes.push((node_id, BinaryNodeData {
                                                    position: node.data.position,
                                                    velocity: node.data.velocity,
                                                    mass: node.data.mass,
                                                    flags: node.data.flags,
                                                    padding: node.data.padding,
                                                }));
                                            } else {
                                                warn!("[WebSocket] Failed to parse node ID as u32: '{}', metadata_id: '{}'", 
                                                    node.id, node.metadata_id);
                                            }
                                        }

                                        // Only generate binary data if we have nodes to send
                                        if nodes.is_empty() {
                                            return None;

                                        }
                                       
                                        let data = binary_protocol::encode_node_data(&nodes);
                                        
                                        // Return detailed debug info along with the data
                                        Some((data, detailed_debug, nodes))
                                    };

                                    // First check if we should log this update (before spawning the future)
                                    let should_log = act.should_log_update();
                                    
                                    let fut = fut.into_actor(act);
                                    ctx.spawn(fut.map(move |result, act, ctx| {
                                        if let Some((binary_data, detailed_debug, nodes)) = result {
                                            // Log debug info if needed
                                            if detailed_debug && should_log {
                                                debug!("[WebSocket] Encoded binary data: {} bytes for {} nodes", binary_data.len(), nodes.len());
                                                
                                                // Log details about a sample node to track position changes
                                                if !nodes.is_empty() {
                                                    let node = &nodes[0];
                                                    debug!(
                                                        "Sample node: id={}, pos=[{:.2},{:.2},{:.2}], vel=[{:.2},{:.2},{:.2}]",
                                                        node.0,
                                                        node.1.position[0], node.1.position[1], node.1.position[2],
                                                        node.1.velocity[0], node.1.velocity[1], node.1.velocity[2]
                                                    );
                                                }
                                            }

                                            let final_data = act.maybe_compress(binary_data);
                                            
                                            // Only log if detailed debugging is enabled
                                            if detailed_debug && should_log {
                                                    debug!("[WebSocket] Final binary update sent to client: {} bytes", final_data.len());
                                            }
                                            
                                            ctx.binary(final_data);
                                        }
                                    }));
                                });

                                let response = serde_json::json!({
                                    "type": "updatesStarted",
                                    "timestamp": chrono::Utc::now().timestamp_millis()
                                });
                                if let Ok(msg_str) = serde_json::to_string(&response) {
                                    ctx.text(msg_str);
                                }
                            }
                            Some("enableRandomization") => {
                                if let Ok(enable_msg) = serde_json::from_value::<serde_json::Value>(msg.clone()) {
                                    let enabled = enable_msg.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false);
                                    info!("Client requested to {} node position randomization (server-side randomization removed)", 
                                         if enabled { "enable" } else { "disable" });
                                    
                                    // Server-side randomization has been removed, but we still acknowledge the client's request
                                    // to maintain backward compatibility with existing clients
                                    actix::spawn(async move {
                                        // Log that we received the request but server-side randomization is no longer supported
                                        info!("Node position randomization request acknowledged, but server-side randomization is no longer supported");
                                        info!("Client-side randomization is now used instead");
                                    });
                                }
                            }
                            _ => {
                                warn!("[WebSocket] Unknown message type: {:?}", msg);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("[WebSocket] Failed to parse text message: {}", e);
                        let error_msg = serde_json::json!({
                            "type": "error",
                            "message": format!("Failed to parse text message: {}", e)
                        });
                        if let Ok(msg_str) = serde_json::to_string(&error_msg) {
                            ctx.text(msg_str);
                        }
                    }
                }
            }
            Ok(ws::Message::Binary(data)) => {
                info!("Received binary message, length: {}", data.len());
                
                // Enhanced logging for binary messages
                if data.len() % 28 != 0 {
                    warn!(
                        "Binary message size mismatch: {} bytes (not a multiple of 28, remainder: {})",
                        data.len(),
                        data.len() % 28
                    );
                }
                
                match binary_protocol::decode_node_data(&data) {
                    Ok(nodes) => {
                        if nodes.len() <= 2 {
                            let app_state = self.app_state.clone();
                            let nodes_vec: Vec<_> = nodes.into_iter().collect();

                            let fut = async move {
                                let mut graph = app_state.graph_service.get_graph_data_mut().await;
                                let mut node_map = app_state.graph_service.get_node_map_mut().await;

                                for (node_id, node_data) in nodes_vec {
                                    let node_id_str = node_id.to_string();
                                    if let Some(node) = node_map.get_mut(&node_id_str) {
                                        // Node exists with this numeric ID
                                        // Explicitly preserve existing mass and flags
                                        let original_mass = node.data.mass;
                                        let original_flags = node.data.flags;
                                        
                                        node.data.position = node_data.position;
                                        node.data.velocity = node_data.velocity;
                                        // Explicitly restore mass and flags after updating position/velocity
                                        node.data.mass = original_mass;
                                        node.data.flags = original_flags; // Restore flags needed for GPU code
                                    // Mass, flags, and padding are not overwritten as they're only 
                                    // present on the server side and not transmitted over the wire
                                    } else {
                                        debug!("Received update for unknown node ID: {}", node_id_str);
                                    }
                                }
                                
                                // Add more detailed debug information for mass maintenance
                                debug!("Updated node positions from binary data (preserving server-side properties)");

                                // Update graph nodes with new positions/velocities from the map, preserving other properties
                                for node in &mut graph.nodes {
                                    if let Some(updated_node) = node_map.get(&node.id) {
                                        // Explicitly preserve mass and flags before updating
                                        let original_mass = node.data.mass;
                                        let original_flags = node.data.flags;
                                        node.data.position = updated_node.data.position;
                                        node.data.velocity = updated_node.data.velocity;
                                        node.data.mass = original_mass; // Restore mass after updating
                                        node.data.flags = original_flags; // Restore flags after updating
                                    }
                                }
                            };

                            let fut = fut.into_actor(self);
                            ctx.spawn(fut.map(|_, _, _| ()));
                        } else {
                            warn!("Received update for too many nodes: {}", nodes.len());
                            let error_msg = serde_json::json!({
                                "type": "error",
                                "message": format!("Too many nodes in update: {}", nodes.len())
                            });
                            if let Ok(msg_str) = serde_json::to_string(&error_msg) {
                                ctx.text(msg_str);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to decode binary message: {}", e);
                        let error_msg = serde_json::json!({
                            "type": "error",
                            "message": format!("Failed to decode binary message: {}", e)
                        });
                        if let Ok(msg_str) = serde_json::to_string(&error_msg) {
                            ctx.text(msg_str);
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
    let should_debug = settings.try_read().map(|s| {
        s.system.debug.enabled && s.system.debug.enable_websocket_debug
    }).unwrap_or(false);

    if should_debug {
        debug!("WebSocket connection attempt from {:?}", req.peer_addr());
    }

    // Check for WebSocket upgrade
    if !req.headers().contains_key("Upgrade") {
        return Ok(HttpResponse::BadRequest().body("WebSocket upgrade required"));
    }

    let ws = SocketFlowServer::new(app_state.into_inner(), settings.get_ref().clone());

    match ws::start(ws, &req, stream) {
        Ok(response) => {
            info!("[WebSocket] Client connected successfully");
            Ok(response)
        }
        Err(e) => {
            error!("[WebSocket] Failed to start WebSocket: {}", e);
            Err(e)
        }
    }
}

use std::sync::Arc;
use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use tokio::sync::RwLock;
use log::{info, debug, error};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

use crate::app_state::AppState;
use crate::config::Settings;

// Constants matching client/state/graphData.ts
const NODE_POSITION_SIZE: usize = 24;  // 6 floats * 4 bytes
const FLOATS_PER_NODE: usize = 6;      // x, y, z, vx, vy, vz
const VERSION_HEADER_SIZE: usize = 4;
const NODE_DATA_SIZE: usize = 24;
const BINARY_PROTOCOL_VERSION: i32 = 1;
const MAX_MESSAGE_SIZE: usize = 1024 * 1024; // 1MB
const MAX_CONNECTIONS: usize = 100;
const HEARTBEAT_INTERVAL: u64 = 30;
const MAX_CLIENT_TIMEOUT: u64 = 60;

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketSettings {
    update_rate: u32,
}

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    settings: Arc<RwLock<Settings>>,
    connection_alive: bool,
    update_handle: Option<SpawnHandle>,
}

impl SocketFlowServer {
    const POSITION_UPDATE_INTERVAL: Duration = Duration::from_millis(16);

    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<Settings>>) -> Self {
        Self {
            app_state,
            settings,
            connection_alive: true,
            update_handle: None,
        }
    }

    fn start_position_updates(&mut self, ctx: &mut <Self as Actor>::Context) {
        // Cancel existing interval if any
        if let Some(handle) = self.update_handle.take() {
            ctx.cancel_future(handle);
        }

        // Clone Arc references for the interval closure
        let app_state = Arc::clone(&self.app_state);
        
        let handle = ctx.run_interval(Self::POSITION_UPDATE_INTERVAL, move |actor, ctx| {
            // Get current node positions and velocities
            let app_state_clone = Arc::clone(&app_state);
            
            // Spawn a future to get positions
            let fut = async move {
                let nodes = app_state_clone.graph_service.get_node_positions().await;
                
                // Create binary data: 24 bytes per node (no header)
                let mut binary_data = Vec::with_capacity(nodes.len() * NODE_POSITION_SIZE);
                
                for node in nodes {
                    // Position (x, y, z)
                    binary_data.extend_from_slice(&node.data.position[0].to_le_bytes());
                    binary_data.extend_from_slice(&node.data.position[1].to_le_bytes());
                    binary_data.extend_from_slice(&node.data.position[2].to_le_bytes());
                    // Velocity (vx, vy, vz)
                    binary_data.extend_from_slice(&node.data.velocity[0].to_le_bytes());
                    binary_data.extend_from_slice(&node.data.velocity[1].to_le_bytes());
                    binary_data.extend_from_slice(&node.data.velocity[2].to_le_bytes());
                }
                
                binary_data
            };
            
            ctx.spawn(fut.into_actor(actor).map(|binary_data, actor, ctx| {
                ctx.binary(binary_data);
            }));
        });

        self.update_handle = Some(handle);
    }
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocket] Client connected");
        self.app_state.increment_connections();
        let current = self.app_state.active_connections.load(std::sync::atomic::Ordering::Relaxed);
        info!("[WebSocket] Active connections: {}", current);
        self.start_position_updates(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("[WebSocket] Client disconnected");
        self.connection_alive = false;
        self.app_state.decrement_connections();
        let current = self.app_state.active_connections.load(std::sync::atomic::Ordering::Relaxed);
        info!("[WebSocket] Remaining active connections: {}", current);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Binary(bin)) => {
                if bin.len() > MAX_MESSAGE_SIZE {
                    error!("[WebSocket] Message too large: {} bytes", bin.len());
                    ctx.stop();
                    return;
                }
                
                // Handle binary node position updates from client
                if bin.len() % NODE_POSITION_SIZE != 0 {
                    error!("[WebSocket] Malformed binary message: length {} is invalid", bin.len());
                    return;
                }
                
                let num_nodes = bin.len() / NODE_POSITION_SIZE;
                let mut positions = Vec::with_capacity(num_nodes);
                
                for i in 0..num_nodes {
                    let mut position = [0.0; 3];
                    let mut velocity = [0.0; 3];
                    
                    // Read position and velocity components
                    for j in 0..3 {
                        let pos_bytes = [
                            bin[i * NODE_POSITION_SIZE + j * 4],
                            bin[i * NODE_POSITION_SIZE + j * 4 + 1],
                            bin[i * NODE_POSITION_SIZE + j * 4 + 2],
                            bin[i * NODE_POSITION_SIZE + j * 4 + 3],
                        ];
                        position[j] = f32::from_le_bytes(pos_bytes);
                        
                        let vel_bytes = [
                            bin[i * NODE_POSITION_SIZE + (j + 3) * 4],
                            bin[i * NODE_POSITION_SIZE + (j + 3) * 4 + 1],
                            bin[i * NODE_POSITION_SIZE + (j + 3) * 4 + 2],
                            bin[i * NODE_POSITION_SIZE + (j + 3) * 4 + 3],
                        ];
                        velocity[j] = f32::from_le_bytes(vel_bytes);
                    }
                    
                    positions.push((position, velocity));
                }
                
                debug!("[WebSocket] Successfully processed {} node position updates", positions.len());
            }
            Ok(ws::Message::Close(reason)) => {
                info!("[WebSocket] Client disconnected: {:?}", reason);
                ctx.stop();
            }
            Ok(ws::Message::Ping(msg)) => {
                debug!("[WebSocket] Received ping");
                ctx.pong(&msg);
            },
            Ok(ws::Message::Pong(_)) => {
                debug!("[WebSocket] Received pong");
            },
            Err(e) => {
                error!("[WebSocket] Protocol error: {}", e);
                ctx.stop();
            }
            _ => () // Ignore other message types
        }
    }
}

pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> Result<HttpResponse, Error> {
    // Enhanced connection debugging
    info!("[WebSocket] New connection request from {:?}", req.peer_addr());
    debug!("[WebSocket] Headers: {:?}", req.headers());
    debug!("[WebSocket] URI: {:?}", req.uri());
    
    // Check connection limits
    let current_connections = app_state.active_connections.load(std::sync::atomic::Ordering::Relaxed);
    if current_connections >= MAX_CONNECTIONS {
        error!("[WebSocket] Connection limit reached: {}/{}", current_connections, MAX_CONNECTIONS);
        return Ok(HttpResponse::ServiceUnavailable().json(json!({
            "error": "Connection limit reached"
        })));
    }

    // Create server instance
    let server = SocketFlowServer::new(
        app_state.into_inner(),
        settings.get_ref().clone()
    );

    // Start WebSocket connection
    info!("[WebSocket] Starting WebSocket connection");
    match ws::start(server, &req, stream) {
        Ok(response) => {
            info!("[WebSocket] WebSocket connection established successfully");
            Ok(response)
        }
        Err(e) => {
            error!("[WebSocket] Failed to start WebSocket connection: {:?}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to start WebSocket connection: {}", e)
            })))
        }
    }
}

pub async fn get_websocket_settings(
    settings: web::Data<Arc<RwLock<Settings>>>
) -> Result<HttpResponse, Error> {
    let settings = settings.read().await;
    let ws_settings = WebSocketSettings {
        update_rate: settings.system.websocket.update_rate,
    };
    
    Ok(HttpResponse::Ok().json(ws_settings))
}

const MAX_UPDATE_RATE: u32 = 120; // Maximum updates per second

pub async fn update_websocket_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    new_settings: web::Json<WebSocketSettings>
) -> Result<HttpResponse, Error> {
    // Validate update rate
    if new_settings.update_rate == 0 {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "Update rate must be greater than 0"
        })));
    }

    if new_settings.update_rate > MAX_UPDATE_RATE {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": format!("Update rate cannot exceed {} updates per second", MAX_UPDATE_RATE)
        })));
    }

    let mut settings = settings.write().await;
    settings.system.websocket.update_rate = new_settings.update_rate;
    debug!("[WebSocket] Updated update rate to: {}", new_settings.update_rate);
    
    Ok(HttpResponse::Ok().json(new_settings.0))
}

use std::sync::Arc;
use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use tokio::sync::RwLock;
use log::{info, debug, error};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Duration, Instant};

use crate::app_state::AppState;
use crate::config::Settings;

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketSettings {
    pub update_rate: u32,
}

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    settings: Arc<RwLock<Settings>>,
    connection_alive: bool,
    update_handle: Option<SpawnHandle>,
    heartbeat_handle: Option<SpawnHandle>,
    last_heartbeat: Instant,
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocket] Client connected");
        self.start_position_updates(ctx);
        self.start_heartbeat(ctx);
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocket] Client disconnected");
        
        // Cancel heartbeat
        if let Some(handle) = self.heartbeat_handle.take() {
            ctx.cancel_future(handle);
        }
    }
}

impl SocketFlowServer {
    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<Settings>>) -> Self {
        Self {
            app_state,
            settings,
            connection_alive: true,
            update_handle: None,
            heartbeat_handle: None,
            last_heartbeat: Instant::now(),
        }
    }

    fn start_heartbeat(&mut self, ctx: &mut <Self as Actor>::Context) {
        // Cancel existing heartbeat if any
        if let Some(handle) = self.heartbeat_handle.take() {
            ctx.cancel_future(handle);
        }

        // Start heartbeat interval
        let handle = ctx.run_interval(HEARTBEAT_INTERVAL, |actor, ctx| {
            if Instant::now().duration_since(actor.last_heartbeat) > HEARTBEAT_TIMEOUT {
                error!("[WebSocket] Client heartbeat timeout");
                actor.connection_alive = false;
                ctx.stop();
                return;
            }

            ctx.ping(b"");
        });

        self.heartbeat_handle = Some(handle);
    }

    fn start_position_updates(&mut self, ctx: &mut <Self as Actor>::Context) {
        // Cancel existing interval if any
        if let Some(handle) = self.update_handle.take() {
            ctx.cancel_future(handle);
        }

        // Clone Arc references for the interval closure
        let app_state = self.app_state.clone();
        let settings = self.settings.clone();

        // Spawn a future to get the current update rate
        let fut = async move {
            let settings = settings.read().await;
            settings.system.websocket.update_rate
        };

        // Convert to actix future and handle it
        let fut = fut.into_actor(self);
        ctx.spawn(fut.map(|update_rate, actor, ctx| {
            // Calculate update interval based on rate
            let update_interval = Duration::from_millis((1000.0 / update_rate as f64) as u64);
            
            // Set up new interval
            let handle = ctx.run_interval(update_interval, move |actor, ctx| {
                if !actor.connection_alive {
                    ctx.stop();
                    return;
                }

                // Get current node positions and velocities
                let app_state_clone = app_state.clone();
                
                // Spawn a future to get positions
                let fut = async move {
                    let nodes = app_state_clone.graph_service.get_node_positions().await;
                    
                    // Create binary data: 24 bytes per node (6 f32s)
                    let mut binary_data = Vec::with_capacity(nodes.len() * 24);
                    
                    for node in nodes {
                        // Position (x, y, z)
                        binary_data.extend_from_slice(&node.data.position[0].to_le_bytes());
                        binary_data.extend_from_slice(&node.data.position[1].to_le_bytes());
                        binary_data.extend_from_slice(&node.data.position[2].to_le_bytes());
                        
                        // Velocity (x, y, z)
                        binary_data.extend_from_slice(&node.data.velocity[0].to_le_bytes());
                        binary_data.extend_from_slice(&node.data.velocity[1].to_le_bytes());
                        binary_data.extend_from_slice(&node.data.velocity[2].to_le_bytes());
                    }
                    
                    binary_data
                };
                
                // Convert the future to an actix future and handle it
                let fut = fut.into_actor(actor);
                ctx.spawn(fut.map(|binary_data, _actor, ctx| {
                    ctx.binary(binary_data);
                }));
            });

            actor.update_handle = Some(handle);
        }));
    }
}

// Message for updating WebSocket settings
#[derive(Message, Debug)]
#[rtype(result = "()")]
pub struct WebSocketSettingsChanged;

// Implement marker traits for SocketFlowServer
impl Unpin for SocketFlowServer {}

// Make SocketFlowServer properly sized
impl actix::Supervised for SocketFlowServer {}

// Implement proper handler for WebSocket settings changes
impl Handler<WebSocketSettingsChanged> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, _: WebSocketSettingsChanged, ctx: &mut <Self as Actor>::Context) {
        debug!("Restarting position updates with new settings");
        self.start_position_updates(ctx);
    }
}

// Implement proper handler for WebSocket messages
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut <Self as Actor>::Context) {
        if !self.connection_alive {
            ctx.stop();
            return;
        }

        match msg {
            Ok(ws::Message::Binary(bin)) => {
                // Handle binary node updates from client
                if bin.len() % 12 != 0 {
                    error!("[WebSocket] Malformed binary message: length {} is not a multiple of 12 bytes", bin.len());
                    ctx.text(json!({
                        "error": "Malformed binary message: incorrect length"
                    }).to_string());
                    return;
                }

                let mut positions = Vec::with_capacity(bin.len() / 12);
                    for (i, chunk) in bin.chunks_exact(12).enumerate() {
                        let mut position = [0.0f32; 3];
                        for j in 0..3 {
                            let start = j * 4;
                            let end = start + 4;
                            
                            match chunk[start..end].try_into() {
                                Ok(bytes) => {
                                    position[j] = f32::from_le_bytes(bytes);
                                    
                                    // Validate the float value
                                    if !position[j].is_finite() {
                                        error!("[WebSocket] Invalid float value at position {}, component {}: {}", i, j, position[j]);
                                        ctx.text(json!({
                                            "error": format!("Invalid float value at position {}, component {}", i, j)
                                        }).to_string());
                                        return;
                                    }
                                },
                                Err(e) => {
                                    error!("[WebSocket] Failed to convert bytes to float at position {}, component {}: {}", i, j, e);
                                    ctx.text(json!({
                                        "error": format!("Failed to convert bytes to float at position {}, component {}", i, j)
                                    }).to_string());
                                    return;
                                }
                            }
                        }

                        positions.push(position);
                    }

                    debug!("Successfully processed {} node position updates", positions.len());
            }
            // Handle protocol-level close frames
            Ok(ws::Message::Close(reason)) => {
                info!("[WebSocket] Client disconnected: {:?}", reason);
                self.connection_alive = false;
                ctx.close(reason);
                ctx.stop();
            }
            // Handle ping frames
            Ok(ws::Message::Ping(msg)) => {
                debug!("[WebSocket] Received ping");
                ctx.pong(&msg);
            },
            // Handle pong frames (heartbeat response)
            Ok(ws::Message::Pong(_)) => {
                debug!("[WebSocket] Received pong");
                self.last_heartbeat = Instant::now();
            },
            // Handle protocol errors
            Err(e) => {
                error!("[WebSocket] Protocol error: {}", e);
                self.connection_alive = false;
                ctx.stop();
            }
            _ => ()
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
    debug!("Updated WebSocket update rate to: {}", new_settings.update_rate);

    // Notify all active WebSocket connections to update their intervals
    // This would be handled by your WebSocket connection manager if you have one
    // For now, we just return success
    
    Ok(HttpResponse::Ok().json(new_settings.0))
}

pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> Result<HttpResponse, Error> {
    let server = SocketFlowServer::new(app_state.into_inner(), settings.get_ref().clone());
    ws::start(server, &req, stream)
}

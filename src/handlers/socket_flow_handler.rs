use std::sync::Arc;
use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use tokio::sync::RwLock;
use log::{info, warn, debug, error};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::utils::socket_flow_constants::POSITION_UPDATE_RATE;
use crate::config::Settings;

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketSettings {
    pub update_rate: u32,
}

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    settings: Arc<RwLock<Settings>>,
    connection_alive: bool,
}

impl SocketFlowServer {
    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<Settings>>) -> Self {
        Self {
            app_state,
            settings,
            connection_alive: true,
        }
    }
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocket] Client connected");
        
        // Clone Arc references for the interval closure
        let app_state = self.app_state.clone();
        
        // Calculate update interval based on rate
        let update_interval = std::time::Duration::from_millis((1000.0 / POSITION_UPDATE_RATE as f64) as u64);
        
        ctx.run_interval(update_interval, move |actor, ctx| {
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
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("[WebSocket] Client disconnected");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        if !self.connection_alive {
            ctx.stop();
            return;
        }

        match msg {
            Ok(ws::Message::Binary(bin)) => {
                // Handle binary node updates from client
                if bin.len() % 12 == 0 {  // Each position is 3 f32s = 12 bytes
                    let positions = bin.chunks_exact(12).map(|chunk| {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(&chunk[0..4]);
                        let x = f32::from_le_bytes(bytes);
                        bytes.copy_from_slice(&chunk[4..8]);
                        let y = f32::from_le_bytes(bytes);
                        bytes.copy_from_slice(&chunk[8..12]);
                        let z = f32::from_le_bytes(bytes);
                        [x, y, z]
                    }).collect::<Vec<_>>();
                    
                    debug!("Received {} node position updates", positions.len());
                } else {
                    warn!("[WebSocket] Received malformed binary message");
                }
            }
            Ok(ws::Message::Close(reason)) => {
                info!("[WebSocket] Client disconnected: {:?}", reason);
                self.connection_alive = false;
                ctx.close(reason);
                ctx.stop();
            }
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Pong(_)) => (),
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

pub async fn update_websocket_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    new_settings: web::Json<WebSocketSettings>
) -> Result<HttpResponse, Error> {
    let mut settings = settings.write().await;
    settings.system.websocket.update_rate = new_settings.update_rate;
    debug!("Updated WebSocket update rate to: {}", new_settings.update_rate);
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

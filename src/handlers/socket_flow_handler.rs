use std::sync::Arc;
use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use tokio::sync::RwLock;
use log::{info, warn, debug};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::utils::socket_flow_messages::{PingMessage, PongMessage};
use crate::config::Settings;

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketSettings {
    pub heartbeat_interval: u64,
    pub heartbeat_timeout: u64,
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub update_rate: u32,
}

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    settings: Arc<RwLock<Settings>>,
    last_ping: Option<u64>,
}

impl SocketFlowServer {
    const POSITION_UPDATE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<Settings>>) -> Self {
        Self {
            app_state,
            settings,
            last_ping: None,
        }
    }

    fn handle_ping(&mut self, msg: PingMessage) -> PongMessage {
        self.last_ping = Some(msg.timestamp);
        PongMessage {
            type_: "pong".to_string(),
            timestamp: msg.timestamp,
        }
    }

    pub async fn get_settings(&self) -> WebSocketSettings {
        let settings = self.settings.read().await;
        WebSocketSettings {
            heartbeat_interval: settings.websocket.heartbeat_interval,
            heartbeat_timeout: settings.websocket.heartbeat_timeout,
            reconnect_attempts: settings.websocket.reconnect_attempts,
            reconnect_delay: settings.websocket.reconnect_delay,
            update_rate: settings.websocket.update_rate,
        }
    }
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocket] Client connected");
        
        // Clone Arc references for the interval closure
        let app_state = self.app_state.clone();
        
        ctx.run_interval(Self::POSITION_UPDATE_INTERVAL, move |_actor, ctx| {
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
            let fut = fut.into_actor(_actor);
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
            Ok(ws::Message::Binary(_)) => {
                warn!("[WebSocket] Unexpected binary message");
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

pub async fn get_websocket_settings(
    settings: web::Data<Arc<RwLock<Settings>>>
) -> Result<HttpResponse, Error> {
    let settings = settings.read().await;
    let ws_settings = WebSocketSettings {
        heartbeat_interval: settings.websocket.heartbeat_interval,
        heartbeat_timeout: settings.websocket.heartbeat_timeout,
        reconnect_attempts: settings.websocket.reconnect_attempts,
        reconnect_delay: settings.websocket.reconnect_delay,
        update_rate: settings.websocket.update_rate,
    };
    
    Ok(HttpResponse::Ok().json(ws_settings))
}

pub async fn update_websocket_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    new_settings: web::Json<WebSocketSettings>
) -> Result<HttpResponse, Error> {
    let mut settings = settings.write().await;
    
    settings.websocket.heartbeat_interval = new_settings.heartbeat_interval;
    settings.websocket.heartbeat_timeout = new_settings.heartbeat_timeout;
    settings.websocket.reconnect_attempts = new_settings.reconnect_attempts;
    settings.websocket.reconnect_delay = new_settings.reconnect_delay;
    settings.websocket.update_rate = new_settings.update_rate;
    
    debug!("Updated WebSocket settings: {:?}", new_settings);
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

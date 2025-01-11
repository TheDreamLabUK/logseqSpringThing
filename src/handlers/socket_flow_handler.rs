use std::sync::Arc;
use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use tokio::sync::RwLock;
use log::{debug, info, warn};

use crate::app_state::AppState;
use crate::utils::socket_flow_messages::{PingMessage, PongMessage};

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    #[allow(dead_code)]
    settings: Arc<RwLock<crate::config::Settings>>,
    last_ping: Option<u64>,
}

impl SocketFlowServer {
    const POSITION_UPDATE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<crate::config::Settings>>) -> Self {
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
            Ok(ws::Message::Binary(data)) => {
                // Handle incoming binary position updates from client
                // Each node is 24 bytes: [x,y,z, vx,vy,vz] as f32
                if data.len() % 24 == 0 {
                    // Process binary data if needed
                    // Currently we don't handle incoming position updates
                    warn!("[WebSocket] Received binary position update");
                } else {
                    warn!("[WebSocket] Invalid binary message length");
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
    if !req.headers().contains_key("upgrade") || 
       !req.headers().contains_key("sec-websocket-key") {
        debug!("Attempted WebSocket connection without proper upgrade headers");
        return Ok(HttpResponse::BadRequest()
            .reason("WebSocket upgrade required")
            .finish());
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

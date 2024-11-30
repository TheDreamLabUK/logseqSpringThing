use actix::prelude::*;
use actix_web_actors::ws;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use bytemuck;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{debug, error, info, warn};
use std::time::{Duration, Instant};
use serde_json;

use crate::models::node::GPUNode;
use crate::models::graph::GraphData;
use crate::utils::websocket_messages::{SendBinary, SendText, ServerMessage};

// Constants for binary protocol
const FLOAT32_SIZE: usize = std::mem::size_of::<f32>();
const HEADER_SIZE: usize = FLOAT32_SIZE; // isInitialLayout flag
const NODE_SIZE: usize = 6 * FLOAT32_SIZE; // x, y, z, vx, vy, vz

// Constants for heartbeat
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15); // Send ping every 15 seconds
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(60); // Wait 60 seconds for pong response

pub struct WebSocketManager {
    binary_buffer: Arc<Mutex<Vec<u8>>>,
    connections: Arc<Mutex<Vec<Addr<WebSocketSession>>>>,
    addr: Option<Addr<Self>>,
}

impl Actor for WebSocketManager {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.addr = Some(ctx.address());
        info!("[WebSocketManager] Actor started");
    }
}

impl WebSocketManager {
    pub fn new() -> Self {
        info!("[WebSocketManager] Creating new instance");
        Self {
            binary_buffer: Arc::new(Mutex::new(Vec::with_capacity(1024 * 1024))), // 1MB initial capacity
            connections: Arc::new(Mutex::new(Vec::new())),
            addr: None,
        }
    }

    pub fn start(mut self) -> Addr<Self> {
        info!("[WebSocketManager] Starting actor");
        let addr = Actor::start(self.clone());
        self.addr = Some(addr.clone());
        addr
    }

    pub fn get_addr(&self) -> Option<Addr<Self>> {
        self.addr.clone()
    }

    pub async fn add_connection(&self, addr: Addr<WebSocketSession>) {
        let mut connections = self.connections.lock().await;
        connections.push(addr);
        info!("[WebSocketManager] New WebSocket connection added. Total connections: {}", connections.len());
    }

    pub async fn remove_connection(&self, addr: &Addr<WebSocketSession>) {
        let mut connections = self.connections.lock().await;
        let before_len = connections.len();
        connections.retain(|x| x != addr);
        info!("[WebSocketManager] WebSocket connection removed. Connections: {} -> {}", before_len, connections.len());
    }

    pub async fn broadcast_binary(&self, nodes: &[GPUNode], is_initial: bool) -> Result<(), Box<dyn std::error::Error>> {
        debug!("[WebSocketManager] Broadcasting binary update for {} nodes", nodes.len());
        let mut buffer = self.binary_buffer.lock().await;
        let total_size = HEADER_SIZE + nodes.len() * NODE_SIZE;
        
        // Create a new buffer with the required capacity
        let mut new_buffer = Vec::with_capacity(total_size);
        
        // Write initial flag as float32
        let initial_flag: f32 = if is_initial { 1.0 } else { 0.0 };
        new_buffer.extend_from_slice(bytemuck::bytes_of(&initial_flag));

        // Write node data directly
        for node in nodes {
            let node_data: [f32; 6] = [
                node.x, node.y, node.z,
                node.vx, node.vy, node.vz
            ];
            new_buffer.extend_from_slice(bytemuck::cast_slice(&node_data));
        }

        // Replace the buffer content
        *buffer = new_buffer;

        // Broadcast to all connections
        let binary_data = buffer.clone();
        let connections = self.connections.lock().await;
        debug!("[WebSocketManager] Broadcasting binary data to {} connections", connections.len());
        for addr in connections.iter() {
            addr.do_send(SendBinary(binary_data.clone()));
        }

        Ok(())
    }

    pub async fn broadcast_message(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        let connections = self.connections.lock().await;
        debug!("[WebSocketManager] Broadcasting message to {} connections", connections.len());
        for addr in connections.iter() {
            addr.do_send(SendText(message.to_string()));
        }
        Ok(())
    }

    pub async fn broadcast_graph_update(&self, graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
        info!("[WebSocketManager] Broadcasting graph update with {} nodes and {} edges", 
            graph.nodes.len(), graph.edges.len());

        // Create message using ServerMessage enum
        let message = ServerMessage::GraphUpdate {
            graph_data: serde_json::to_value(graph)?
        };

        // Serialize to string and broadcast
        let message_str = serde_json::to_string(&message)?;
        debug!("[WebSocketManager] Graph update message size: {} bytes", message_str.len());
        
        // Get connections and broadcast
        let connections = self.connections.lock().await;
        debug!("[WebSocketManager] Broadcasting to {} connections", connections.len());
        for addr in connections.iter() {
            addr.do_send(SendText(message_str.clone()));
        }
        
        Ok(())
    }

    pub async fn handle_websocket(
        req: HttpRequest,
        stream: web::Payload,
        websocket_manager: web::Data<Arc<WebSocketManager>>,
    ) -> Result<HttpResponse, Error> {
        info!("[WebSocketManager] New websocket connection request from {:?}", 
            req.peer_addr().unwrap_or_else(|| std::net::SocketAddr::from(([0, 0, 0, 0], 0))));
        
        let ws = WebSocketSession::new(Arc::clone(&websocket_manager));
        ws::start(ws, &req, stream)
    }
}

impl Clone for WebSocketManager {
    fn clone(&self) -> Self {
        Self {
            binary_buffer: self.binary_buffer.clone(),
            connections: self.connections.clone(),
            addr: self.addr.clone(),
        }
    }
}

// WebSocket session actor
pub struct WebSocketSession {
    manager: Arc<WebSocketManager>,
    hb: Instant,
    last_pong: Instant,
}

impl WebSocketSession {
    pub fn new(manager: Arc<WebSocketManager>) -> Self {
        info!("[WebSocketSession] Creating new session");
        Self {
            manager,
            hb: Instant::now(),
            last_pong: Instant::now(),
        }
    }

    fn start_heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        debug!("[WebSocketSession] Starting heartbeat checks");
        ctx.run_interval(HEARTBEAT_INTERVAL, |act, ctx| {
            // Check client heartbeat
            if Instant::now().duration_since(act.last_pong) > HEARTBEAT_TIMEOUT {
                warn!("[WebSocketSession] Client heartbeat failed, disconnecting!");
                ctx.stop();
                return;
            }

            debug!("[WebSocketSession] Sending ping");
            ctx.ping(b"");
        });
    }
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocketSession] Session started");
        self.start_heartbeat(ctx);
        
        // Add connection to manager
        let addr = ctx.address();
        let manager = self.manager.clone();
        actix::spawn(async move {
            manager.add_connection(addr).await;
        });
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        info!("[WebSocketSession] Session stopped");
        
        // Remove connection from manager
        let addr = ctx.address();
        let manager = self.manager.clone();
        actix::spawn(async move {
            manager.remove_connection(&addr).await;
        });
    }
}

impl Handler<SendBinary> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendBinary, ctx: &mut Self::Context) {
        debug!("[WebSocketSession] Sending binary message of size {}", msg.0.len());
        ctx.binary(msg.0);
    }
}

impl Handler<SendText> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        debug!("[WebSocketSession] Sending text message: {}", msg.0);
        ctx.text(msg.0);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                debug!("[WebSocketSession] Ping received");
                self.hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                debug!("[WebSocketSession] Pong received");
                self.last_pong = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                debug!("[WebSocketSession] Text message received: {}", text);
                if text.contains("\"type\":\"ping\"") {
                    ctx.text("{\"type\":\"pong\"}");
                } else {
                    ctx.text(text);
                }
            }
            Ok(ws::Message::Binary(bin)) => {
                // Direct binary position/velocity updates
                if bin.len() >= HEADER_SIZE {
                    let mut header_bytes = [0u8; 4];
                    header_bytes.copy_from_slice(&bin[0..4]);
                    let is_initial = f32::from_le_bytes(header_bytes) >= 1.0;

                    let num_nodes = (bin.len() - HEADER_SIZE) / NODE_SIZE;
                    debug!("[WebSocketSession] Received binary update: {} nodes, initial={}", num_nodes, is_initial);

                    // Forward binary data to other clients
                    let connections = self.manager.connections.clone();
                    let bin_data = bin.to_vec();
                    actix::spawn(async move {
                        let connections = connections.lock().await;
                        for addr in connections.iter() {
                            addr.do_send(SendBinary(bin_data.clone()));
                        }
                    });
                }
            }
            Ok(ws::Message::Close(reason)) => {
                info!("[WebSocketSession] Close message received: {:?}", reason);
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

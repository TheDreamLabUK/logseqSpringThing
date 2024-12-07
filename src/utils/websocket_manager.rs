use actix::prelude::*;
use actix_web_actors::ws;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use bytemuck;
use std::sync::Arc;
use log::{debug, warn, trace, error};
use std::time::{Duration, Instant};
use serde_json::{self, Value};
use std::sync::Mutex;
use std::error::Error as StdError;

use crate::models::node::GPUNode;
use crate::models::graph::GraphData;
use crate::utils::websocket_messages::{SendBinary, SendText, ServerMessage};
use crate::AppState;

// Constants for binary protocol
pub(crate) const FLOAT32_SIZE: usize = std::mem::size_of::<f32>();
pub(crate) const HEADER_SIZE: usize = FLOAT32_SIZE; // isInitialLayout flag
pub(crate) const NODE_SIZE: usize = 6 * FLOAT32_SIZE; // x, y, z, vx, vy, vz

// Constants for heartbeat
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15); // Send ping every 15 seconds
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(60); // Wait 60 seconds for pong response

pub struct WebSocketManager {
    binary_buffer: Arc<Mutex<Vec<u8>>>,
    connections: Arc<Mutex<Vec<Addr<WebSocketSession>>>>,
    addr: Option<Addr<Self>>,
    debug_mode: bool,
    app_state: web::Data<AppState>,
}

impl Actor for WebSocketManager {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.addr = Some(ctx.address());
    }
}

impl WebSocketManager {
    pub fn new(debug_mode: bool, app_state: web::Data<AppState>) -> Self {
        if debug_mode {
            debug!("[WebSocketManager] Creating new instance with debug_mode={}", debug_mode);
        }
        Self {
            binary_buffer: Arc::new(Mutex::new(Vec::with_capacity(1024 * 1024))), // 1MB initial capacity
            connections: Arc::new(Mutex::new(Vec::new())),
            addr: None,
            debug_mode,
            app_state,
        }
    }

    pub fn start(mut self) -> Addr<Self> {
        let addr = Actor::start(self.clone());
        self.addr = Some(addr.clone());
        addr
    }

    pub fn get_addr(&self) -> Option<Addr<Self>> {
        self.addr.clone()
    }

    pub async fn add_connection(&self, addr: Addr<WebSocketSession>) {
        let mut connections = self.connections.lock().unwrap();
        connections.push(addr);
        if self.debug_mode {
            debug!("[WebSocketManager] New connection added. Total connections: {}", connections.len());
        }
    }

    pub async fn remove_connection(&self, addr: &Addr<WebSocketSession>) {
        let mut connections = self.connections.lock().unwrap();
        let before_len = connections.len();
        connections.retain(|x| x != addr);
        if self.debug_mode && before_len != connections.len() {
            debug!("[WebSocketManager] Connection removed. Total connections: {}", connections.len());
        }
    }

    pub async fn broadcast_message(&self, message: String) -> Result<(), Box<dyn StdError>> {
        let connections = self.connections.lock().map_err(|e| {
            error!("[WebSocketManager] Failed to lock connections mutex: {}", e);
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn StdError>
        })?;
        
        for addr in connections.iter() {
            addr.do_send(SendText(message.clone()));
        }
        Ok(())
    }

    pub async fn broadcast_binary(&self, nodes: &[GPUNode], is_initial: bool) -> Result<(), Box<dyn StdError>> {
        if self.debug_mode {
            debug!("[WebSocketManager] Broadcasting binary update for {} nodes, is_initial={}", nodes.len(), is_initial);
        }

        let total_size = HEADER_SIZE + nodes.len() * NODE_SIZE;
        let mut new_buffer = Vec::with_capacity(total_size);
        
        // Add initial layout flag
        let initial_flag: f32 = if is_initial { 1.0 } else { 0.0 };
        new_buffer.extend_from_slice(bytemuck::bytes_of(&initial_flag));

        // Add node data
        for node in nodes.iter() {
            let node_data: [f32; 6] = [
                node.x, node.y, node.z,
                node.vx, node.vy, node.vz
            ];
            new_buffer.extend_from_slice(bytemuck::cast_slice(&node_data));
        }

        if self.debug_mode {
            trace!("[WebSocketManager] Binary message size: {} bytes", new_buffer.len());
            if !new_buffer.is_empty() {
                let hex_dump: String = new_buffer.iter()
                    .take(32)
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                trace!("[WebSocketManager] Binary header (first 32 bytes): {}", hex_dump);
            }
        }

        // Update buffer and broadcast
        {
            let mut buffer = self.binary_buffer.lock().map_err(|e| {
                error!("[WebSocketManager] Failed to lock binary buffer mutex: {}", e);
                Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn StdError>
            })?;
            *buffer = new_buffer.clone();
        }

        let connections = self.connections.lock().map_err(|e| {
            error!("[WebSocketManager] Failed to lock connections mutex: {}", e);
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn StdError>
        })?;
        
        for addr in connections.iter() {
            addr.do_send(SendBinary(new_buffer.clone()));
        }

        Ok(())
    }

    pub async fn broadcast_graph_update(&self, graph: &GraphData) -> Result<(), Box<dyn StdError>> {
        if self.debug_mode {
            debug!("[WebSocketManager] Broadcasting graph update with {} nodes and {} edges", 
                graph.nodes.len(), graph.edges.len());
            
            for (i, node) in graph.nodes.iter().take(3).enumerate() {
                trace!("[WebSocketManager] Sample node {}: id={}", i, node.id);
            }
            for (i, edge) in graph.edges.iter().take(3).enumerate() {
                trace!("[WebSocketManager] Sample edge {}: source={}, target={}", 
                    i, edge.source, edge.target);
            }
        }

        let message = ServerMessage::GraphUpdate {
            graph_data: graph.clone()
        };

        let message_str = serde_json::to_string(&message).map_err(|e| {
            error!("[WebSocketManager] Failed to serialize graph update: {}", e);
            Box::new(e) as Box<dyn StdError>
        })?;
        
        let connections = self.connections.lock().map_err(|e| {
            error!("[WebSocketManager] Failed to lock connections mutex: {}", e);
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn StdError>
        })?;
        
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
        let ws = WebSocketSession::new(Arc::clone(&websocket_manager));
        ws::start(ws, &req, stream)
    }

    pub async fn send_initial_data(&self, addr: &Addr<WebSocketSession>) -> Result<(), Box<dyn StdError>> {
        if self.debug_mode {
            debug!("[WebSocketManager] Starting send_initial_data");
        }
        
        let graph = self.app_state.graph_service.graph_data.read().await;
        let settings = self.app_state.settings.read().await;

        let initial_data = ServerMessage::InitialData {
            graph_data: (*graph).clone(),
            settings: serde_json::to_value(&*settings).map_err(|e| {
                error!("[WebSocketManager] Failed to serialize settings: {}", e);
                Box::new(e) as Box<dyn StdError>
            })?,
        };

        let initial_data_str = serde_json::to_string(&initial_data).map_err(|e| {
            error!("[WebSocketManager] Failed to serialize initial data: {}", e);
            Box::new(e) as Box<dyn StdError>
        })?;

        if self.debug_mode {
            debug!("[WebSocketManager] Serialized initial data size: {} bytes", initial_data_str.len());
        }
        
        addr.do_send(SendText(initial_data_str));
        Ok(())
    }
}

impl Clone for WebSocketManager {
    fn clone(&self) -> Self {
        Self {
            binary_buffer: self.binary_buffer.clone(),
            connections: self.connections.clone(),
            addr: self.addr.clone(),
            debug_mode: self.debug_mode,
            app_state: self.app_state.clone(),
        }
    }
}

pub struct WebSocketSession {
    manager: Arc<WebSocketManager>,
    hb: Instant,
    last_pong: Instant,
}

impl WebSocketSession {
    pub fn new(manager: Arc<WebSocketManager>) -> Self {
        Self {
            manager,
            hb: Instant::now(),
            last_pong: Instant::now(),
        }
    }

    fn start_heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(HEARTBEAT_INTERVAL, |act, ctx| {
            if Instant::now().duration_since(act.last_pong) > HEARTBEAT_TIMEOUT {
                warn!("[WebSocketSession] Client heartbeat failed, disconnecting!");
                ctx.stop();
                return;
            }

            ctx.ping(b"");
        });
    }
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        if self.manager.debug_mode {
            debug!("[WebSocketSession] Session started");
        }
        self.start_heartbeat(ctx);
        
        let addr = ctx.address();
        let manager = self.manager.clone();
        actix::spawn(async move {
            manager.add_connection(addr).await;
        });
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        if self.manager.debug_mode {
            debug!("[WebSocketSession] Session stopped");
        }
        
        let addr = ctx.address();
        let manager = self.manager.clone();
        actix::spawn(async move {
            manager.remove_connection(&addr).await;
        });
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_pong = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                if self.manager.debug_mode {
                    debug!("[WebSocketSession] Text message received: {}", text);
                }
                if let Ok(json) = serde_json::from_str::<Value>(&text) {
                    if let Some(msg_type) = json["type"].as_str() {
                        let snake_type = msg_type.replace("initialData", "initial_data");
                        if snake_type == "initial_data" {
                            if self.manager.debug_mode {
                                debug!("[WebSocketSession] Initial data request received");
                            }
                            let addr = ctx.address();
                            let manager = self.manager.clone();
                            actix::spawn(async move {
                                if let Err(e) = manager.send_initial_data(&addr).await {
                                    error!("[WebSocketSession] Failed to send initial data: {}", e);
                                }
                            });
                        }
                    }
                }

                if text.contains("\"type\":\"ping\"") {
                    ctx.text("{\"type\":\"pong\"}");
                }
            }
            Ok(ws::Message::Binary(bin)) => {
                if self.manager.debug_mode {
                    debug!("[WebSocketSession] Binary message received: {} bytes", bin.len());
                }
                let connections = self.manager.connections.clone();
                let bin_data = bin.to_vec();
                actix::spawn(async move {
                    if let Ok(connections) = connections.lock() {
                        for addr in connections.iter() {
                            addr.do_send(SendBinary(bin_data.clone()));
                        }
                    }
                });
            }
            Ok(ws::Message::Close(reason)) => {
                if self.manager.debug_mode {
                    debug!("[WebSocketSession] Close message received: {:?}", reason);
                }
                ctx.close(reason);
                ctx.stop();
            }
            Err(e) => {
                error!("[WebSocketSession] Error in websocket message: {}", e);
                ctx.stop();
            }
            _ => (),
        }
    }
}

impl Handler<SendBinary> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendBinary, ctx: &mut Self::Context) {
        let data = &msg.0;
        if data.len() >= HEADER_SIZE && self.manager.debug_mode {
            let mut header_bytes = [0u8; 4];
            header_bytes.copy_from_slice(&data[0..4]);
            let is_initial = f32::from_le_bytes(header_bytes) >= 1.0;
            let num_nodes = (data.len() - HEADER_SIZE) / NODE_SIZE;

            trace!("[WebSocketSession] Sending binary update: is_initial={}, num_nodes={}", is_initial, num_nodes);
            
            let hex_dump: String = data.iter()
                .take(32)
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ");
            trace!("[WebSocketSession] Binary header (first 32 bytes): {}", hex_dump);
        }
        ctx.binary(msg.0);
    }
}

impl Handler<SendText> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        if self.manager.debug_mode {
            debug!("[WebSocketSession] Sending text message: {}", msg.0);
        }
        ctx.text(msg.0);
    }
}

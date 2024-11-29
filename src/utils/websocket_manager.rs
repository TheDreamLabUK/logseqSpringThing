use actix::prelude::*;
use actix_web_actors::ws;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use bytemuck;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde_json::json;

use crate::models::node::GPUNode;
use crate::models::graph::GraphData;

// Constants for binary data
const FLOAT32_SIZE: usize = std::mem::size_of::<f32>();
const HEADER_SIZE: usize = FLOAT32_SIZE; // isInitialLayout flag
const NODE_SIZE: usize = 6 * FLOAT32_SIZE; // x, y, z, vx, vy, vz

pub struct WebSocketManager {
    binary_buffer: Arc<Mutex<Vec<u8>>>,
    connections: Arc<Mutex<Vec<Addr<WebSocketSession>>>>,
    addr: Option<Addr<Self>>,
}

// Message types
#[derive(Message, Clone)]
#[rtype(result = "()")]
pub enum BroadcastMsg {
    Text(String),
    Binary(Vec<u8>),
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastBinary {
    pub nodes: Arc<Vec<GPUNode>>,
    pub is_initial: bool,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastGraph {
    pub graph: Arc<GraphData>,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastError {
    pub message: String,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastAudio {
    pub data: Vec<u8>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        Self {
            binary_buffer: Arc::new(Mutex::new(Vec::with_capacity(1024 * 1024))), // 1MB initial capacity
            connections: Arc::new(Mutex::new(Vec::new())),
            addr: None,
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
        let mut connections = self.connections.lock().await;
        connections.push(addr);
    }

    pub async fn remove_connection(&self, addr: &Addr<WebSocketSession>) {
        let mut connections = self.connections.lock().await;
        connections.retain(|x| x != addr);
    }

    pub async fn broadcast_binary(&self, nodes: &[GPUNode], is_initial: bool) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(addr) = self.get_addr() {
            addr.do_send(BroadcastBinary {
                nodes: Arc::new(nodes.to_vec()),
                is_initial,
            });
        }
        Ok(())
    }

    pub async fn broadcast_graph_update(&self, graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(addr) = self.get_addr() {
            addr.do_send(BroadcastGraph {
                graph: Arc::new(graph.clone()),
            });
        }
        Ok(())
    }

    pub async fn broadcast_message(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(addr) = self.get_addr() {
            addr.do_send(BroadcastMsg::Text(message.to_string()));
        }
        Ok(())
    }

    pub async fn handle_websocket(
        req: HttpRequest,
        stream: web::Payload,
        websocket_manager: web::Data<Arc<WebSocketManager>>,
    ) -> Result<HttpResponse, Error> {
        let ws = WebSocketSession::new(websocket_manager.get_ref().clone());
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

impl Actor for WebSocketManager {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.addr = Some(ctx.address());
    }
}

impl Handler<BroadcastBinary> for WebSocketManager {
    type Result = ResponseActFuture<Self, ()>;

    fn handle(&mut self, msg: BroadcastBinary, _: &mut Context<Self>) -> Self::Result {
        let binary_buffer = self.binary_buffer.clone();
        let connections = self.connections.clone();
        let nodes = msg.nodes.clone();
        
        Box::pin(async move {
            let mut buffer = binary_buffer.lock().await;
            let total_size = HEADER_SIZE + nodes.len() * NODE_SIZE;
            
            // Create a new buffer with the required capacity
            let mut new_buffer = Vec::with_capacity(total_size);
            
            // Write initial flag
            let initial_flag: f32 = if msg.is_initial { 1.0 } else { 0.0 };
            new_buffer.extend_from_slice(bytemuck::bytes_of(&initial_flag));

            // Write node data
            for node in nodes.iter() {
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
            let connections = connections.lock().await;
            for addr in connections.iter() {
                addr.do_send(BroadcastMsg::Binary(binary_data.clone()));
            }
        }.into_actor(self))
    }
}

impl Handler<BroadcastGraph> for WebSocketManager {
    type Result = ResponseActFuture<Self, ()>;

    fn handle(&mut self, msg: BroadcastGraph, _: &mut Context<Self>) -> Self::Result {
        let connections = self.connections.clone();
        let graph = msg.graph.clone();
        
        Box::pin(async move {
            let msg = json!({
                "type": "graphUpdate",
                "graphData": &*graph
            });

            let connections = connections.lock().await;
            for addr in connections.iter() {
                addr.do_send(BroadcastMsg::Text(msg.to_string()));
            }
        }.into_actor(self))
    }
}

impl Handler<BroadcastError> for WebSocketManager {
    type Result = ResponseActFuture<Self, ()>;

    fn handle(&mut self, msg: BroadcastError, _: &mut Context<Self>) -> Self::Result {
        let connections = self.connections.clone();
        let error_msg = msg.message.clone();
        
        Box::pin(async move {
            let msg = json!({
                "type": "error",
                "message": error_msg
            });

            let connections = connections.lock().await;
            for addr in connections.iter() {
                addr.do_send(BroadcastMsg::Text(msg.to_string()));
            }
        }.into_actor(self))
    }
}

impl Handler<BroadcastAudio> for WebSocketManager {
    type Result = ResponseActFuture<Self, ()>;

    fn handle(&mut self, msg: BroadcastAudio, _: &mut Context<Self>) -> Self::Result {
        let connections = self.connections.clone();
        let audio_data = msg.data.clone();
        
        Box::pin(async move {
            let connections = connections.lock().await;
            for addr in connections.iter() {
                addr.do_send(BroadcastMsg::Binary(audio_data.clone()));
            }
        }.into_actor(self))
    }
}

// WebSocket session actor
pub struct WebSocketSession {
    manager: WebSocketManager,
}

impl WebSocketSession {
    pub fn new(manager: WebSocketManager) -> Self {
        Self { manager }
    }
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;
}

impl Handler<BroadcastMsg> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: BroadcastMsg, ctx: &mut Self::Context) {
        match msg {
            BroadcastMsg::Text(text) => ctx.text(text),
            BroadcastMsg::Binary(data) => ctx.binary(data),
        }
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => ctx.text(text),
            Ok(ws::Message::Binary(bin)) => ctx.binary(bin),
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

use actix::prelude::*;
use actix_web_actors::ws;
use bytestring::ByteString;
use log::{debug, error, info};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

use crate::models::node::GPUNode;
use crate::models::position_update::NodePositionVelocity;
use crate::utils::websocket_messages::{SendBinary, SendText};

pub struct WebSocketManager {
    pub sessions: Arc<RwLock<Vec<Addr<WebSocketSession>>>>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        WebSocketManager {
            sessions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn broadcast_binary(&self, nodes: &[GPUNode], is_initial: bool) {
        debug!("Broadcasting binary update: {} nodes, initial: {}", nodes.len(), is_initial);

        // Create binary message
        let mut binary_data = Vec::with_capacity(4 + nodes.len() * std::mem::size_of::<NodePositionVelocity>());
        
        // Add initial layout flag as float32
        let flag_bytes = (if is_initial { 1.0f32 } else { 0.0f32 }).to_le_bytes();
        binary_data.extend_from_slice(&flag_bytes);

        // Add node positions and velocities
        for node in nodes {
            let update = NodePositionVelocity {
                x: node.x,
                y: node.y,
                z: node.z,
                vx: node.vx,
                vy: node.vy,
                vz: node.vz,
            };
            binary_data.extend_from_slice(bytemuck::bytes_of(&update));
        }

        debug!("Binary message created: {} bytes, sample nodes: {:?}", 
            binary_data.len(),
            nodes.iter().take(3).map(|n| json!({
                "x": n.x,
                "y": n.y,
                "z": n.z,
                "vx": n.vx,
                "vy": n.vy,
                "vz": n.vz
            })).collect::<Vec<_>>()
        );

        // Send to all connected clients
        let sessions = self.sessions.read().await;
        for session in sessions.iter() {
            if let Err(e) = session.try_send(SendBinary(binary_data.clone())) {
                error!("Failed to send binary update to session: {}", e);
            }
        }

        debug!("Binary update broadcast complete: {} recipients", sessions.len());
    }

    pub async fn broadcast_text(&self, message: &str) {
        debug!("Broadcasting text message: {} chars", message.len());
        
        let sessions = self.sessions.read().await;
        for session in sessions.iter() {
            if let Err(e) = session.try_send(SendText(message.to_string())) {
                error!("Failed to send text message to session: {}", e);
            }
        }

        debug!("Text message broadcast complete: {} recipients", sessions.len());
    }

    pub async fn add_session(&self, addr: Addr<WebSocketSession>) {
        let mut sessions = self.sessions.write().await;
        debug!("Adding new WebSocket session. Total sessions: {}", sessions.len() + 1);
        sessions.push(addr);
    }

    pub async fn remove_session(&self, addr: &Addr<WebSocketSession>) {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|x| x != addr);
        debug!("Removed WebSocket session. Remaining sessions: {}", sessions.len());
    }
}

pub struct WebSocketSession;

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        debug!("WebSocket session started");
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        debug!("WebSocket session stopped");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                debug!("Received ping");
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                debug!("Received pong");
            }
            Ok(ws::Message::Text(text)) => {
                debug!("Received text message: {} chars", text.len());
                ctx.text(text);
            }
            Ok(ws::Message::Binary(bin)) => {
                debug!("Received binary message: {} bytes", bin.len());
                ctx.binary(bin);
            }
            Ok(ws::Message::Close(reason)) => {
                debug!("Received close message: {:?}", reason);
                ctx.close(reason);
            }
            _ => {}
        }
    }
}

impl Handler<SendText> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        debug!("Sending text message: {} chars", msg.0.len());
        ctx.text(ByteString::from(msg.0));
    }
}

impl Handler<SendBinary> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendBinary, ctx: &mut Self::Context) {
        debug!("Sending binary message: {} bytes", msg.0.len());
        ctx.binary(msg.0);
    }
}

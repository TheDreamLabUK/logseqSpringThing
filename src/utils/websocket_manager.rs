// websocket_manager.rs


use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use crate::AppState;
use log::info;
use std::sync::{Mutex}; // Import Mutex

/// Manages WebSocket connections and broadcasts updates to connected clients.
pub struct WebSocketManager {
    pub sessions: Mutex<Vec<Addr<WebSocketSession>>>, // Use Mutex to wrap the sessions vector
}

impl WebSocketManager {
    /// Creates a new WebSocketManager instance.
    pub fn new() -> Self {
        WebSocketManager {
            sessions: Mutex::new(Vec::new()), // Initialize the Mutex with an empty vector
        }
    }

    /// Sets up a WebSocket route handler.
    pub async fn handle_websocket(req: HttpRequest, stream: web::Payload, state: web::Data<AppState>) -> Result<HttpResponse, Error> {
        let session = WebSocketSession::new(state.clone());
        let resp = ws::start(session, &req, stream)?;
        Ok(resp)
    }

    /// Broadcasts a message to all connected WebSocket clients.
    pub fn broadcast_message(&self, message: &str) {
        let sessions = self.sessions.lock().unwrap(); // Lock the Mutex to access the sessions vector
        for session in &*sessions { // Iterate over the sessions
            session.do_send(BroadcastMessage(message.to_string()));
        }
    }
}

/// Represents a WebSocket session with a client.
pub struct WebSocketSession {
    state: web::Data<AppState>,
}

impl WebSocketSession {
    /// Creates a new WebSocketSession instance.
    fn new(state: web::Data<AppState>) -> Self {
        WebSocketSession { state }
    }
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    /// Called when the WebSocket session is started.
    fn started(&mut self, ctx: &mut Self::Context) {
        // Add the session's address to the WebSocketManager's sessions list.
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().push(addr); // Lock the mutex
        info!("WebSocket session started. Total sessions: {}", self.state.websocket_manager.sessions.lock().unwrap().len()); // Lock the mutex
    }

    /// Called when the WebSocket session is stopped.
    fn stopped(&mut self, ctx: &mut Self::Context) {
        // Remove the session from the manager.
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().retain(|session| session != &addr); // Lock the mutex
        info!("WebSocket session stopped. Total sessions: {}", self.state.websocket_manager.sessions.lock().unwrap().len()); // Lock the mutex
    }
}

/// Message for broadcasting data to WebSocket clients.
#[derive(Message)]
#[rtype(result = "()")]
struct BroadcastMessage(String);

impl Handler<BroadcastMessage> for WebSocketSession {
    type Result = ();

    /// Handles the broadcast message by sending it to the client.
    fn handle(&mut self, msg: BroadcastMessage, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    /// Handles incoming WebSocket messages from the client.
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                ctx.pong(&msg);
            },
            Ok(ws::Message::Pong(_)) => {
                // Optionally handle pong responses.
            },
            Ok(ws::Message::Text(text)) => {
                // Handle incoming text messages if necessary.
                info!("Received message from client: {}", text);
                // Echo the message back for demonstration purposes.
                ctx.text(format!("Echo: {}", text));
            },
            Ok(ws::Message::Binary(bin)) => {
                // Handle binary messages if necessary.
                ctx.binary(bin);
            },
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket closed: {:?}", reason);
                ctx.stop();
            },
            _ => (),
        }
    }
}

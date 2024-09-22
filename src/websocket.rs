use actix::{Actor, StreamHandler};
use actix_web_actors::ws;
use crate::app_state::AppState;
use std::sync::Arc;
use serde_json::json;

pub struct WebSocketSession {
    app_state: Arc<AppState>,
}

impl WebSocketSession {
    pub fn new(app_state: Arc<AppState>) -> Self {
        WebSocketSession { app_state }
    }
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                println!("Received message: {}", text);
                // Convert ByteString to String
                let text_str = text.to_string();
                // Echo the received message back to the client
                let response = json!({
                    "type": "echo",
                    "content": text_str
                });
                ctx.text(response.to_string());
            }
            Ok(ws::Message::Binary(bin)) => ctx.binary(bin),
            _ => (),
        }
    }
}

impl WebSocketSession {
    pub fn send_graph_update(&self, ctx: &mut ws::WebsocketContext<Self>) {
        // This method can be implemented later for sending graph updates
        // For now, we'll just send a dummy update
        let dummy_update = json!({
            "type": "graph_update",
            "content": "Dummy graph update"
        });
        ctx.text(dummy_update.to_string());
    }
}

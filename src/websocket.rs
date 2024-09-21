use actix::{Actor, StreamHandler};
use actix_web_actors::ws;
use crate::app_state::AppState;
use std::sync::Arc;

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
                // Here you can implement the logic to handle incoming messages
                // For example, updating the graph data or triggering a refresh
            }
            Ok(ws::Message::Binary(bin)) => ctx.binary(bin),
            _ => (),
        }
    }
}

impl WebSocketSession {
    // Add methods to send updates to the client
    pub fn send_graph_update(&self, ctx: &mut ws::WebsocketContext<Self>) {
        // Implement logic to send graph updates to the client
        // For example:
        // let graph_data = self.app_state.graph_data.read().await;
        // let json = serde_json::to_string(&*graph_data).unwrap();
        // ctx.text(json);
    }
}

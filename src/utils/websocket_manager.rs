use tokio::sync::mpsc;
use futures::{StreamExt, SinkExt};
use warp::ws::{Message, WebSocket};

pub struct WebSocketManager;

impl WebSocketManager {
    pub fn new() -> Self {
        WebSocketManager
    }

    pub async fn handle_websocket(&self, ws: WebSocket) {
        let (mut ws_sender, mut ws_receiver) = ws.split();
        let (tx, mut rx) = mpsc::channel(32);

        tokio::task::spawn(async move {
            while let Some(message) = rx.recv().await {
                ws_sender.send(message).await.unwrap();
            }
        });

        while let Some(result) = ws_receiver.next().await {
            match result {
                Ok(msg) => {
                    if msg.is_text() || msg.is_binary() {
                        tx.send(msg).await.unwrap();
                    }
                }
                Err(e) => {
                    eprintln!("websocket error: {:?}", e);
                    break;
                }
            }
        }
    }
}

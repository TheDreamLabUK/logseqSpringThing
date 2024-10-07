use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::connect_async;
use tungstenite::protocol::Message;
use serde_json::json;
use std::sync::Arc;
use tokio::task;
use crate::config::Settings;
use log::{info, error, debug};
use futures::{SinkExt, StreamExt};
use std::error::Error;
use std::collections::HashSet;
use actix::Addr;
use crate::utils::websocket_manager::{WebSocketSession, WebSocketManager};
use crate::services::sonata_service::SonataService;

pub struct SpeechService {
    sender: Arc<Mutex<mpsc::Sender<SpeechCommand>>>,
    sessions: Arc<Mutex<HashSet<Addr<WebSocketSession>>>>,
    sonata_service: Arc<SonataService>,
    websocket_manager: Arc<WebSocketManager>,
}

#[derive(Debug)]
enum SpeechCommand {
    Initialize,
    SendMessage(String),
    Close,
}

impl SpeechService {
    pub fn new(sonata_service: Arc<SonataService>, websocket_manager: Arc<WebSocketManager>) -> Self {
        let (tx, rx) = mpsc::channel(100);
        let sender = Arc::new(Mutex::new(tx));
        let sessions = Arc::new(Mutex::new(HashSet::new()));

        let service = SpeechService {
            sender,
            sessions,
            sonata_service,
            websocket_manager,
        };

        service.start(rx);

        service
    }

    fn start(&self, mut receiver: mpsc::Receiver<SpeechCommand>) {
        let sonata_service = self.sonata_service.clone();
        let websocket_manager = self.websocket_manager.clone();

        task::spawn(async move {
            let mut ws_stream = None;

            while let Some(command) = receiver.recv().await {
                match command {
                    SpeechCommand::Initialize => {
                        // Initialize WebSocket connection to OpenAI
                        let url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01";
                        match connect_async(url).await {
                            Ok((stream, _)) => {
                                info!("Connected to OpenAI Realtime API");
                                ws_stream = Some(stream);
                            },
                            Err(e) => error!("Failed to connect to OpenAI Realtime API: {}", e),
                        }
                    },
                    SpeechCommand::SendMessage(msg) => {
                        if let Some(stream) = &mut ws_stream {
                            let (mut write, mut read) = stream.split();

                            // Send message to OpenAI
                            let event = json!({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": msg
                                        }
                                    ]
                                }
                            });

                            if let Err(e) = write.send(Message::Text(event.to_string())).await {
                                error!("Failed to send message to OpenAI: {}", e);
                            } else {
                                info!("Sent message to OpenAI: {}", msg);
                            }

                            // Handle response from OpenAI
                            while let Some(message) = read.next().await {
                                match message {
                                    Ok(Message::Text(text)) => {
                                        debug!("Received message from OpenAI: {}", text);
                                        // Process the incoming message and synthesize audio
                                        if let Ok(json_msg) = serde_json::from_str::<serde_json::Value>(&text) {
                                            if let Some(content) = json_msg["content"].as_str() {
                                                // Synthesize audio using Sonata
                                                match sonata_service.synthesize(content).await {
                                                    Ok(audio_bytes) => {
                                                        // Broadcast audio to all WebSocket sessions
                                                        if let Err(e) = websocket_manager.broadcast_audio(audio_bytes).await {
                                                            error!("Failed to broadcast audio: {}", e);
                                                        }
                                                    },
                                                    Err(e) => error!("Failed to synthesize audio: {}", e),
                                                }
                                            }
                                        }
                                    },
                                    Ok(Message::Close(_)) => {
                                        info!("OpenAI WebSocket connection closed by server");
                                        break;
                                    },
                                    Err(e) => {
                                        error!("OpenAI WebSocket error: {}", e);
                                        break;
                                    },
                                    _ => {},
                                }
                            }
                        } else {
                            error!("WebSocket connection not initialized");
                        }
                    },
                    SpeechCommand::Close => {
                        if let Some(stream) = &mut ws_stream {
                            if let Err(e) = stream.close(None).await {
                                error!("Failed to close WebSocket connection: {}", e);
                            }
                        }
                        break;
                    },
                }
            }
        });
    }

    pub async fn initialize(&self, _settings: &Settings) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::Initialize;
        self.sender.lock().await.send(command).await?;
        Ok(())
    }

    pub async fn send_message(&self, message: String) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::SendMessage(message);
        self.sender.lock().await.send(command).await?;
        Ok(())
    }

    pub async fn close(&self) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::Close;
        self.sender.lock().await.send(command).await?;
        Ok(())
    }

    pub async fn register_session(&self, addr: Addr<WebSocketSession>) {
        self.sessions.lock().await.insert(addr);
    }

    pub async fn unregister_session(&self, addr: &Addr<WebSocketSession>) {
        self.sessions.lock().await.remove(addr);
    }
}
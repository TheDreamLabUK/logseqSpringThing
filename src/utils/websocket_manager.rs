use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use crate::AppState;
use log::{info, error, debug};
use std::sync::{Mutex, Arc};
use serde_json::json;
use futures::stream::StreamExt;
use std::error::Error as StdError;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bytestring::ByteString;

use crate::models::simulation_params::SimulationMode;
use crate::utils::compression::{compress_message, decompress_message};
use crate::models::simulation_params::SimulationParams;
use crate::utils::websocket_messages::{ClientMessage, SendCompressedMessage, MessageHandler};
use crate::utils::websocket_openai::OpenAIWebSocket;
use crate::utils::websocket_messages::OpenAIMessage;

/// Manages WebSocket sessions and communication.
pub struct WebSocketManager {
    pub sessions: Mutex<Vec<Addr<WebSocketSession>>>,
    pub conversation_id: Arc<Mutex<Option<String>>>,
}

impl WebSocketManager {
    /// Creates a new WebSocketManager instance.
    pub fn new() -> Self {
        WebSocketManager {
            sessions: Mutex::new(Vec::new()),
            conversation_id: Arc::new(Mutex::new(None)),
        }
    }

    /// Initializes the WebSocketManager with a conversation ID.
    pub async fn initialize(&self, ragflow_service: &crate::services::ragflow_service::RAGFlowService) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let conversation_id = ragflow_service.create_conversation("default_user".to_string()).await?;
        let mut conv_id_lock = self.conversation_id.lock().unwrap();
        *conv_id_lock = Some(conversation_id.clone());
        info!("Initialized conversation with ID: {}", conversation_id);
        Ok(())
    }

    /// Handles incoming WebSocket connection requests.
    pub async fn handle_websocket(&self, req: HttpRequest, stream: web::Payload, state: web::Data<AppState>) -> Result<HttpResponse, Error> {
        info!("New WebSocket connection request");
        let session = WebSocketSession::new(state.clone(), Some(self.conversation_id.clone()));
        ws::start(session, &req, stream)
    }

    /// Broadcasts a compressed message to all connected WebSocket sessions.
    pub async fn broadcast_message_compressed(&self, message: &str) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let compressed = compress_message(message)?;
        let sessions = self.sessions.lock().unwrap().clone();
        for session in sessions.iter() {
            session.do_send(SendCompressedMessage(compressed.clone()));
        }
        Ok(())
    }

    /// Broadcasts audio data to all connected WebSocket sessions.
    pub async fn broadcast_audio(&self, audio_bytes: Vec<u8>) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let json_data = json!({
            "type": "audio_data",
            "audio_data": BASE64.encode(audio_bytes.as_slice())
        });
        let message = json_data.to_string();
        self.broadcast_message_compressed(&message).await
    }
}

/// WebSocket session actor.
pub struct WebSocketSession {
    state: web::Data<AppState>,
    tts_method: String,
    openai_ws: Option<Addr<OpenAIWebSocket>>,
    simulation_mode: SimulationMode,
    conversation_id: Option<Arc<Mutex<Option<String>>>>,
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().push(addr.clone());
        info!(
            "WebSocket session started. Total sessions: {}",
            self.state.websocket_manager.sessions.lock().unwrap().len()
        );

        // Initialize OpenAI WebSocket
        self.openai_ws = Some(OpenAIWebSocket::new(ctx.address(), self.state.settings.clone()).start());
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().retain(|session| session != &addr);
        info!(
            "WebSocket session stopped. Total sessions: {}",
            self.state.websocket_manager.sessions.lock().unwrap().len()
        );
    }
}

impl MessageHandler for WebSocketSession {}

impl Handler<SendCompressedMessage> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendCompressedMessage, ctx: &mut Self::Context) {
        ctx.binary(msg.0);
    }
}

impl WebSocketSession {
    pub fn new(state: web::Data<AppState>, conversation_id: Option<Arc<Mutex<Option<String>>>>) -> Self {
        WebSocketSession {
            state,
            tts_method: "piper".to_string(),
            openai_ws: None,
            simulation_mode: SimulationMode::Remote,
            conversation_id,
        }
    }

    fn handle_chat_message(&mut self, ctx: &mut ws::WebsocketContext<Self>, message: String, use_openai: bool) {
        let state = self.state.clone();
        let conversation_id = self.conversation_id.clone();
        let ctx_addr = ctx.address();
        let openai_ws = self.openai_ws.clone();
        
        ctx.spawn(async move {
            let conv_id = if let Some(conv_arc) = conversation_id {
                if let Some(id) = conv_arc.lock().unwrap().clone() {
                    id
                } else {
                    match state.ragflow_service.create_conversation("default_user".to_string()).await {
                        Ok(new_id) => new_id,
                        Err(e) => {
                            error!("Failed to create conversation: {}", e);
                            return;
                        }
                    }
                }
            } else {
                error!("No conversation ID available");
                return;
            };

            match state.ragflow_service.send_message(
                conv_id.clone(),
                message.clone(),
                false,
                None,
                false,
            ).await {
                Ok(mut stream) => {
                    debug!("RAGFlow service initialized for conversation {}", conv_id);
                    
                    if let Some(result) = stream.next().await {
                        match result {
                            Ok(expanded_text) => {
                                let response = json!({
                                    "type": "ragflow_response",
                                    "answer": expanded_text.clone()
                                });
                                if let Ok(response_str) = serde_json::to_string(&response) {
                                    if let Ok(compressed) = compress_message(&response_str) {
                                        ctx_addr.do_send(SendCompressedMessage(compressed));
                                    }
                                }

                                if use_openai {
                                    if let Some(ref openai_ws) = openai_ws {
                                        openai_ws.do_send(OpenAIMessage(expanded_text));
                                    } else {
                                        error!("OpenAI WebSocket not initialized");
                                        let error_message = json!({
                                            "type": "error",
                                            "message": "OpenAI WebSocket not initialized"
                                        });
                                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                                            if let Ok(compressed) = compress_message(&error_str) {
                                                ctx_addr.do_send(SendCompressedMessage(compressed));
                                            }
                                        }
                                    }
                                } else {
                                    if let Err(e) = state.speech_service.send_message(expanded_text).await {
                                        error!("Failed to generate speech: {}", e);
                                        let error_message = json!({
                                            "type": "error",
                                            "message": format!("Failed to generate speech: {}", e)
                                        });
                                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                                            if let Ok(compressed) = compress_message(&error_str) {
                                                ctx_addr.do_send(SendCompressedMessage(compressed));
                                            }
                                        }
                                    }
                                }
                            },
                            Err(e) => {
                                error!("Error processing RAGFlow response: {}", e);
                                let error_message = json!({
                                    "type": "error",
                                    "message": format!("Error processing RAGFlow response: {}", e)
                                });
                                if let Ok(error_str) = serde_json::to_string(&error_message) {
                                    if let Ok(compressed) = compress_message(&error_str) {
                                        ctx_addr.do_send(SendCompressedMessage(compressed));
                                    }
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    error!("Failed to send message to RAGFlow: {}", e);
                    let error_message = json!({
                        "type": "error",
                        "message": format!("Failed to send message to RAGFlow: {}", e)
                    });
                    if let Ok(error_str) = serde_json::to_string(&error_message) {
                        if let Ok(compressed) = compress_message(&error_str) {
                            ctx_addr.do_send(SendCompressedMessage(compressed));
                        }
                    }
                }
            }
        }.into_actor(self));
    }

    fn handle_simulation(&mut self, ctx: &mut ws::WebsocketContext<Self>, mode: &str) {
        self.simulation_mode = match mode {
            "remote" => {
                info!("Simulation mode set to Remote (GPU-accelerated)");
                SimulationMode::Remote
            },
            "gpu" => {
                info!("Simulation mode set to GPU (local)");
                SimulationMode::GPU
            },
            "local" => {
                info!("Simulation mode set to Local (CPU)");
                SimulationMode::Local
            },
            _ => {
                error!("Invalid simulation mode: {}, defaulting to Remote", mode);
                SimulationMode::Remote
            }
        };

        let response = json!({
            "type": "simulation_mode_set",
            "mode": mode,
            "gpu_enabled": matches!(self.simulation_mode, SimulationMode::Remote | SimulationMode::GPU)
        });
        self.send_json_response(response, ctx);
    }

    fn handle_layout(&mut self, ctx: &mut ws::WebsocketContext<Self>, params: SimulationParams) {
        let state = self.state.clone();
        let simulation_mode = self.simulation_mode.clone();
        
        ctx.spawn(async move {
            let graph_service = state.graph_service.read().await;
            if let Some(service) = graph_service.as_ref() {
                let result = match simulation_mode {
                    SimulationMode::Remote => service.recalculate_layout_gpu(&params).await,
                    _ => service.recalculate_layout(&params).await,
                };

                match result {
                    Ok(_) => {
                        if let Ok(graph_data) = service.get_graph_data().await {
                            let response = json!({
                                "type": "layout_update",
                                "layout_data": graph_data
                            });
                            state.websocket_manager.broadcast_message_compressed(&response.to_string()).await.unwrap();
                        }
                    },
                    Err(e) => {
                        error!("Failed to recalculate layout: {}", e);
                        let error_message = json!({
                            "type": "error",
                            "message": format!("Layout calculation failed: {}", e)
                        });
                        state.websocket_manager.broadcast_message_compressed(&error_message.to_string()).await.unwrap();
                    }
                }
            }
        }.into_actor(self));
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Pong(_)) => (),
            Ok(ws::Message::Text(text)) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(client_msg) => match client_msg {
                        ClientMessage::ChatMessage { message, use_openai } => {
                            self.handle_chat_message(ctx, message, use_openai);
                        },
                        ClientMessage::SetTTSMethod { method } => {
                            self.tts_method = method.clone();
                            let response = json!({
                                "type": "tts_method_set",
                                "method": method
                            });
                            self.send_json_response(response, ctx);
                        },
                        ClientMessage::SetSimulationMode { mode } => {
                            self.handle_simulation(ctx, &mode);
                        },
                        ClientMessage::RecalculateLayout { params } => {
                            self.handle_layout(ctx, params);
                        },
                        ClientMessage::GetInitialData => {
                            // Handle initial data request
                        },
                    },
                    Err(e) => {
                        error!("Failed to parse client message: {}", e);
                        let error_message = json!({
                            "type": "error",
                            "message": "Invalid message format"
                        });
                        self.send_json_response(error_message, ctx);
                    }
                }
            },
            Ok(ws::Message::Binary(bin)) => {
                if let Ok(text) = decompress_message(&bin) {
                    StreamHandler::handle(self, Ok(ws::Message::Text(ByteString::from(text))), ctx);
                } else {
                    error!("Failed to decompress binary message");
                }
            },
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            },
            Err(e) => {
                error!("WebSocket error: {}", e);
                ctx.stop();
            },
            _ => (),
        }
    }
}

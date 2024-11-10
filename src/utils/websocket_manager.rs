use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use log::{info, error};
use std::sync::{Mutex, Arc};
use serde_json::json;
use actix_web_actors::ws::WebsocketContext;

use crate::AppState;
use crate::models::simulation_params::SimulationMode;
use crate::handlers::{WebSocketSession, WebSocketSessionHandler};
use crate::utils::websocket_messages::{MessageHandler, SendText, ClientMessage};

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
    pub async fn initialize(&self, ragflow_service: &crate::services::ragflow_service::RAGFlowService) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conversation_id = ragflow_service.create_conversation("default_user".to_string()).await?;
        let mut conv_id_lock = self.conversation_id.lock().unwrap();
        *conv_id_lock = Some(conversation_id.clone());
        info!("Initialized conversation with ID: {}", conversation_id);
        Ok(())
    }

    /// Handles incoming WebSocket connection requests.
    pub async fn handle_websocket(req: HttpRequest, stream: web::Payload, state: web::Data<AppState>) -> Result<HttpResponse, Error> {
        info!("New WebSocket connection request");
        let session = WebSocketSession {
            state: state.clone(),
            tts_method: "piper".to_string(),
            openai_ws: None,
            simulation_mode: SimulationMode::Remote,
            conversation_id: Some(state.websocket_manager.conversation_id.clone()),
        };
        ws::start(session, &req, stream)
    }

    /// Broadcasts a message to all connected WebSocket sessions.
    pub async fn broadcast_message(&self, message: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let sessions = self.sessions.lock().unwrap().clone();
        for session in sessions {
            let msg: SendText = SendText(message.to_string());
            session.do_send(msg);
        }
        Ok(())
    }

    /// Broadcasts graph update to all connected WebSocket sessions.
    pub async fn broadcast_graph_update(&self, graph_data: &crate::models::graph::GraphData) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json_data = json!({
            "type": "graph_update",
            "graph_data": graph_data
        });
        let message = json_data.to_string();
        self.broadcast_message(&message).await
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut WebsocketContext<Self>) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                let ctx: &mut WebsocketContext<WebSocketSession> = ctx;
                ctx.pong(&msg);
            },
            Ok(ws::Message::Pong(_)) => (),
            Ok(ws::Message::Text(text)) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(client_msg) => match client_msg {
                        ClientMessage::ChatMessage { message, use_openai } => {
                            WebSocketSessionHandler::handle_chat_message(self, ctx, message, use_openai);
                        },
                        ClientMessage::SetSimulationMode { mode } => {
                            WebSocketSessionHandler::handle_simulation_mode(self, ctx, &mode);
                        },
                        ClientMessage::RecalculateLayout { params } => {
                            WebSocketSessionHandler::handle_layout(self, ctx, params);
                        },
                        ClientMessage::GetInitialData => {
                            WebSocketSessionHandler::handle_initial_data(self, ctx);
                        },
                        ClientMessage::UpdateFisheyeSettings { enabled, strength, focus_point, radius } => {
                            WebSocketSessionHandler::handle_fisheye_settings(self, ctx, enabled, strength, focus_point, radius);
                        },
                        _ => {
                            error!("Unhandled client message type");
                            let error_message = json!({
                                "type": "error",
                                "message": "Unhandled message type"
                            });
                            MessageHandler::send_json_response(self, error_message, ctx);
                        }
                    },
                    Err(e) => {
                        error!("Failed to parse client message: {}", e);
                        let error_message = json!({
                            "type": "error",
                            "message": format!("Invalid message format: {}", e)
                        });
                        MessageHandler::send_json_response(self, error_message, ctx);
                    }
                }
            },
            Ok(ws::Message::Binary(bin)) => {
                if let Some(gpu_compute) = &self.state.gpu_compute {
                    let gpu = gpu_compute.clone();
                    let bin_data = bin.to_vec();
                    let ctx_addr = ctx.address();

                    ctx.spawn(
                        async move {
                            let mut gpu = gpu.write().await;
                            if let Err(e) = gpu.update_positions(&bin_data).await {
                                error!("Failed to update node positions: {}", e);
                                let error_message = json!({
                                    "type": "error",
                                    "message": format!("Failed to update node positions: {}", e)
                                });
                                if let Ok(error_str) = serde_json::to_string(&error_message) {
                                    let msg: SendText = SendText(error_str);
                                    ctx_addr.do_send(msg);
                                }
                            }
                            let msg: SendText = SendText("Position update complete".to_string());
                            ctx_addr.do_send(msg);
                        }
                        .into_actor(self)
                    );
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

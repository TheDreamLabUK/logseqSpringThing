use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use log::{info, error, debug};
use std::sync::{Mutex, Arc};
use serde_json::json;
use futures::stream::StreamExt;
use serde::Deserialize;
use bytestring::ByteString;
use bytes::Bytes;

use crate::AppState;
use crate::models::simulation_params::SimulationMode;
use crate::utils::websocket_messages::{
    MessageHandler, OpenAIMessage,
    OpenAIConnected, OpenAIConnectionFailed, SendText, SendBinary
};
use crate::utils::websocket_openai::OpenAIWebSocket;
use crate::handlers::{WebSocketSession, WebSocketSessionHandler, GpuUpdate};

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "chatMessage")]
    ChatMessage {
        message: String,
        #[serde(rename = "tts_provider")]
        tts_provider: String,
    },
    #[serde(rename = "setSimulationMode")]
    SetSimulationMode {
        mode: String,
    },
    #[serde(rename = "recalculateLayout")]
    RecalculateLayout {
        params: crate::models::simulation_params::SimulationParams,
    },
    #[serde(rename = "getInitialData")]
    GetInitialData,
    #[serde(rename = "updateFisheyeSettings")]
    UpdateFisheyeSettings {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    },
}

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
            session.do_send(SendText(message.to_string()));
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

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().push(addr.clone());
        info!(
            "WebSocket session started. Total sessions: {}",
            self.state.websocket_manager.sessions.lock().unwrap().len()
        );

        // Start GPU updates if in remote mode
        if matches!(self.simulation_mode, SimulationMode::Remote) {
            if let Some(_) = &self.state.gpu_compute {
                WebSocketSessionHandler::start_gpu_updates(self, ctx);
            }
        }
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

impl Handler<GpuUpdate> for WebSocketSession {
    type Result = ResponseActFuture<Self, ()>;

    fn handle(&mut self, _: GpuUpdate, ctx: &mut Self::Context) -> Self::Result {
        let state = self.state.clone();
        let gpu_compute = if let Some(gpu) = &state.gpu_compute {
            gpu.clone()
        } else {
            return Box::pin(async {}.into_actor(self));
        };
        let ctx_addr = ctx.address();

        Box::pin(async move {
            let mut gpu = gpu_compute.write().await;
            if let Err(e) = gpu.step() {
                error!("GPU compute step failed: {}", e);
                return;
            }

            if let Ok(nodes) = gpu.get_node_positions().await {
                let binary_data = nodes.iter()
                    .flat_map(|node| {
                        let mut data = Vec::with_capacity(24);
                        data.extend_from_slice(&node.x.to_le_bytes());
                        data.extend_from_slice(&node.y.to_le_bytes());
                        data.extend_from_slice(&node.z.to_le_bytes());
                        data.extend_from_slice(&node.vx.to_le_bytes());
                        data.extend_from_slice(&node.vy.to_le_bytes());
                        data.extend_from_slice(&node.vz.to_le_bytes());
                        data
                    })
                    .collect::<Vec<u8>>();

                if let Ok(sessions) = state.websocket_manager.sessions.lock() {
                    for session in sessions.iter() {
                        if session != &ctx_addr {
                            let _ = session.do_send(SendBinary(binary_data.clone()));
                        }
                    }
                }
            }
        }
        .into_actor(self)
        .map(move |_, act, ctx| {
            // Send completion message
            act.send_json_response(json!({"type": "gpu_update_complete"}), ctx);
        }))
    }
}

impl MessageHandler for WebSocketSession {}

impl Handler<SendText> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        ctx.text(ByteString::from(msg.0));
    }
}

impl Handler<SendBinary> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendBinary, ctx: &mut Self::Context) {
        ctx.binary(Bytes::from(msg.0));
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
                        ClientMessage::ChatMessage { message, tts_provider } => {
                            WebSocketSessionHandler::handle_chat_message(self, ctx, message, tts_provider == "openai");
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
                    },
                    Err(e) => {
                        error!("Failed to parse client message: {}", e);
                        let error_message = json!({
                            "type": "error",
                            "message": format!("Invalid message format: {}", e)
                        });
                        self.send_json_response(error_message, ctx);
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
                                    ctx_addr.do_send(SendText(error_str));
                                }
                            }
                            ctx_addr.do_send(SendText("Position update complete".to_string()));
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

impl Handler<OpenAIMessage> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: OpenAIMessage, _ctx: &mut Self::Context) {
        if let Some(ref ws) = self.openai_ws {
            ws.do_send(msg);
        }
    }
}

impl Handler<OpenAIConnected> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, _: OpenAIConnected, _ctx: &mut Self::Context) {
        debug!("OpenAI WebSocket connected");
    }
}

impl Handler<OpenAIConnectionFailed> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, _: OpenAIConnectionFailed, _ctx: &mut Self::Context) {
        error!("OpenAI WebSocket connection failed");
        self.openai_ws = None;
    }
}

impl WebSocketSession {
    pub fn send_json_response(&self, response: serde_json::Value, ctx: &mut ws::WebsocketContext<Self>) {
        if let Ok(response_str) = serde_json::to_string(&response) {
            ctx.text(response_str);
        }
    }
}

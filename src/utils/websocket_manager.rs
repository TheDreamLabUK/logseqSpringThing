use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use log::{info, error, debug};
use std::sync::{Mutex, Arc};
use serde_json::json;
use futures::stream::StreamExt;
use futures::future::join_all;
use std::error::Error as StdError;
use bytestring::ByteString;
use serde::Deserialize;
use tokio::time::Duration;

use crate::AppState;
use crate::models::simulation_params::{SimulationMode, SimulationParams};
use crate::utils::compression::{compress_message, decompress_message};
use crate::utils::websocket_messages::{SendCompressedMessage, MessageHandler, OpenAIMessage};
use crate::utils::websocket_openai::OpenAIWebSocket;
use crate::services::graph_service::GraphService;

const OPENAI_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

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
        params: SimulationParams,
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

/// Helper function to convert hex color to proper format
fn format_color(color: &str) -> String {
    let color = color.trim_matches('"')
        .trim_start_matches("0x")
        .trim_start_matches('#');
    format!("#{}", color)
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
    pub async fn initialize(&self, ragflow_service: &crate::services::ragflow_service::RAGFlowService) -> Result<(), Box<dyn StdError + Send + Sync>> {
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
    pub async fn broadcast_message(&self, message: &str) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let sessions = self.sessions.lock().unwrap().clone();
        let futures: Vec<_> = sessions.iter()
            .map(|session| {
                let compressed = compress_message(message).unwrap_or_default();
                session.send(SendCompressedMessage(compressed))
            })
            .collect();
        
        let results = join_all(futures).await;
        for result in results {
            if let Err(e) = result {
                error!("Failed to broadcast message: {}", e);
            }
        }
        Ok(())
    }

    /// Broadcasts graph update to all connected WebSocket sessions.
    pub async fn broadcast_graph_update(&self, graph_data: &crate::models::graph::GraphData) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let json_data = json!({
            "type": "graph_update",
            "graph_data": graph_data
        });
        let message = json_data.to_string();
        self.broadcast_message(&message).await
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

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Pong(_)) => (),
            Ok(ws::Message::Text(text)) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(client_msg) => match client_msg {
                        ClientMessage::ChatMessage { message, tts_provider } => {
                            self.handle_chat_message(ctx, message, tts_provider == "openai");
                        },
                        ClientMessage::SetSimulationMode { mode } => {
                            self.handle_simulation(ctx, &mode);
                        },
                        ClientMessage::RecalculateLayout { params } => {
                            self.handle_layout(ctx, params);
                        },
                        ClientMessage::GetInitialData => {
                            self.handle_initial_data(ctx);
                        },
                        ClientMessage::UpdateFisheyeSettings { enabled, strength, focus_point, radius } => {
                            let state = self.state.clone();
                            let ctx_addr = ctx.address();
                            
                            ctx.spawn(async move {
                                if let Some(gpu_compute) = &state.gpu_compute {
                                    let mut gpu = gpu_compute.write().await;
                                    gpu.set_fisheye_params(enabled, strength, focus_point, radius);
                                    
                                    let response = json!({
                                        "type": "fisheye_settings_updated",
                                        "enabled": enabled,
                                        "strength": strength,
                                        "focus_point": focus_point,
                                        "radius": radius
                                    });
                                    if let Ok(response_str) = serde_json::to_string(&response) {
                                        if let Ok(compressed) = compress_message(&response_str) {
                                            ctx_addr.do_send(SendCompressedMessage(compressed));
                                        }
                                    }
                                } else {
                                    error!("GPU compute service not available");
                                    let error_message = json!({
                                        "type": "error",
                                        "message": "GPU compute service not available"
                                    });
                                    if let Ok(error_str) = serde_json::to_string(&error_message) {
                                        if let Ok(compressed) = compress_message(&error_str) {
                                            ctx_addr.do_send(SendCompressedMessage(compressed));
                                        }
                                    }
                                }
                            }.into_actor(self));
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

impl WebSocketSession {
    fn handle_chat_message(&mut self, ctx: &mut ws::WebsocketContext<Self>, message: String, use_openai: bool) {
        let state = self.state.clone();
        let conversation_id = self.conversation_id.clone();
        let ctx_addr = ctx.address();
        let settings = self.state.settings.clone();
        
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
                            Ok(text) => {
                                debug!("Received text response from RAGFlow: {}", text);
                                
                                if use_openai {
                                    debug!("Creating OpenAI WebSocket for TTS");
                                    let openai_ws = OpenAIWebSocket::new(ctx_addr.clone(), settings);
                                    let addr = openai_ws.start();
                                    
                                    debug!("Waiting for OpenAI WebSocket to be ready");
                                    tokio::time::sleep(OPENAI_CONNECT_TIMEOUT).await;
                                    
                                    debug!("Sending text to OpenAI TTS: {}", text);
                                    addr.do_send(OpenAIMessage(text));
                                } else {
                                    debug!("Using local TTS service");
                                    if let Err(e) = state.speech_service.send_message(text).await {
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
        let ctx_addr = ctx.address();
        
        ctx.spawn(async move {
            let mut graph_data = state.graph_data.write().await;
            
            let result = match simulation_mode {
                SimulationMode::Remote => {
                    if let Some(gpu_compute) = &state.gpu_compute {
                        GraphService::calculate_layout(
                            &Some(gpu_compute.clone()),
                            &mut *graph_data,
                            &params
                        ).await
                    } else {
                        GraphService::calculate_layout(
                            &None,
                            &mut *graph_data,
                            &params
                        ).await
                    }
                },
                _ => GraphService::calculate_layout(
                    &None,
                    &mut *graph_data,
                    &params
                ).await,
            };

            match result {
                Ok(_) => {
                    let response = json!({
                        "type": "layout_update",
                        "graph_data": &*graph_data
                    });
                    if let Ok(response_str) = serde_json::to_string(&response) {
                        if let Ok(compressed) = compress_message(&response_str) {
                            ctx_addr.do_send(SendCompressedMessage(compressed));
                        }
                    }
                },
                Err(e) => {
                    error!("Failed to recalculate layout: {}", e);
                    let error_message = json!({
                        "type": "error",
                        "message": format!("Layout calculation failed: {}", e)
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

    fn handle_initial_data(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();
        
        ctx.spawn(async move {
            let graph_data = state.graph_data.read().await;
            let settings = state.settings.read().await;
            
            let response = json!({
                "type": "getInitialData",
                "graph_data": &*graph_data,
                "settings": {
                    "visualization": {
                        "nodeColor": format_color(&settings.visualization.node_color),
                        "edgeColor": format_color(&settings.visualization.edge_color),
                        "hologramColor": format_color(&settings.visualization.hologram_color),
                        "nodeSizeScalingFactor": settings.visualization.node_size_scaling_factor,
                        "hologramScale": settings.visualization.hologram_scale,
                        "hologramOpacity": settings.visualization.hologram_opacity,
                        "edgeOpacity": settings.visualization.edge_opacity,
                        "labelFontSize": settings.visualization.label_font_size,
                        "fogDensity": settings.visualization.fog_density,
                        "forceDirectedIterations": settings.visualization.force_directed_iterations,
                        "forceDirectedRepulsion": settings.visualization.force_directed_repulsion,
                        "forceDirectedAttraction": settings.visualization.force_directed_attraction,
                    },
                    "bloom": {
                        "nodeBloomStrength": settings.bloom.node_bloom_strength,
                        "nodeBloomRadius": settings.bloom.node_bloom_radius,
                        "nodeBloomThreshold": settings.bloom.node_bloom_threshold,
                        "edgeBloomStrength": settings.bloom.edge_bloom_strength,
                        "edgeBloomRadius": settings.bloom.edge_bloom_radius,
                        "edgeBloomThreshold": settings.bloom.edge_bloom_threshold,
                        "environmentBloomStrength": settings.bloom.environment_bloom_strength,
                        "environmentBloomRadius": settings.bloom.environment_bloom_radius,
                        "environmentBloomThreshold": settings.bloom.environment_bloom_threshold,
                    },
                    "fisheye": {
                        "enabled": settings.fisheye.enabled,
                        "strength": settings.fisheye.strength,
                        "focusPoint": settings.fisheye.focus_point,
                        "radius": settings.fisheye.radius,
                    }
                }
            });

            debug!("Sending initial data response: {:?}", response);

            if let Ok(response_str) = serde_json::to_string(&response) {
                if let Ok(compressed) = compress_message(&response_str) {
                    ctx_addr.do_send(SendCompressedMessage(compressed));
                }
            }
        }.into_actor(self));
    }

    // ... (rest of the implementation remains unchanged)
}

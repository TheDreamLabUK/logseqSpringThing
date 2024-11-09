use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use log::{info, error, debug};
use std::sync::{Mutex, Arc};
use serde_json::json;
use futures::stream::StreamExt;
use serde::Deserialize;
use tokio::time::Duration;
use actix_web::web::Bytes;
use bytestring::ByteString;

use crate::AppState;
use crate::models::simulation_params::{SimulationMode, SimulationParams};
use crate::utils::websocket_messages::{
    MessageHandler, OpenAIMessage, ServerMessage,
    OpenAIConnected, OpenAIConnectionFailed, SendText, SendBinary
};
use crate::utils::websocket_openai::OpenAIWebSocket;

const OPENAI_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const GPU_UPDATE_INTERVAL: Duration = Duration::from_millis(50); // 20 fps

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

/// Helper function to convert positions to binary Float32Array data
fn positions_to_binary(positions: &Vec<[f32; 3]>) -> Vec<u8> {
    let mut binary_data = Vec::with_capacity(positions.len() * 12); // 3 floats * 4 bytes each
    for pos in positions {
        // Ensure little-endian byte order for JavaScript Float32Array compatibility
        binary_data.extend_from_slice(&pos[0].to_le_bytes());
        binary_data.extend_from_slice(&pos[1].to_le_bytes());
        binary_data.extend_from_slice(&pos[2].to_le_bytes());
    }
    binary_data
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

        // Start GPU updates if in remote mode
        if matches!(self.simulation_mode, SimulationMode::Remote) {
            if let Some(_) = &self.state.gpu_compute {
                self.start_gpu_updates(ctx);
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
                                    gpu.update_fisheye_params(enabled, strength, focus_point, radius);
                                    
                                    let response = json!({
                                        "type": "fisheye_settings_updated",
                                        "enabled": enabled,
                                        "strength": strength,
                                        "focus_point": focus_point,
                                        "radius": radius
                                    });
                                    if let Ok(response_str) = serde_json::to_string(&response) {
                                        ctx_addr.do_send(SendText(response_str));
                                    }
                                } else {
                                    error!("GPU compute service not available");
                                    let error_message = json!({
                                        "type": "error",
                                        "message": "GPU compute service not available"
                                    });
                                    if let Ok(error_str) = serde_json::to_string(&error_message) {
                                        ctx_addr.do_send(SendText(error_str));
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
                error!("Unexpected binary message received");
                let error_message = json!({
                    "type": "error",
                    "message": "Unexpected binary message"
                });
                self.send_json_response(error_message, ctx);
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
                                            ctx_addr.do_send(SendText(error_str));
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
                                    ctx_addr.do_send(SendText(error_str));
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
                        ctx_addr.do_send(SendText(error_str));
                    }
                }
            }
        }.into_actor(self));
    }

    fn handle_simulation(&mut self, ctx: &mut ws::WebsocketContext<Self>, mode: &str) {
        self.simulation_mode = match mode {
            "remote" => {
                info!("Simulation mode set to Remote (GPU-accelerated)");
                // Start GPU position updates when switching to remote mode
                if let Some(_) = &self.state.gpu_compute {
                    self.start_gpu_updates(ctx);
                }
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
        let ctx_addr = ctx.address();
        
        ctx.spawn(async move {
            if let Some(gpu_compute) = &state.gpu_compute {
                let mut gpu = gpu_compute.write().await;
                
                // Update simulation parameters
                if let Err(e) = gpu.update_simulation_params(&params) {
                    error!("Failed to update simulation parameters: {}", e);
                    let error_message = json!({
                        "type": "error",
                        "message": format!("Failed to update simulation parameters: {}", e)
                    });
                    if let Ok(error_str) = serde_json::to_string(&error_message) {
                        ctx_addr.do_send(SendText(error_str));
                    }
                    return;
                }

                // Run initial GPU computation steps
                for _ in 0..params.iterations {
                    if let Err(e) = gpu.step() {
                        error!("GPU compute step failed: {}", e);
                        let error_message = json!({
                            "type": "error",
                            "message": format!("GPU compute step failed: {}", e)
                        });
                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                            ctx_addr.do_send(SendText(error_str));
                        }
                        return;
                    }
                }

                // Get final positions after layout
                match gpu.get_node_positions().await {
                    Ok(nodes) => {
                        // Convert GPU nodes to position arrays
                        let positions: Vec<[f32; 3]> = nodes.iter()
                            .map(|node| [node.x, node.y, node.z])
                            .collect();

                        // Convert positions to binary Float32Array data
                        let binary_data = positions_to_binary(&positions);

                        // Send binary data
                        ctx_addr.do_send(SendBinary(binary_data));
                    },
                    Err(e) => {
                        error!("Failed to get GPU node positions: {}", e);
                        let error_message = json!({
                            "type": "error",
                            "message": format!("Failed to get GPU node positions: {}", e)
                        });
                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                            ctx_addr.do_send(SendText(error_str));
                        }
                    }
                }
            } else {
                error!("GPU compute service not available");
                let error_message = json!({
                    "type": "error",
                    "message": "GPU compute service not available"
                });
                if let Ok(error_str) = serde_json::to_string(&error_message) {
                    ctx_addr.do_send(SendText(error_str));
                }
            }
        }.into_actor(self));
    }

    fn start_gpu_updates(&self, ctx: &mut ws::WebsocketContext<Self>) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();

        // Start a timer to periodically fetch and broadcast GPU positions
        ctx.run_interval(GPU_UPDATE_INTERVAL, move |_act, _ctx| {
            let state_clone = state.clone();
            let addr_clone = ctx_addr.clone();

            actix::spawn(async move {
                if let Some(gpu_compute) = &state_clone.gpu_compute {
                    let mut gpu = gpu_compute.write().await;
                    
                    // Run one step of GPU computation
                    if let Err(e) = gpu.step() {
                        error!("GPU compute step failed: {}", e);
                        return;
                    }

                    // Get updated positions
                    match gpu.get_node_positions().await {
                        Ok(nodes) => {
                            // Convert GPU nodes to position arrays
                            let positions: Vec<[f32; 3]> = nodes.iter()
                                .map(|node| [node.x, node.y, node.z])
                                .collect();

                            // Convert positions to binary Float32Array data
                            let binary_data = positions_to_binary(&positions);

                            // Send binary data
                            addr_clone.do_send(SendBinary(binary_data));
                        },
                        Err(e) => {
                            error!("Failed to get GPU node positions: {}", e);
                        }
                    }
                }
            });
        });
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
                ctx_addr.do_send(SendText(response_str));
            }
        }.into_actor(self));
    }
}

// Implement Handler traits for WebSocketSession
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

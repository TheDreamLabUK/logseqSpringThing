use actix::prelude::*;
use actix_web::web;
use actix_web_actors::ws::WebsocketContext;
use bytestring::ByteString;
use bytemuck;
use futures::StreamExt;
use log::{debug, error, info, warn};
use serde_json::json;
use std::sync::{Arc, Mutex};
use tokio::time::Duration;
use actix_web_actors::ws;
use actix::StreamHandler;

use crate::AppState;
use crate::models::node::GPUNode;
use crate::models::simulation_params::{SimulationMode, SimulationParams};
use crate::models::position_update::NodePositionVelocity;
use crate::utils::websocket_messages::{
    OpenAIMessage, SendBinary, SendText,
    ServerMessage,
};
use crate::utils::websocket_openai::OpenAIWebSocket;

pub const OPENAI_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const GPU_UPDATE_INTERVAL: Duration = Duration::from_millis(16);

// Helper function to convert positions to binary data
fn positions_to_binary(nodes: &[GPUNode]) -> Vec<u8> {
    let mut binary_data = Vec::with_capacity(nodes.len() * std::mem::size_of::<NodePositionVelocity>());
    for node in nodes {
        let position = NodePositionVelocity {
            x: node.x,
            y: node.y,
            z: node.z,
            vx: node.vx,
            vy: node.vy,
            vz: node.vz,
        };
        binary_data.extend_from_slice(bytemuck::bytes_of(&position));
    }
    binary_data
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct GpuUpdate;

pub struct WebSocketSession {
    pub state: web::Data<AppState>,
    pub tts_method: String,
    pub openai_ws: Option<Addr<OpenAIWebSocket>>,
    pub simulation_mode: SimulationMode,
    pub conversation_id: Option<Arc<Mutex<Option<String>>>>,
}

// Implement Handler for SendText
impl Handler<SendText> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}

// Implement Handler for SendBinary
impl Handler<SendBinary> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendBinary, ctx: &mut Self::Context) {
        ctx.binary(msg.0);
    }
}

// Implement Handler for GpuUpdate
impl Handler<GpuUpdate> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, _: GpuUpdate, ctx: &mut Self::Context) {
        if let Some(gpu_compute) = &self.state.gpu_compute {
            let gpu_compute = gpu_compute.clone();
            let ctx_addr = ctx.address();
            
            actix::spawn(async move {
                let gpu = gpu_compute.read().await;
                if let Ok(nodes) = gpu.get_node_positions().await {
                    let binary_data = positions_to_binary(&nodes);
                    ctx_addr.do_send(SendBinary(binary_data));
                }
            });
        }
    }
}

impl WebSocketSession {
    pub fn new(state: web::Data<AppState>) -> Self {
        Self {
            state,
            tts_method: String::from("local"),
            openai_ws: None,
            simulation_mode: SimulationMode::Remote,
            conversation_id: Some(Arc::new(Mutex::new(None))),
        }
    }

    fn validate_binary_data(&self, data: &[u8]) -> bool {
        let node_size = std::mem::size_of::<NodePositionVelocity>();
        if data.len() % node_size != 0 {
            warn!(
                "Invalid binary data length: {} (not a multiple of {})",
                data.len(),
                node_size
            );
            return false;
        }
        true
    }

    fn process_binary_update(&mut self, data: &[u8]) -> Result<(), String> {
        if !self.validate_binary_data(data) {
            return Err("Invalid binary data format".to_string());
        }

        let positions: Vec<NodePositionVelocity> = bytemuck::cast_slice(data).to_vec();
        if positions.is_empty() {
            warn!("Received empty positions array");
            return Ok(());
        }

        let state = self.state.clone();
        let positions = positions.clone();

        actix::spawn(async move {
            let mut graph_data = state.graph_data.write().await;
            for (i, pos) in positions.iter().enumerate() {
                if i < graph_data.nodes.len() {
                    graph_data.nodes[i].x = pos.x;
                    graph_data.nodes[i].y = pos.y;
                    graph_data.nodes[i].z = pos.z;
                    graph_data.nodes[i].vx = pos.vx;
                    graph_data.nodes[i].vy = pos.vy;
                    graph_data.nodes[i].vz = pos.vz;
                }
            }
            debug!("Updated {} node positions", positions.len());
        });

        Ok(())
    }
}

impl Actor for WebSocketSession {
    type Context = WebsocketContext<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("WebSocket session started");
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("WebSocket session stopped");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                debug!("Ping received");
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                debug!("Pong received");
            }
            Ok(ws::Message::Text(text)) => {
                debug!("Text message received: {}", text);
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                    match value.get("type").and_then(|t| t.as_str()) {
                        Some("chat") => {
                            if let Some(message) = value.get("message").and_then(|m| m.as_str()) {
                                let use_openai = value.get("useOpenAI")
                                    .and_then(|o| o.as_bool())
                                    .unwrap_or(false);
                                self.handle_chat_message(ctx, message.to_string(), use_openai);
                            }
                        }
                        Some("simulation_mode") => {
                            if let Some(mode) = value.get("mode").and_then(|m| m.as_str()) {
                                self.handle_simulation_mode(ctx, mode);
                            }
                        }
                        Some("layout") => {
                            if let Ok(params) = serde_json::from_value::<SimulationParams>(value["params"].clone()) {
                                self.handle_layout(ctx, params);
                            }
                        }
                        Some("fisheye") => {
                            let enabled = value.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false);
                            let strength = value.get("strength").and_then(|s| s.as_f64()).unwrap_or(1.0) as f32;
                            let focus_point = value.get("focusPoint")
                                .and_then(|f| f.as_array())
                                .and_then(|arr| {
                                    if arr.len() == 3 {
                                        Some([
                                            arr[0].as_f64().unwrap_or(0.0) as f32,
                                            arr[1].as_f64().unwrap_or(0.0) as f32,
                                            arr[2].as_f64().unwrap_or(0.0) as f32,
                                        ])
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or([0.0, 0.0, 0.0]);
                            let radius = value.get("radius").and_then(|r| r.as_f64()).unwrap_or(1.0) as f32;
                            self.handle_fisheye_settings(ctx, enabled, strength, focus_point, radius);
                        }
                        Some("initial_data") => {
                            self.handle_initial_data(ctx);
                        }
                        _ => {
                            error!("Unknown message type received");
                            let error_message = ServerMessage::Error {
                                message: "Unknown message type".to_string(),
                                code: Some("UNKNOWN_MESSAGE_TYPE".to_string()),
                                details: Some("The received message type is not recognized by the server".to_string()),
                            };
                            if let Ok(error_str) = serde_json::to_string(&error_message) {
                                ctx.text(ByteString::from(error_str));
                            }
                        }
                    }
                }
            }
            Ok(ws::Message::Binary(bin)) => {
                debug!("Binary message received: {} bytes", bin.len());
                match self.process_binary_update(&bin) {
                    Ok(_) => {
                        debug!("Binary update processed successfully");
                    },
                    Err(e) => {
                        error!("Failed to process binary update: {}", e);
                        let error_message = ServerMessage::Error {
                            message: format!("Binary update processing failed: {}", e),
                            code: Some("BINARY_UPDATE_ERROR".to_string()),
                            details: Some("Error occurred while processing binary position update data".to_string()),
                        };
                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                            ctx.text(ByteString::from(error_str));
                        }
                    }
                }
            }
            Ok(ws::Message::Close(reason)) => {
                info!("Client disconnected: {:?}", reason);
                ctx.close(reason);
                ctx.stop();
            }
            Ok(ws::Message::Continuation(_)) => {
                debug!("Continuation frame received");
            }
            Ok(ws::Message::Nop) => {
                debug!("Nop frame received");
            }
            Err(e) => {
                error!("Error in WebSocket message handling: {}", e);
                ctx.stop();
            }
        }
    }
}

// WebSocketSessionHandler Trait
pub trait WebSocketSessionHandler {
    fn start_gpu_updates(&self, ctx: &mut WebsocketContext<WebSocketSession>);
    fn handle_chat_message(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, message: String, use_openai: bool);
    fn handle_simulation_mode(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, mode: &str);
    fn handle_layout(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, params: SimulationParams);
    fn handle_initial_data(&mut self, ctx: &mut WebsocketContext<WebSocketSession>);
    fn handle_fisheye_settings(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, enabled: bool, strength: f32, focus_point: [f32; 3], radius: f32);
}

// WebSocketSessionHandler Implementation
impl WebSocketSessionHandler for WebSocketSession {
    fn start_gpu_updates(&self, ctx: &mut WebsocketContext<WebSocketSession>) {
        let addr = ctx.address();
        ctx.run_interval(GPU_UPDATE_INTERVAL, move |_, _| {
            addr.do_send(GpuUpdate);
        });
    }

    fn handle_chat_message(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, message: String, use_openai: bool) {
        let state = self.state.clone();
        let conversation_id = self.conversation_id.clone();
        let ctx_addr = ctx.address();
        let settings = self.state.settings.clone();
        let weak_addr = ctx.address().downgrade();

        let fut = async move {
            let conv_id = if let Some(conv_arc) = conversation_id {
                let mut lock = conv_arc.lock().unwrap();
                if let Some(ref id) = *lock {
                    id.clone()
                } else {
                    match state.ragflow_service.create_conversation("default_user".to_string()).await {
                        Ok(new_id) => {
                            *lock = Some(new_id.clone());
                            new_id
                        },
                        Err(e) => {
                            error!("Failed to create conversation: {}", e);
                            return;
                        }
                    }
                }
            } else {
                error!("Failed to acquire conversation ID");
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
                                        let error_message = ServerMessage::Error {
                                            message: format!("Failed to generate speech: {}", e),
                                            code: Some("SPEECH_GENERATION_ERROR".to_string()),
                                            details: Some("Error occurred while generating speech using local TTS service".to_string()),
                                        };
                                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                                            ctx_addr.do_send(SendText(error_str));
                                        }
                                    }
                                }
                            },
                            Err(e) => {
                                error!("Error processing RAGFlow response: {}", e);
                                let error_message = ServerMessage::Error {
                                    message: format!("Error processing RAGFlow response: {}", e),
                                    code: Some("RAGFLOW_PROCESSING_ERROR".to_string()),
                                    details: Some("Failed to process the response from RAGFlow service".to_string()),
                                };
                                if let Ok(error_str) = serde_json::to_string(&error_message) {
                                    ctx_addr.do_send(SendText(error_str));
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    error!("Failed to send message to RAGFlow: {}", e);
                    let error_message = ServerMessage::Error {
                        message: format!("Failed to send message to RAGFlow: {}", e),
                        code: Some("RAGFLOW_SEND_ERROR".to_string()),
                        details: Some("Error occurred while sending message to RAGFlow service".to_string()),
                    };
                    if let Ok(error_str) = serde_json::to_string(&error_message) {
                        ctx_addr.do_send(SendText(error_str));
                    }
                }
            }

            if let Some(addr) = weak_addr.upgrade() {
                let completion = json!({
                    "type": "completion",
                    "message": "Chat message handled"
                });
                if let Ok(completion_str) = serde_json::to_string(&completion) {
                    addr.do_send(SendText(completion_str));
                }
            }
        };

        ctx.spawn(fut.into_actor(self));
    }

    fn handle_simulation_mode(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, mode: &str) {
        self.simulation_mode = match mode {
            "remote" => {
                info!("Simulation mode set to Remote (GPU-accelerated)");
                if self.state.gpu_compute.is_some() {
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

        let response = ServerMessage::SimulationModeSet {
            mode: mode.to_string(),
            gpu_enabled: matches!(self.simulation_mode, SimulationMode::Remote | SimulationMode::GPU),
        };
        if let Ok(response_str) = serde_json::to_string(&response) {
            ctx.text(ByteString::from(response_str));
        }
    }

    fn handle_layout(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, params: SimulationParams) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();
        let weak_addr = ctx.address().downgrade();

        let fut = async move {
            if let Some(gpu_compute) = &state.gpu_compute {
                let mut gpu = gpu_compute.write().await;
                
                if let Err(e) = gpu.update_simulation_params(&params) {
                    error!("Failed to update simulation parameters: {}", e);
                    let error_message = ServerMessage::Error {
                        message: format!("Failed to update simulation parameters: {}", e),
                        code: Some("SIMULATION_PARAMS_ERROR".to_string()),
                        details: Some("Error occurred while updating GPU simulation parameters".to_string()),
                    };
                    if let Ok(error_str) = serde_json::to_string(&error_message) {
                        ctx_addr.do_send(SendText(error_str));
                    }
                    return;
                }

                for _ in 0..params.iterations {
                    if let Err(e) = gpu.step() {
                        error!("GPU compute step failed: {}", e);
                        let error_message = ServerMessage::Error {
                            message: format!("GPU compute step failed: {}", e),
                            code: Some("GPU_COMPUTE_ERROR".to_string()),
                            details: Some("Error occurred during GPU computation step".to_string()),
                        };
                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                            ctx_addr.do_send(SendText(error_str));
                        }
                        return;
                    }
                }

                match gpu.get_node_positions().await {
                    Ok(nodes) => {
                        let binary_data = positions_to_binary(&nodes);
                        ctx_addr.do_send(SendBinary(binary_data));
                    },
                    Err(e) => {
                        error!("Failed to get GPU node positions: {}", e);
                        let error_message = ServerMessage::Error {
                            message: format!("Failed to get GPU node positions: {}", e),
                            code: Some("GPU_POSITION_ERROR".to_string()),
                            details: Some("Error occurred while retrieving node positions from GPU".to_string()),
                        };
                        if let Ok(error_str) = serde_json::to_string(&error_message) {
                            ctx_addr.do_send(SendText(error_str));
                        }
                    }
                }
            } else {
                error!("GPU compute service not available");
                let error_message = ServerMessage::Error {
                    message: "GPU compute service not available".to_string(),
                    code: Some("GPU_SERVICE_ERROR".to_string()),
                    details: Some("The GPU compute service is not initialized or unavailable".to_string()),
                };
                if let Ok(error_str) = serde_json::to_string(&error_message) {
                    ctx_addr.do_send(SendText(error_str));
                }
            }

            if let Some(addr) = weak_addr.upgrade() {
                let completion = json!({
                    "type": "completion",
                    "message": "Layout update complete"
                });
                if let Ok(completion_str) = serde_json::to_string(&completion) {
                    addr.do_send(SendText(completion_str));
                }
            }
        };

        ctx.spawn(fut.into_actor(self));
    }

    fn handle_initial_data(&mut self, ctx: &mut WebsocketContext<WebSocketSession>) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();

        let fut = async move {
            info!("Handling initial_data request");
            
            let graph_data = match state.graph_data.try_read() {
                Ok(data) => {
                    info!("Successfully acquired graph data read lock");
                    info!("Current graph state: {} nodes, {} edges, {} metadata entries",
                        data.nodes.len(),
                        data.edges.len(),
                        data.metadata.len()
                    );
                    data
                },
                Err(e) => {
                    error!("Failed to acquire graph data read lock: {}", e);
                    return;
                }
            };

            let settings = match state.settings.try_read() {
                Ok(s) => {
                    info!("Successfully acquired settings read lock");
                    s
                },
                Err(e) => {
                    error!("Failed to acquire settings read lock: {}", e);
                    return;
                }
            };

            info!("Preparing graph update message");
            let graph_update = ServerMessage::GraphUpdate {
                graph_data: serde_json::to_value(&*graph_data).unwrap_or_default(),
            };

            info!("Sending graph data to client");
            if let Ok(graph_str) = serde_json::to_string(&graph_update) {
                debug!("Graph data JSON size: {} bytes", graph_str.len());
                ctx_addr.do_send(SendText(graph_str));
            } else {
                error!("Failed to serialize graph data");
            }

            // Prepare and send settings update
            info!("Preparing settings update");
            let settings_update = ServerMessage::SettingsUpdated {
                settings: serde_json::to_value(&*settings).unwrap_or_default(),
            };

            info!("Sending settings to client");
            if let Ok(settings_str) = serde_json::to_string(&settings_update) {
                debug!("Settings JSON size: {} bytes", settings_str.len());
                ctx_addr.do_send(SendText(settings_str));
            } else {
                error!("Failed to serialize settings");
            }

            let completion = json!({
                "type": "completion",
                "message": "Initial data sent"
            });
            if let Ok(completion_str) = serde_json::to_string(&completion) {
                ctx_addr.do_send(SendText(completion_str));
            }
        };

        ctx.spawn(fut.into_actor(self));

        self.simulation_mode = SimulationMode::Remote;
        if self.state.gpu_compute.is_some() {
            info!("Starting GPU updates");
            self.start_gpu_updates(ctx);
        } else {
            warn!("GPU compute not available");
        }
    }

    fn handle_fisheye_settings(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, enabled: bool, strength: f32, focus_point: [f32; 3], radius: f32) {
        // TODO: Remove server-side fisheye handling
        // Fisheye effect should be purely client-side in the visualization layer
        // This handler is temporarily disabled until proper client-side implementation
        
        let ctx_addr = ctx.address();
        let completion = json!({
            "type": "completion",
            "message": "Fisheye settings acknowledged (to be handled client-side)"
        });
        if let Ok(completion_str) = serde_json::to_string(&completion) {
            ctx_addr.do_send(SendText(completion_str));
        }
    }
}

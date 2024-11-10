use actix::prelude::*;
use actix_web::web;
use actix_web_actors::ws::WebsocketContext;
use log::{info, error, debug};
use serde_json::json;
use tokio::time::Duration;
use std::sync::{Arc, Mutex};
use futures::StreamExt;
use bytes::Bytes;
use bytestring::ByteString;

use crate::AppState;
use crate::models::simulation_params::{SimulationMode, SimulationParams};
use crate::utils::websocket_messages::{SendText, SendBinary, OpenAIMessage, MessageHandler, OpenAIConnected, OpenAIConnectionFailed};
use crate::utils::websocket_openai::OpenAIWebSocket;

pub const OPENAI_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const GPU_UPDATE_INTERVAL: Duration = Duration::from_millis(16); // ~60fps for smooth updates

// Message for GPU updates
#[derive(Message)]
#[rtype(result = "()")]
pub struct GpuUpdate;

/// WebSocket session actor.
pub struct WebSocketSession {
    pub state: web::Data<AppState>,
    pub tts_method: String,
    pub openai_ws: Option<Addr<OpenAIWebSocket>>,
    pub simulation_mode: SimulationMode,
    pub conversation_id: Option<Arc<Mutex<Option<String>>>>,
}

impl Actor for WebSocketSession {
    type Context = WebsocketContext<Self>;
}

impl MessageHandler for WebSocketSession {}

/// Helper function to convert hex color to proper format
pub fn format_color(color: &str) -> String {
    let color = color.trim_matches('"')
        .trim_start_matches("0x")
        .trim_start_matches('#');
    format!("#{}", color)
}

/// Helper function to convert positions to binary Float32Array data
pub fn positions_to_binary(positions: &Vec<[f32; 3]>) -> Vec<u8> {
    let mut binary_data = Vec::with_capacity(positions.len() * 12); // 3 floats * 4 bytes each
    for pos in positions {
        // Ensure little-endian byte order for JavaScript Float32Array compatibility
        binary_data.extend_from_slice(&pos[0].to_le_bytes());
        binary_data.extend_from_slice(&pos[1].to_le_bytes());
        binary_data.extend_from_slice(&pos[2].to_le_bytes());
    }
    binary_data
}

pub trait WebSocketSessionHandler {
    fn start_gpu_updates(&self, ctx: &mut WebsocketContext<WebSocketSession>);
    fn handle_chat_message(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, message: String, use_openai: bool);
    fn handle_simulation_mode(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, mode: &str);
    fn handle_layout(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, params: SimulationParams);
    fn handle_initial_data(&mut self, ctx: &mut WebsocketContext<WebSocketSession>);
    fn handle_fisheye_settings(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, enabled: bool, strength: f32, focus_point: [f32; 3], radius: f32);
}

impl Handler<GpuUpdate> for WebSocketSession {
    type Result = ResponseActFuture<Self, ()>;

    fn handle(&mut self, _: GpuUpdate, ctx: &mut Self::Context) -> Self::Result {
        let state = self.state.clone();
        let gpu_compute = if let Some(gpu) = &state.gpu_compute {
            gpu.clone()
        } else {
            return Box::pin(futures::future::ready(()).into_actor(self));
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
        .into_actor(self))
    }
}

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

            // Only send completion message if the actor is still alive
            if let Some(addr) = weak_addr.upgrade() {
                addr.do_send(SendText("Chat message handled".to_string()));
            }
        };

        ctx.spawn(fut.into_actor(self));
    }

    fn handle_simulation_mode(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, mode: &str) {
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
        <Self as MessageHandler>::send_json_response(self, response, ctx);
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
                    let error_message = json!({
                        "type": "error",
                        "message": format!("Failed to update simulation parameters: {}", e)
                    });
                    if let Ok(error_str) = serde_json::to_string(&error_message) {
                        ctx_addr.do_send(SendText(error_str));
                    }
                    return;
                }

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

                match gpu.get_node_positions().await {
                    Ok(nodes) => {
                        let positions: Vec<[f32; 3]> = nodes.iter()
                            .map(|node| [node.x, node.y, node.z])
                            .collect();

                        let binary_data = positions_to_binary(&positions);

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

            // Only send completion message if the actor is still alive
            if let Some(addr) = weak_addr.upgrade() {
                addr.do_send(SendText("Layout update complete".to_string()));
            }
        };

        ctx.spawn(fut.into_actor(self));
    }

    fn handle_initial_data(&mut self, ctx: &mut WebsocketContext<WebSocketSession>) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();
        let weak_addr = ctx.address().downgrade();

        let fut = async move {
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

            // Only send completion message if the actor is still alive
            if let Some(addr) = weak_addr.upgrade() {
                addr.do_send(SendText("Initial data sent".to_string()));
            }
        };

        ctx.spawn(fut.into_actor(self));
    }

    fn handle_fisheye_settings(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, enabled: bool, strength: f32, focus_point: [f32; 3], radius: f32) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();
        let weak_addr = ctx.address().downgrade();

        let fut = async move {
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

            // Only send completion message if the actor is still alive
            if let Some(addr) = weak_addr.upgrade() {
                addr.do_send(SendText("Fisheye settings updated".to_string()));
            }
        };

        ctx.spawn(fut.into_actor(self));
    }
}

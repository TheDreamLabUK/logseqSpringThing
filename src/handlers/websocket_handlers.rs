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
use crate::models::simulation_params::{SimulationMode, SimulationParams};
use crate::models::position_update::NodePositionVelocity;
use crate::utils::websocket_messages::{
    OpenAIMessage, SendBinary, SendText,
    ServerMessage, UpdatePositionsMessage,
};
use crate::utils::websocket_openai::OpenAIWebSocket;

pub const OPENAI_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const GPU_UPDATE_INTERVAL: Duration = Duration::from_millis(16);

// Constants for binary protocol
const NODE_SIZE: usize = 6 * std::mem::size_of::<f32>(); // x, y, z, vx, vy, vz

// Helper function to convert positions to binary data
fn positions_to_binary(nodes: &[NodePositionVelocity]) -> Vec<u8> {
    let mut binary_data = Vec::with_capacity(nodes.len() * std::mem::size_of::<NodePositionVelocity>());
    for node in nodes {
        binary_data.extend_from_slice(bytemuck::bytes_of(node));
    }
    binary_data
}

// Helper function to send binary position update
fn send_binary_update(ctx: &mut WebsocketContext<WebSocketSession>, binary_data: Vec<u8>, is_initial: bool) {
    // Send message indicating binary update type
    let update_type = ServerMessage::BinaryPositionUpdate {
        is_initial_layout: is_initial
    };
    
    if let Ok(type_str) = serde_json::to_string(&update_type) {
        ctx.text(ByteString::from(type_str));
    }
    
    // Send the binary data
    ctx.binary(binary_data);
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
            let mut graph = state.graph_service.graph_data.write().await;
            for (i, pos) in positions.iter().enumerate() {
                if i < graph.nodes.len() {
                    graph.nodes[i].x = pos.x;
                    graph.nodes[i].y = pos.y;
                    graph.nodes[i].z = pos.z;
                    graph.nodes[i].vx = pos.vx;
                    graph.nodes[i].vy = pos.vy;
                    graph.nodes[i].vz = pos.vz;
                    graph.nodes[i].position = Some([pos.x, pos.y, pos.z]);
                }
            }
            debug!("Updated {} node positions", positions.len());
        });

        Ok(())
    }

    fn handle_position_update(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, msg: UpdatePositionsMessage) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();

        let fut = async move {
            debug!("Processing position update for {} nodes", msg.nodes.len());

            // Update graph data with new positions
            {
                let mut graph = state.graph_service.graph_data.write().await;
                for node_pos in &msg.nodes {
                    if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_pos.id) {
                        node.x = node_pos.position[0];
                        node.y = node_pos.position[1];
                        node.z = node_pos.position[2];
                        node.vx = 0.0;
                        node.vy = 0.0;
                        node.vz = 0.0;
                        node.position = Some(node_pos.position);
                    }
                }
            }

            // Send completion message
            let completion = ServerMessage::PositionUpdateComplete {
                status: "success".to_string(),
            };
            if let Ok(completion_str) = serde_json::to_string(&completion) {
                ctx_addr.do_send(SendText(completion_str));
            }
        };

        ctx.spawn(fut.into_actor(self));
    }

    fn handle_layout(&mut self, ctx: &mut WebsocketContext<WebSocketSession>, _params: SimulationParams) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();

        let fut = async move {
            let graph = state.graph_service.graph_data.write().await;
            
            // Update positions
            let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                x: node.x,
                y: node.y,
                z: node.z,
                vx: node.vx,
                vy: node.vy,
                vz: node.vz,
            }).collect();
            
            let binary_data = positions_to_binary(&nodes);
            
            // Send binary update type first
            let update_type = ServerMessage::BinaryPositionUpdate {
                is_initial_layout: true
            };
            
            if let Ok(type_str) = serde_json::to_string(&update_type) {
                ctx_addr.do_send(SendText(type_str));
            }
            
            // Then send binary data
            ctx_addr.do_send(SendBinary(binary_data));

            // Send completion message
            let completion = json!({
                "type": "completion",
                "message": "Layout update complete"
            });
            if let Ok(completion_str) = serde_json::to_string(&completion) {
                ctx_addr.do_send(SendText(completion_str));
            }
        };

        ctx.spawn(fut.into_actor(self));
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
                                    // Get read lock on speech service
                                    if let Some(speech_service) = state.get_speech_service().await {
                                        if let Err(e) = speech_service.send_message(text).await {
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
                                    } else {
                                        error!("Speech service not initialized");
                                        let error_message = ServerMessage::Error {
                                            message: "Speech service not initialized".to_string(),
                                            code: Some("SPEECH_SERVICE_ERROR".to_string()),
                                            details: Some("Speech service is not available".to_string()),
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
                self.start_gpu_updates(ctx);
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

    fn start_gpu_updates(&self, ctx: &mut WebsocketContext<WebSocketSession>) {
        let addr = ctx.address();
        ctx.run_interval(GPU_UPDATE_INTERVAL, move |_, _| {
            addr.do_send(GpuUpdate);
        });
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
                        Some("updatePositions") => {
                            if let Ok(update_msg) = serde_json::from_value::<UpdatePositionsMessage>(value) {
                                self.handle_position_update(ctx, update_msg);
                            }
                        }
                        Some("chat") => {
                            if let Some(message) = value.get("message").and_then(|m| m.as_str()) {
                                let use_openai = value.get("use_openai")
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
                        Some("initial_data") => {
                            let addr = ctx.address();
                            let state = self.state.clone();
                            actix::spawn(async move {
                                let graph = state.graph_service.graph_data.read().await;
                                let settings = state.settings.read().await;

                                let initial_data = ServerMessage::InitialData {
                                    graph_data: (*graph).clone(),
                                    settings: serde_json::to_value(&*settings).unwrap_or_default(),
                                };

                                if let Ok(initial_data_str) = serde_json::to_string(&initial_data) {
                                    addr.do_send(SendText(initial_data_str));
                                }
                            });

                            self.simulation_mode = SimulationMode::Remote;
                            self.start_gpu_updates(ctx);
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

impl Handler<SendBinary> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendBinary, ctx: &mut Self::Context) {
        let num_nodes = msg.0.len() / NODE_SIZE;
        debug!("[WebSocketSession] Sending binary message: {} nodes, size={} bytes", 
            num_nodes, msg.0.len());
        
        // Send binary update type first
        let update_type = ServerMessage::BinaryPositionUpdate {
            is_initial_layout: false
        };
        
        if let Ok(type_str) = serde_json::to_string(&update_type) {
            ctx.text(ByteString::from(type_str));
        }
        
        // Then send binary data
        ctx.binary(msg.0);
    }
}

impl Handler<SendText> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: SendText, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}

impl Handler<GpuUpdate> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, _: GpuUpdate, ctx: &mut Self::Context) {
        let state = self.state.clone();
        let ctx_addr = ctx.address();
        
        actix::spawn(async move {
            let graph = state.graph_service.graph_data.read().await;
            let nodes: Vec<NodePositionVelocity> = graph.nodes.iter().map(|node| NodePositionVelocity {
                x: node.x,
                y: node.y,
                z: node.z,
                vx: node.vx,
                vy: node.vy,
                vz: node.vz,
            }).collect();
            
            let binary_data = positions_to_binary(&nodes);
            ctx_addr.do_send(SendBinary(binary_data));
        });
    }
}

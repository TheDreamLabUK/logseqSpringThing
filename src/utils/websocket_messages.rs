use actix::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::models::simulation_params::SimulationParams;
use actix_web_actors::ws;
use log::{error, debug};
use actix_web::web::Bytes;
use bytestring::ByteString;

/// Helper function to convert hex color to proper format
fn format_color(color: &str) -> String {
    let color = color.trim_matches('"')
        .trim_start_matches("0x")
        .trim_start_matches('#');
    format!("#{}", color)
}

/// GPU-computed node positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUPositionUpdate {
    pub positions: Vec<[f32; 3]>,
    pub timestamp: u64,
}

/// Message for sending text data
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendText(pub String);

/// Message for sending binary data
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendBinary(pub Vec<u8>);

/// Message for OpenAI text-to-speech
#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIMessage(pub String);

/// Message indicating OpenAI connection success
#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnected;

/// Message indicating OpenAI connection failure
#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnectionFailed;

/// Represents messages sent from the client.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "setTtsMethod")]
    SetTTSMethod { method: String },
    
    #[serde(rename = "chatMessage")]
    ChatMessage { 
        message: String, 
        use_openai: bool 
    },
    
    #[serde(rename = "getInitialData")]
    GetInitialData,
    
    #[serde(rename = "setSimulationMode")]
    SetSimulationMode { mode: String },
    
    #[serde(rename = "recalculateLayout")]
    RecalculateLayout { params: SimulationParams },
    
    #[serde(rename = "ragflowQuery")]
    RagflowQuery {
        message: String,
        quote: Option<bool>,
        doc_ids: Option<Vec<String>>
    },
    
    #[serde(rename = "openaiQuery")]
    OpenAIQuery { message: String },

    #[serde(rename = "updateFisheyeSettings")]
    UpdateFisheyeSettings {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    }
}

/// Represents messages sent from the server to the client.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "audioData")]
    AudioData {
        audio_data: String // base64 encoded audio
    },
    
    #[serde(rename = "ragflowResponse")]
    RagflowResponse {
        answer: String
    },
    
    #[serde(rename = "openaiResponse")]
    OpenAIResponse {
        response: String,
        audio: Option<String> // base64 encoded audio
    },
    
    #[serde(rename = "error")]
    Error {
        message: String,
        code: Option<String>
    },
    
    #[serde(rename = "graphUpdate")]
    GraphUpdate {
        graph_data: Value
    },
    
    #[serde(rename = "simulationModeSet")]
    SimulationModeSet {
        mode: String,
        gpu_enabled: bool
    },

    #[serde(rename = "fisheyeSettingsUpdated")]
    FisheyeSettingsUpdated {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    },

    #[serde(rename = "gpuPositions")]
    GPUPositions(GPUPositionUpdate)
}

pub trait MessageHandler: Actor<Context = ws::WebsocketContext<Self>> {
    fn send_json_response(&self, response: Value, ctx: &mut ws::WebsocketContext<Self>) {
        match serde_json::to_string(&response) {
            Ok(json_string) => {
                debug!("Sending JSON response: {}", json_string);
                ctx.text(ByteString::from(json_string));
            },
            Err(e) => {
                error!("Failed to serialize JSON response: {}", e);
                let error_message = json!({
                    "type": "error",
                    "message": format!("Failed to serialize response: {}", e),
                    "code": "SERIALIZATION_ERROR"
                });
                if let Ok(error_string) = serde_json::to_string(&error_message) {
                    ctx.text(ByteString::from(error_string));
                }
            }
        }
    }

    fn send_server_message(&self, message: ServerMessage, ctx: &mut ws::WebsocketContext<Self>) {
        match serde_json::to_value(message) {
            Ok(value) => self.send_json_response(value, ctx),
            Err(e) => {
                error!("Failed to serialize ServerMessage: {}", e);
                let error_message = json!({
                    "type": "error",
                    "message": "Internal server error",
                    "code": "SERIALIZATION_ERROR"
                });
                self.send_json_response(error_message, ctx);
            }
        }
    }

    fn handle_fisheye_update(&self, settings: ClientMessage, gpu_compute: &mut crate::utils::gpu_compute::GPUCompute, ctx: &mut ws::WebsocketContext<Self>) {
        if let ClientMessage::UpdateFisheyeSettings { enabled, strength, focus_point, radius } = settings {
            match gpu_compute.update_fisheye_params(enabled, strength, focus_point, radius) {
                Ok(_) => {
                    // Send confirmation back to client
                    self.send_server_message(
                        ServerMessage::FisheyeSettingsUpdated {
                            enabled,
                            strength,
                            focus_point,
                            radius,
                        },
                        ctx
                    );
                },
                Err(e) => {
                    error!("Failed to update fisheye settings: {}", e);
                    self.send_server_message(
                        ServerMessage::Error {
                            message: format!("Failed to update fisheye settings: {}", e),
                            code: Some("FISHEYE_UPDATE_ERROR".to_string()),
                        },
                        ctx
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_message_serialization() {
        let message = ClientMessage::ChatMessage {
            message: "Hello".to_string(),
            use_openai: true
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert!(serialized.contains("chatMessage"));
        assert!(serialized.contains("Hello"));

        let fisheye_message = ClientMessage::UpdateFisheyeSettings {
            enabled: true,
            strength: 0.5,
            focus_point: [0.0, 0.0, 0.0],
            radius: 100.0,
        };
        let serialized = serde_json::to_string(&fisheye_message).unwrap();
        assert!(serialized.contains("updateFisheyeSettings"));
        assert!(serialized.contains("strength"));
    }

    #[test]
    fn test_server_message_serialization() {
        let message = ServerMessage::RagflowResponse {
            answer: "Test answer".to_string()
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert!(serialized.contains("ragflowResponse"));
        assert!(serialized.contains("Test answer"));

        let fisheye_message = ServerMessage::FisheyeSettingsUpdated {
            enabled: true,
            strength: 0.5,
            focus_point: [0.0, 0.0, 0.0],
            radius: 100.0,
        };
        let serialized = serde_json::to_string(&fisheye_message).unwrap();
        assert!(serialized.contains("fisheyeSettingsUpdated"));
        assert!(serialized.contains("strength"));
    }

    #[test]
    fn test_color_formatting() {
        assert_eq!(format_color("0xFF0000"), "#FF0000");
        assert_eq!(format_color("\"0xFF0000\""), "#FF0000");
        assert_eq!(format_color("#FF0000"), "#FF0000");
        assert_eq!(format_color("\"#FF0000\""), "#FF0000");
    }
}

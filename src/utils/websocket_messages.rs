use actix::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::models::simulation_params::SimulationParams;
use actix_web_actors::ws;
use log::{error, debug};
use crate::utils::compression::compress_message;

/// Helper function to convert hex color to proper format
fn format_color(color: &str) -> String {
    let color = color.trim_matches('"')
        .trim_start_matches("0x")
        .trim_start_matches('#');
    format!("#{}", color)
}

/// Represents messages sent to the client as compressed binary data.
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendCompressedMessage(pub Vec<u8>);

/// Represents messages sent from the client.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "set_tts_method")]
    SetTTSMethod { method: String },
    
    #[serde(rename = "chat_message")]
    ChatMessage { 
        message: String, 
        use_openai: bool 
    },
    
    #[serde(rename = "get_initial_data")]
    GetInitialData,
    
    #[serde(rename = "set_simulation_mode")]
    SetSimulationMode { mode: String },
    
    #[serde(rename = "recalculate_layout")]
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
    #[serde(rename = "audio_data")]
    AudioData {
        audio_data: String // base64 encoded audio
    },
    
    #[serde(rename = "ragflow_response")]
    RagflowResponse {
        answer: String
    },
    
    #[serde(rename = "openai_response")]
    OpenAIResponse {
        response: String,
        audio: Option<String> // base64 encoded audio
    },
    
    #[serde(rename = "error")]
    Error {
        message: String,
        code: Option<String>
    },
    
    #[serde(rename = "graph_update")]
    GraphUpdate {
        graph_data: Value
    },
    
    #[serde(rename = "simulation_mode_set")]
    SimulationModeSet {
        mode: String,
        gpu_enabled: bool
    },

    #[serde(rename = "fisheye_settings_updated")]
    FisheyeSettingsUpdated {
        enabled: bool,
        strength: f32,
        focus_point: [f32; 3],
        radius: f32,
    }
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnected;

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIConnectionFailed;

#[derive(Message)]
#[rtype(result = "()")]
pub struct OpenAIMessage(pub String);

pub trait MessageHandler: Actor<Context = ws::WebsocketContext<Self>> {
    fn send_json_response(&self, response: Value, ctx: &mut ws::WebsocketContext<Self>) {
        match serde_json::to_string(&response) {
            Ok(json_string) => {
                debug!("Sending JSON response: {}", json_string);
                match compress_message(&json_string) {
                    Ok(compressed) => {
                        debug!("Compressed response size: {} bytes", compressed.len());
                        ctx.binary(compressed)
                    },
                    Err(e) => {
                        error!("Failed to compress JSON response: {}", e);
                        // Fallback to uncompressed JSON if compression fails
                        ctx.text(json_string);
                    }
                }
            },
            Err(e) => {
                error!("Failed to serialize JSON response: {}", e);
                let error_message = json!({
                    "type": "error",
                    "message": format!("Failed to serialize response: {}", e),
                    "code": "SERIALIZATION_ERROR"
                });
                if let Ok(error_string) = serde_json::to_string(&error_message) {
                    ctx.text(error_string);
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
        assert!(serialized.contains("chat_message"));
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
        assert!(serialized.contains("ragflow_response"));
        assert!(serialized.contains("Test answer"));

        let fisheye_message = ServerMessage::FisheyeSettingsUpdated {
            enabled: true,
            strength: 0.5,
            focus_point: [0.0, 0.0, 0.0],
            radius: 100.0,
        };
        let serialized = serde_json::to_string(&fisheye_message).unwrap();
        assert!(serialized.contains("fisheye_settings_updated"));
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

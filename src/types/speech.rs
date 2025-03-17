use tokio::sync::mpsc;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum SpeechError {
    WebSocketError(tungstenite::Error),
    ConnectionError(String),
    SendError(mpsc::error::SendError<SpeechCommand>),
    SerializationError(serde_json::Error),
    ProcessError(std::io::Error),
    Base64Error(base64::DecodeError),
    BroadcastError(String),
    TTSError(String),
}

impl fmt::Display for SpeechError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpeechError::WebSocketError(e) => write!(f, "WebSocket error: {}", e),
            SpeechError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            SpeechError::SendError(e) => write!(f, "Send error: {}", e),
            SpeechError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            SpeechError::ProcessError(e) => write!(f, "Process error: {}", e),
            SpeechError::Base64Error(e) => write!(f, "Base64 error: {}", e),
            SpeechError::BroadcastError(msg) => write!(f, "Broadcast error: {}", msg),
            SpeechError::TTSError(msg) => write!(f, "TTS error: {}", msg),
        }
    }
}

impl Error for SpeechError {}

impl From<tungstenite::Error> for SpeechError {
    fn from(err: tungstenite::Error) -> Self {
        SpeechError::WebSocketError(err)
    }
}

impl From<mpsc::error::SendError<SpeechCommand>> for SpeechError {
    fn from(err: mpsc::error::SendError<SpeechCommand>) -> Self {
        SpeechError::SendError(err)
    }
}

impl From<serde_json::Error> for SpeechError {
    fn from(err: serde_json::Error) -> Self {
        SpeechError::SerializationError(err)
    }
}

impl From<std::io::Error> for SpeechError {
    fn from(err: std::io::Error) -> Self {
        SpeechError::ProcessError(err)
    }
}

impl From<base64::DecodeError> for SpeechError {
    fn from(err: base64::DecodeError) -> Self {
        SpeechError::Base64Error(err)
    }
}

#[derive(Debug, Clone)]
pub enum TTSProvider {
    OpenAI,
    Kokoro,
}

#[derive(Debug)]
pub enum SpeechCommand {
    Initialize,
    SendMessage(String),
    TextToSpeech(String, SpeechOptions),
    Close,
    SetTTSProvider(TTSProvider),
}

#[derive(Debug, Clone)]
pub struct SpeechOptions {
    pub voice: String,
    pub speed: f32,
    pub stream: bool,
}

impl Default for SpeechOptions {
    fn default() -> Self {
        Self {
            voice: "af_heart".to_string(), // Default Kokoro voice
            speed: 1.0,
            stream: true,
        }
    }
}

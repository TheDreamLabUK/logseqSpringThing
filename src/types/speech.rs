use std::fmt;
use tokio::sync::mpsc::error::SendError;

#[derive(Debug, Clone)]
pub enum TTSProvider {
    OpenAI,
    Sonata,
}

#[derive(Debug)]
pub enum SpeechCommand {
    Initialize,
    SendMessage(String),
    Close,
    SetTTSProvider(TTSProvider),
}

#[derive(Debug)]
pub struct SpeechError {
    message: String,
}

impl fmt::Display for SpeechError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SpeechError {}

impl From<SendError<SpeechCommand>> for SpeechError {
    fn from(error: SendError<SpeechCommand>) -> Self {
        SpeechError {
            message: error.to_string(),
        }
    }
}

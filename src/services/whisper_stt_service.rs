use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use log::{info, error, debug};

use crate::types::speech::SpeechError; // Assuming SttError will be part of SpeechError or a new error type
use crate::config::AppFullSettings; // To potentially get whisper-webui URL from settings

#[derive(Deserialize, Debug)]
struct WhisperApiResponse {
    // Define the expected structure of a successful response from Whisper
    // This is a placeholder and needs to be adjusted based on the actual API
    text: Option<String>, // Assuming the transcription is in a "text" field
    // language: Option<String>,
    // segments: Option<Vec<WhisperSegment>>,
    // etc.
}

// Placeholder for segment if needed
// #[derive(Deserialize, Debug)]
// struct WhisperSegment {
//     id: u32,
//     seek: u32,
//     start: f32,
//     end: f32,
//     text: String,
//     // ... other fields
// }


#[derive(Debug)]
pub enum SttError {
    RequestError(reqwest::Error),
    ApiError(String), // For errors returned by the Whisper API itself
    NoTranscription,
    ConfigError(String),
}

impl std::fmt::Display for SttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SttError::RequestError(e) => write!(f, "STT HTTP request error: {}", e),
            SttError::ApiError(msg) => write!(f, "STT API error: {}", msg),
            SttError::NoTranscription => write!(f, "STT service did not return a transcription."),
            SttError::ConfigError(msg) => write!(f, "STT configuration error: {}", msg),
        }
    }
}

impl std::error::Error for SttError {}

impl From<reqwest::Error> for SttError {
    fn from(err: reqwest::Error) -> Self {
        SttError::RequestError(err)
    }
}


pub struct WhisperSttService {
    http_client: Arc<Client>,
    settings: Arc<tokio::sync::RwLock<AppFullSettings>>,
    whisper_api_url: String, // We'll populate this from settings or a default
}

impl WhisperSttService {
    pub fn new(settings: Arc<tokio::sync::RwLock<AppFullSettings>>, http_client: Arc<Client>) -> Result<Self, SttError> {
        let whisper_api_url = {
            let s = settings.blocking_read(); // Use blocking_read for initial setup if in sync context
                                           // Or make `new` async if preferred
            s.whisper.as_ref()
                .and_then(|w_settings| w_settings.api_url.clone())
                .ok_or_else(|| SttError::ConfigError("Whisper API URL not configured".to_string()))?
        };
        
        // Ensure the URL ends with /asr or the correct endpoint.
        // For now, let's assume the base URL is provided and we append a known endpoint.
        // This needs to be verified based on whisper-webui's actual API.
        // Example: let full_api_url = format!("{}/asr", whisper_api_url.trim_end_matches('/'));
        // For now, we'll assume the provided URL is the full endpoint.
        let full_api_url = whisper_api_url;


        info!("WhisperSttService initialized with API URL: {}", full_api_url);

        Ok(Self {
            http_client,
            settings,
            whisper_api_url: full_api_url,
        })
    }

    pub async fn transcribe(&self, audio_data: Vec<u8>) -> Result<String, SttError> {
        if audio_data.is_empty() {
            return Err(SttError::ApiError("Audio data is empty".to_string()));
        }

        debug!("Sending {} bytes of audio data to Whisper STT: {}", audio_data.len(), self.whisper_api_url);

        // The actual request format depends heavily on the whisper-webui API.
        // It might be a multipart/form-data request with the audio file,
        // or it might accept raw audio bytes with a specific content type.

        // Placeholder: Assuming a POST request with raw audio bytes.
        // This will likely need to be changed to multipart/form-data.
        // Example for multipart:
        // let part = reqwest::multipart::Part::bytes(audio_data)
        //     .file_name("audio.wav") // Or .ogg, .mp3, etc. depending on what client sends and whisper accepts
        //     .mime_str("audio/wav")?; // Or appropriate MIME type
        // let form = reqwest::multipart::Form::new().part("audio_file", part);
        // let response = self.http_client.post(&self.whisper_api_url)
        //     .multipart(form)
        //     .send()
        //     .await?;

        // For now, a simple POST with bytes - THIS IS LIKELY INCORRECT for whisper-webui
        let response = self.http_client.post(&self.whisper_api_url)
            .header("Content-Type", "audio/wav") // This needs to match what whisper-webui expects
            .body(audio_data)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            error!("Whisper API error {}: {}", status, error_text);
            return Err(SttError::ApiError(format!("Whisper API request failed with status {}: {}", status, error_text)));
        }

        let api_response = response.json::<WhisperApiResponse>().await?;
        debug!("Whisper API raw response: {:?}", api_response);

        api_response.text.ok_or(SttError::NoTranscription)
    }
}
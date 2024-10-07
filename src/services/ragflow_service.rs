use reqwest::{Client, StatusCode};
use log::{error, info};
use crate::config::Settings;
use std::fmt;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

// Import Sonata synthesizer
use crate::services::sonata_service::{SonataSpeechSynthesizer, SonataSynthError};
use std::path::Path;

#[derive(Debug)]
pub enum RAGFlowError {
    ReqwestError(reqwest::Error),
    StatusError(StatusCode, String),
    AudioGenerationError(String),
    IoError(std::io::Error),
    SonataError(String),
}

impl fmt::Display for RAGFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RAGFlowError::ReqwestError(e) => write!(f, "Reqwest error: {}", e),
            RAGFlowError::StatusError(status, msg) => write!(f, "Status error ({}): {}", status, msg),
            RAGFlowError::AudioGenerationError(msg) => write!(f, "Audio generation error: {}", msg),
            RAGFlowError::IoError(e) => write!(f, "IO error: {}", e),
            RAGFlowError::SonataError(msg) => write!(f, "Sonata error: {}", msg),
        }
    }
}

impl std::error::Error for RAGFlowError {}

impl From<reqwest::Error> for RAGFlowError {
    fn from(err: reqwest::Error) -> Self {
        RAGFlowError::ReqwestError(err)
    }
}

impl From<std::io::Error> for RAGFlowError {
    fn from(err: std::io::Error) -> Self {
        RAGFlowError::IoError(err)
    }
}

impl From<SonataSynthError> for RAGFlowError {
    fn from(err: SonataSynthError) -> Self {
        RAGFlowError::SonataError(err.to_string())
    }
}

pub struct RAGFlowService {
    client: Client,
    api_key: String,
    base_url: String,
    synthesizer: Arc<SonataSpeechSynthesizer>,
}

impl RAGFlowService {
    pub fn new(settings: &Settings) -> Result<Self, RAGFlowError> {
        // Initialize Sonata Synthesizer
        let voice_config_path = &settings.sonata.voice_config_path;
        let synthesizer = SonataSpeechSynthesizer::new(Path::new(voice_config_path))
            .map_err(|e| RAGFlowError::SonataError(format!("Failed to initialize synthesizer: {}", e)))?;
        
        Ok(RAGFlowService {
            client: Client::new(),
            api_key: settings.ragflow.api_key.clone(),
            base_url: settings.ragflow.base_url.clone(),
            synthesizer: Arc::new(synthesizer),
        })
    }

    pub async fn create_conversation(&self, user_id: String) -> Result<String, RAGFlowError> {
        info!("Creating conversation for user: {}", user_id);
        let url = format!("{}api/new_conversation", self.base_url);
        info!("Full URL for create_conversation: {}", url);
        
        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .query(&[("user_id", user_id)])
            .send()
            .await?;

        info!("Response status: {}", response.status());

        if response.status().is_success() {
            let result: serde_json::Value = response.json().await?;
            info!("Successful response: {:?}", result);
            Ok(result["data"]["id"].as_str().unwrap_or("").to_string())
        } else {
            let status = response.status();
            let error_message = response.text().await?;
            error!("Failed to create conversation. Status: {}, Error: {}", status, error_message);
            Err(RAGFlowError::StatusError(status, error_message))
        }
    }

    pub async fn send_message(
        &self,
        conversation_id: String,
        message: String,
        quote: bool,
        doc_ids: Option<Vec<String>>,
        stream: bool,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<u8>, RAGFlowError>> + Send + 'static>>, RAGFlowError> {
        info!("Sending message to conversation: {}", conversation_id);
        let url = format!("{}api/completion", self.base_url);
        info!("Full URL for send_message: {}", url);
        
        let mut request_body = serde_json::json!({
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "content": message}],
            "quote": quote,
            "stream": stream
        });

        if let Some(ids) = doc_ids {
            request_body["doc_ids"] = serde_json::json!(ids.join(","));
        }

        info!("Request body: {:?}", request_body);

        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        info!("Response status: {}", response.status());
       
        if response.status().is_success() {
            let synthesizer = Arc::clone(&self.synthesizer);
            let stream = response.bytes_stream().map(move |chunk_result| {
                match chunk_result {
                    Ok(chunk) => {
                        let text = String::from_utf8_lossy(&chunk);
                        // Synthesize audio using Sonata
                        match synthesizer.synthesize(text.as_ref()) {
                            Ok(audio) => Ok(audio),
                            Err(e) => Err(RAGFlowError::SonataError(format!("Synthesis failed: {}", e))),
                        }
                    },
                    Err(e) => Err(RAGFlowError::ReqwestError(e)),
                }
            });

            Ok(Box::pin(stream))
        } else {
            let status = response.status();
            let error_message = response.text().await?;
            error!("Failed to send message. Status: {}, Error: {}", status, error_message);
            Err(RAGFlowError::StatusError(status, error_message))
        }
    }
}

impl Clone for RAGFlowService {
    fn clone(&self) -> Self {
        RAGFlowService {
            client: self.client.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            synthesizer: Arc::clone(&self.synthesizer),
        }
    }
}
use reqwest::{Client, StatusCode};
use log::{error, info};
use crate::config::Settings;
use std::fmt;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum RAGFlowError {
    ReqwestError(reqwest::Error),
    StatusError(StatusCode, String),
    ParseError(String),
    IoError(std::io::Error),
}

impl fmt::Display for RAGFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RAGFlowError::ReqwestError(e) => write!(f, "Reqwest error: {}", e),
            RAGFlowError::StatusError(status, msg) => write!(f, "Status error ({}): {}", status, msg),
            RAGFlowError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            RAGFlowError::IoError(e) => write!(f, "IO error: {}", e),
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

#[derive(Debug, Deserialize)]
struct SessionResponse {
    code: i32,
    data: SessionData,
}

#[derive(Debug, Deserialize)]
struct SessionData {
    id: String,
    message: Option<Vec<Message>>,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    code: i32,
    data: CompletionData,
}

#[derive(Debug, Deserialize)]
struct CompletionData {
    answer: Option<String>,
    reference: Option<serde_json::Value>,
    id: Option<String>,
    session_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct CompletionRequest {
    question: String,
    stream: bool,
    session_id: Option<String>,
    user_id: Option<String>,
    sync_dsl: Option<bool>,
}

pub struct RAGFlowService {
    client: Client,
    api_key: String,
    base_url: String,
    agent_id: String,
}

impl RAGFlowService {
    pub async fn new(settings: Arc<RwLock<Settings>>) -> Result<Self, RAGFlowError> {
        let client = Client::new();
        let settings = settings.read().await;

        Ok(RAGFlowService {
            client,
            api_key: settings.ragflow.api_key.clone(),
            base_url: settings.ragflow.api_base_url.clone(),
            agent_id: settings.ragflow.agent_id.clone(),
        })
    }

    pub async fn create_session(&self, user_id: String) -> Result<String, RAGFlowError> {
        info!("Creating session for user: {}", user_id);
        let url = format!(
            "{}/api/v1/agents/{}/sessions?user_id={}", 
            self.base_url.trim_end_matches('/'), 
            self.agent_id,
            user_id
        );
        info!("Full URL for create_session: {}", url);
        
        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .body("{}")  // Empty JSON body as we don't have any Begin parameters
            .send()
            .await?;

        let status = response.status();
        info!("Response status: {}", status);

        if status.is_success() {
            let result: serde_json::Value = response.json().await?;
            info!("Successful response: {:?}", result);
            
            // Extract session ID from the response
            match result["data"]["id"].as_str() {
                Some(id) => Ok(id.to_string()),
                None => {
                    error!("Failed to parse session ID from response: {:?}", result);
                    Err(RAGFlowError::ParseError("Failed to parse session ID".to_string()))
                }
            }
        } else {
            let error_message = response.text().await?;
            error!("Failed to create session. Status: {}, Error: {}", status, error_message);
            Err(RAGFlowError::StatusError(status, error_message))
        }
    }

    pub async fn send_message(
        &self,
        session_id: String,
        message: String,
        _quote: bool,  // Not used in new API
        _doc_ids: Option<Vec<String>>,  // Not used in new API
        stream: bool,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, RAGFlowError>> + Send + 'static>>, RAGFlowError> {
        info!("Sending message to session: {}", session_id);
        let url = format!(
            "{}/api/v1/agents/{}/completions", 
            self.base_url.trim_end_matches('/'),
            self.agent_id
        );
        info!("Full URL for send_message: {}", url);
        
        let request_body = CompletionRequest {
            question: message,
            stream,
            session_id: Some(session_id),
            user_id: None,
            sync_dsl: Some(false),
        };

        info!("Request body: {:?}", serde_json::to_string(&request_body).unwrap_or_default());

        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        info!("Response status: {}", status);
       
        if status.is_success() {
            if stream {
                let stream = response.bytes_stream().map(move |chunk_result| {
                    match chunk_result {
                        Ok(chunk) => {
                            let chunk_str = String::from_utf8_lossy(&chunk);
                            // Handle SSE format (data: {...})
                            let chunk_str = chunk_str.trim();
                            
                            if chunk_str.starts_with("data:") {
                                let json_str = chunk_str.trim_start_matches("data:").trim();
                                match serde_json::from_str::<serde_json::Value>(json_str) {
                                    Ok(json_response) => {
                                        if let Some(true) = json_response["data"].as_bool() {
                                            // This is the end marker
                                            Ok("".to_string())
                                        } else if let Some(answer) = json_response["data"]["answer"].as_str() {
                                            Ok(answer.to_string())
                                        } else {
                                            Err(RAGFlowError::ParseError("No answer found in response".to_string()))
                                        }
                                    },
                                    Err(e) => Err(RAGFlowError::ParseError(format!("Failed to parse JSON: {}, content: {}", e, json_str))),
                                }
                            } else {
                                Err(RAGFlowError::ParseError(format!("Invalid SSE format: {}", chunk_str)))
                            }
                        },
                        Err(e) => Err(RAGFlowError::ReqwestError(e)),
                    }
                });

                Ok(Box::pin(stream))
            } else {
                // Non-streaming response handling
                let result: serde_json::Value = response.json().await?;
                
                if let Some(answer) = result["data"]["answer"].as_str() {
                    // Create a one-item stream with the answer
                    let stream = futures::stream::once(futures::future::ok(answer.to_string()));
                    Ok(Box::pin(stream))
                } else {
                    Err(RAGFlowError::ParseError("No answer found in response".to_string()))
                }
            }
        } else {
            let error_message = response.text().await?;
            error!("Failed to send message. Status: {}, Error: {}", status, error_message);
            Err(RAGFlowError::StatusError(status, error_message))
        }
    }

    pub async fn get_session_history(&self, session_id: String) -> Result<serde_json::Value, RAGFlowError> {
        let url = format!(
            "{}/api/v1/agents/{}/sessions?id={}", 
            self.base_url.trim_end_matches('/'), 
            self.agent_id,
            session_id
        );
        
        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            let history: serde_json::Value = response.json().await?;
            Ok(history)
        } else {
            let error_message = response.text().await?;
            error!("Failed to get session history. Status: {}, Error: {}", status, error_message);
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
            agent_id: self.agent_id.clone(),
        }
    }
}

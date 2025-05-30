use reqwest::{Client, StatusCode};
use log::{error, info};
use crate::config::AppFullSettings; // Use AppFullSettings, ConfigRagFlowSettings removed
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
    // Updated signature and logic to handle optional settings
    pub async fn new(_settings: Arc<RwLock<AppFullSettings>>) -> Result<Self, RAGFlowError> { // settings might still be needed for other parts if any
        let client = Client::new();
        // let settings_read = settings.read().await; // Keep if other ragflow settings (timeout, etc.) are used from config

        info!("[RAGFlowService::new] Attempting to load RAGFlow config directly from environment variables.");

        let api_key = std::env::var("RAGFLOW_API_KEY")
            .map_err(|e| {
                error!("[RAGFlowService::new] Failed to read RAGFLOW_API_KEY: {}", e);
                RAGFlowError::ParseError(format!("RAGFLOW_API_KEY environment variable not found or invalid: {}", e))
            })?;
            
        let base_url = std::env::var("RAGFLOW_API_BASE_URL")
            .map_err(|e| {
                error!("[RAGFlowService::new] Failed to read RAGFLOW_API_BASE_URL: {}", e);
                RAGFlowError::ParseError(format!("RAGFLOW_API_BASE_URL environment variable not found or invalid: {}", e))
            })?;
            
        let agent_id = std::env::var("RAGFLOW_AGENT_ID")
            .map_err(|e| {
                error!("[RAGFlowService::new] Failed to read RAGFLOW_AGENT_ID: {}", e);
                RAGFlowError::ParseError(format!("RAGFLOW_AGENT_ID environment variable not found or invalid: {}", e))
            })?;

        info!("[RAGFlowService::new] RAGFLOW_API_KEY: loaded (value redacted)");
        info!("[RAGFlowService::new] RAGFLOW_API_BASE_URL: {}", base_url);
        info!("[RAGFlowService::new] RAGFLOW_AGENT_ID: {}", agent_id);

        // Check if essential fields are empty after loading from env
        if api_key.is_empty() {
            error!("[RAGFlowService::new] RAGFLOW_API_KEY is empty after loading from environment.");
            return Err(RAGFlowError::ParseError("RAGFLOW_API_KEY environment variable is empty".to_string()));
        }
        if base_url.is_empty() {
            error!("[RAGFlowService::new] RAGFLOW_API_BASE_URL is empty after loading from environment.");
            return Err(RAGFlowError::ParseError("RAGFLOW_API_BASE_URL environment variable is empty".to_string()));
        }
        if agent_id.is_empty() {
            error!("[RAGFlowService::new] RAGFLOW_AGENT_ID is empty after loading from environment.");
            return Err(RAGFlowError::ParseError("RAGFLOW_AGENT_ID environment variable is empty".to_string()));
        }
        
        info!("[RAGFlowService::new] Successfully loaded RAGFlow API key, base URL, and agent ID from environment variables.");

        Ok(RAGFlowService {
            client,
            api_key,
            base_url,
            agent_id,
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

    pub async fn send_chat_message(
        &self,
        session_id: String,
        message: String,
        stream_preference: bool, // Added stream_preference parameter
    ) -> Result<(String, String), RAGFlowError> { // Returns (answer, session_id)
        info!("Sending chat message to RAGFlow session: {}, stream_preference: {}", session_id, stream_preference);
        let url = format!(
            "{}/api/v1/agents/{}/completions",
            self.base_url.trim_end_matches('/'),
            self.agent_id
        );

        let request_body = CompletionRequest {
            question: message,
            stream: stream_preference, // Use the passed-in stream_preference
            session_id: Some(session_id.clone()),
            user_id: None,
            sync_dsl: Some(false),
        };

        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_message = response.text().await?;
            error!("RAGFlow chat API error. Status: {}, Error: {}", status, error_message);
            return Err(RAGFlowError::StatusError(status, error_message));
        }

        if !stream_preference {
            // Handle non-streamed response directly
            let result: serde_json::Value = response.json().await
                .map_err(|e| RAGFlowError::ParseError(format!("Failed to parse non-streamed JSON response: {}", e)))?;
            
            info!("RAGFlow non-streamed response: {:?}", result);

            let answer = result.get("data")
                .and_then(|data| data.get("answer"))
                .and_then(|ans| ans.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| RAGFlowError::ParseError("Answer not found in non-streamed RAGFlow response".to_string()))?;
            
            // The session_id in the response data might be the same or a new one if RAGFlow refreshed it.
            // For consistency, we can use the session_id from the response if available, otherwise the one we sent.
            let final_session_id = result.get("data")
                .and_then(|data| data.get("session_id"))
                .and_then(|sid| sid.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| session_id.clone()); // Fallback to original session_id

            Ok((answer, final_session_id))
        } else {
            // Aggregate streamed response (existing logic)
            let mut full_answer = String::new();
            let mut response_stream = response.bytes_stream();

            while let Some(chunk_result) = response_stream.next().await {
                match chunk_result {
                    Ok(chunk_bytes) => {
                        let chunk_vec: Vec<u8> = chunk_bytes.to_vec();
                        let chunk_str = String::from_utf8_lossy(&chunk_vec);
                        for line in chunk_str.lines() {
                            if line.starts_with("data:") {
                                let json_str = line.trim_start_matches("data:").trim();
                                if json_str.is_empty() { continue; }

                                match serde_json::from_str::<serde_json::Value>(json_str) {
                                    Ok(json_val) => {
                                        if json_val.get("code").and_then(|c| c.as_i64()) == Some(0) &&
                                           json_val.get("data").and_then(|d| d.as_bool()) == Some(true) {
                                            // End of stream marker
                                            return Ok((full_answer, session_id)); // Return aggregated answer
                                        }
                                        
                                        if let Some(answer_chunk) = json_val.get("data").and_then(|d| d.get("answer")).and_then(|a| a.as_str()) {
                                            full_answer.push_str(answer_chunk);
                                        } else if let Some(answer_chunk) = json_val.get("answer").and_then(|a| a.as_str()) {
                                            full_answer.push_str(answer_chunk);
                                        }
                                    },
                                    Err(e) => log::warn!("Failed to parse RAGFlow stream chunk JSON: {}. Chunk: '{}'", e, json_str),
                                }
                            }
                        }
                         // Check for end marker in the whole chunk if not caught by line-by-line parsing
                        if chunk_str.contains(r#"{"code":0,"data":true}"#) || chunk_str.contains(r#"{"code": 0, "data": true}"#) {
                             return Ok((full_answer, session_id)); // Return aggregated answer
                        }
                    },
                    Err(e) => {
                        log::error!("Error reading RAGFlow stream chunk: {}", e);
                        return Err(RAGFlowError::ReqwestError(e));
                    }
                }
            }
            // If loop finishes without explicit end marker, return what was aggregated.
            Ok((full_answer, session_id))
        }
    }
} // This closing brace now correctly closes impl RAGFlowService

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

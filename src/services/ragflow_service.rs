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

// This struct is used by speech_service to construct a request for RAGFlow
// It should align with the parameters expected by the RAGFlow API endpoint
// that speech_service intends to call (e.g., via a send_chat_message_full method).
#[derive(Debug, Serialize, Deserialize, Clone)] // Added Deserialize for completeness if ever needed
#[serde(rename_all = "camelCase")] // Assuming client/API consistency
pub struct RAGFlowBody {
    pub chat_id: String,
    pub query: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_citation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_rag_citation: Option<bool>, // Specific to RAGFlow's RAG capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_rag_rewrite: Option<bool>, // Specific to RAGFlow's RAG capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_rewrite: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_vertical_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_config: Option<serde_json::Value>, // Using Value for flexibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_variables: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieve_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// Structs for OpenAI-Compatible Chat Completions API
#[derive(Debug, Serialize, Clone)]
pub struct OpenAIChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct OpenAIChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAIChatMessage>,
    pub stream: bool,
    // Add other OpenAI compatible fields if needed, e.g., temperature, max_tokens
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIMessageContent {
    pub content: String,
    // role: Option<String>, // if needed
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIChatCompletionChoice {
    pub message: OpenAIMessageContent,
    // finish_reason: Option<String>, // if needed
    // index: Option<u32>, // if needed
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIChatCompletionResponse {
    pub choices: Vec<OpenAIChatCompletionChoice>,
    // id: Option<String>, // if needed
    // object: Option<String>, // if needed
    // created: Option<u64>, // if needed
    // model: Option<String>, // if needed
    // usage: Option<OpenAIUsage>, // if needed
}

// #[derive(Debug, Deserialize, Clone)]
// pub struct OpenAIUsage {
//     prompt_tokens: u32,
//     completion_tokens: u32,
//     total_tokens: u32,
// }


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
    ) -> Result<(String, String), RAGFlowError> { // Returns (answer, session_id)
        info!("Sending chat message to RAGFlow session: {}", session_id);
        let url = format!(
            "{}/api/v1/agents/{}/completions",
            self.base_url.trim_end_matches('/'),
            self.agent_id
        );

        let request_body = CompletionRequest {
            question: message,
            stream: true, // RAGFlow itself might be streaming, so we request stream and aggregate
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

        // Aggregate streamed response
        let mut full_answer = String::new();
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk_bytes) => { // chunk_bytes is Bytes
                    let chunk_vec: Vec<u8> = chunk_bytes.to_vec(); // Convert to Vec<u8>
                    let chunk_str = String::from_utf8_lossy(&chunk_vec);
                    // RAGFlow SSE format is typically "data: {...}\n\n"
                    // Multiple "data:" lines can appear in a single chunk in some implementations.
                    for line in chunk_str.lines() {
                        if line.starts_with("data:") {
                            let json_str = line.trim_start_matches("data:").trim();
                            if json_str.is_empty() { // Skip empty data lines often sent as keep-alives
                                continue;
                            }
                            // Deprecated: if json_str == "[DONE]"
                            // The RAGFlow API doc indicates the end marker is a JSON object:
                            // data: {"code": 0, "data": true}

                            match serde_json::from_str::<serde_json::Value>(json_str) {
                                Ok(json_val) => {
                                    // Check for the end-of-stream marker: {"code": 0, "data": true}
                                    if json_val.get("code").and_then(|c| c.as_i64()) == Some(0) &&
                                       json_val.get("data").and_then(|d| d.as_bool()) == Some(true) {
                                        // info!("RAGFlow stream end marker received via JSON.");
                                        break; // Exit the inner loop over lines (and will break outer due to no more content)
                                    }
                                    
                                    // Extract answer chunk
                                    if let Some(answer_chunk) = json_val.get("data").and_then(|d| d.get("answer")).and_then(|a| a.as_str()) {
                                        full_answer.push_str(answer_chunk);
                                    } else if let Some(answer_chunk) = json_val.get("answer").and_then(|a| a.as_str()) { // Fallback for direct answer
                                        full_answer.push_str(answer_chunk);
                                    }
                                    // Session ID is usually static for the conversation.
                                },
                                Err(e) => log::warn!("Failed to parse RAGFlow stream chunk JSON: {}. Chunk: '{}'", e, json_str),
                            }
                        }
                    }
                    // Check if the specific end-of-stream JSON structure was found in any line of the chunk
                    // This ensures the outer loop breaks correctly.
                    if chunk_str.contains(r#"{"code":0,"data":true}"#) || chunk_str.contains(r#"{"code": 0, "data": true}"#) { // Handle potential spacing variations
                        break;
                    }
                },
                Err(e) => {
                    log::error!("Error reading RAGFlow stream chunk: {}", e);
                    return Err(RAGFlowError::ReqwestError(e));
                }
            }
        }
        Ok((full_answer, session_id)) // Return aggregated answer and original session_id
    }

    // New method for OpenAI-Compatible Agent Chat Completions
    pub async fn send_openai_compatible_completion(
        &self,
        query: String,
        // session_id: Option<String>, // The OpenAI compatible API for agents doesn't seem to use session_id in the path or body explicitly for this endpoint
    ) -> Result<String, RAGFlowError> {
        info!("Sending OpenAI-compatible completion request to RAGFlow agent_id: {}", self.agent_id);
        let url = format!(
            "{}/api/v1/agents_openai/{}/chat/completions",
            self.base_url.trim_end_matches('/'),
            self.agent_id
        );

        let messages = vec![OpenAIChatMessage {
            role: "user".to_string(),
            content: query,
        }];

        let request_body = OpenAIChatCompletionRequest {
            model: "ragflow-agent".to_string(), // As per docs, this can be any string
            messages,
            stream: false, // We want the full response for TTS
        };

        let response = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            error!(
                "RAGFlow OpenAI-compatible API error. Status: {}, Body: {}",
                status, error_body
            );
            return Err(RAGFlowError::StatusError(status, error_body));
        }

        let completion_response = response.json::<OpenAIChatCompletionResponse>().await?;

        if let Some(first_choice) = completion_response.choices.get(0) {
            Ok(first_choice.message.content.clone())
        } else {
            Err(RAGFlowError::ParseError(
                "No choices found in RAGFlow OpenAI-compatible response".to_string(),
            ))
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

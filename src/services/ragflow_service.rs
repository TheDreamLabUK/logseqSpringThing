use reqwest::{Client, StatusCode};
use log::{error, info};
use crate::config::Settings;
use std::fmt;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;

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

pub struct RAGFlowService {
    client: Client,
    api_key: String,
    base_url: String,
}

impl RAGFlowService {
    pub async fn new(settings: Arc<RwLock<Settings>>) -> Result<Self, RAGFlowError> {
        let client = Client::new();
        let settings = settings.read().await;

        Ok(RAGFlowService {
            client,
            api_key: settings.ragflow.api_key.clone(),
            base_url: settings.ragflow.base_url.clone(),
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

        let status = response.status();
        info!("Response status: {}", status);

        if status.is_success() {
            let result: serde_json::Value = response.json().await?;
            info!("Successful response: {:?}", result);
            Ok(result["data"]["id"].as_str().unwrap_or("").to_string())
        } else {
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, RAGFlowError>> + Send + 'static>>, RAGFlowError> {
        info!("Sending message to conversation: {}", conversation_id);
        let url = format!("{}api/completion", self.base_url);
        info!("Full URL for send_message: {}", url);
        
        let mut request_body = json!({
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

        let status = response.status();
        info!("Response status: {}", status);
       
        if status.is_success() {
            let stream = response.bytes_stream().map(move |chunk_result| {
                match chunk_result {
                    Ok(chunk) => {
                        match serde_json::from_slice::<serde_json::Value>(&chunk) {
                            Ok(json_response) => {
                                // Extract text answer from the response
                                match json_response["data"]["answer"].as_str()
                                    .or_else(|| json_response["answer"].as_str()) {
                                    Some(answer) => Ok(answer.to_string()),
                                    None => Err(RAGFlowError::ParseError("No answer found in response".to_string()))
                                }
                            },
                            Err(e) => Err(RAGFlowError::ParseError(format!("Failed to parse JSON response: {}", e))),
                        }
                    },
                    Err(e) => Err(RAGFlowError::ReqwestError(e)),
                }
            });

            Ok(Box::pin(stream))
        } else {
            let error_message = response.text().await?;
            error!("Failed to send message. Status: {}, Error: {}", status, error_message);
            Err(RAGFlowError::StatusError(status, error_message))
        }
    }

    pub async fn get_conversation_history(&self, conversation_id: String) -> Result<serde_json::Value, RAGFlowError> {
        let url = format!("{}api/conversation/{}", self.base_url, conversation_id);
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
            error!("Failed to get conversation history. Status: {}, Error: {}", status, error_message);
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
        }
    }
}

use serde::{Deserialize, Serialize};
use reqwest::{Client, StatusCode};
use log::{debug, error, info};
use crate::config::Settings;
use std::fmt;

#[derive(Debug)]
pub enum RAGFlowError {
    ReqwestError(reqwest::Error),
    StatusError(StatusCode, String),
    DeserializationError(String),
}

impl fmt::Display for RAGFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RAGFlowError::ReqwestError(e) => write!(f, "Reqwest error: {}", e),
            RAGFlowError::StatusError(status, msg) => write!(f, "Status error ({}): {}", status, msg),
            RAGFlowError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
        }
    }
}

impl std::error::Error for RAGFlowError {}

impl From<reqwest::Error> for RAGFlowError {
    fn from(err: reqwest::Error) -> Self {
        RAGFlowError::ReqwestError(err)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatResponse {
    pub retcode: i32,
    pub data: ChatResponseData,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ChatResponseData {
    MessageArray { message: Vec<Message> },
    SingleMessage { message: Message },
    Empty {},
}

pub struct RAGFlowService {
    client: Client,
    api_key: String,
    base_url: String,
}

impl RAGFlowService {
    pub fn new(settings: &Settings) -> Self {
        info!("Creating RAGFlowService with base URL: {}", settings.ragflow.ragflow_api_base_url);
        Self {
            client: Client::new(),
            api_key: settings.ragflow.ragflow_api_key.clone(),
            base_url: settings.ragflow.ragflow_api_base_url.clone(),
        }
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

    pub async fn send_message(&self, conversation_id: String, message: String, quote: bool, doc_ids: Option<Vec<String>>, stream: bool) -> Result<ChatResponse, RAGFlowError> {
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
            let response_text = response.text().await?;
            info!("Raw response: {}", response_text);
            
            match serde_json::from_str::<ChatResponse>(&response_text) {
                Ok(result) => {
                    info!("Successful response: {:?}", result);
                    Ok(result)
                },
                Err(e) => {
                    error!("Failed to deserialize response: {}", e);
                    Err(RAGFlowError::DeserializationError(e.to_string()))
                }
            }
        } else {
            let status = response.status();
            let error_message = response.text().await?;
            error!("Failed to send message. Status: {}, Error: {}", status, error_message);
            Err(RAGFlowError::StatusError(status, error_message))
        }
    }

    pub async fn get_chat_history(&self, conversation_id: String) -> Result<ChatResponse, RAGFlowError> {
        debug!("Fetching chat history for conversation: {}", conversation_id);
        let url = format!("{}api/chat/history/{}", self.base_url, conversation_id);
        debug!("Full URL for get_chat_history: {}", url);
        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if response.status().is_success() {
            let result: ChatResponse = response.json().await?;
            Ok(result)
        } else {
            let status = response.status();
            let error_message = response.text().await?;
            error!("Failed to fetch chat history. Status: {}, Error: {}", status, error_message);
            Err(RAGFlowError::StatusError(status, error_message))
        }
    }
}
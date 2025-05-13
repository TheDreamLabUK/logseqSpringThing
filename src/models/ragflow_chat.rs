// src/models/ragflow_chat.rs
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RagflowChatRequest {
    pub question: String,
    pub session_id: Option<String>, // Client might send existing session ID
    // Add any other RAGFlow specific params client might send
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RagflowChatResponse {
    pub answer: String,
    pub session_id: String, // Server returns session_id for future requests
    // Add any other RAGFlow specific response fields
}
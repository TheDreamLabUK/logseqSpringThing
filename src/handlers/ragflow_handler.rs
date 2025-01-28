use actix_web::{web, HttpResponse, ResponseError, Responder};
use crate::AppState;
use serde::{Serialize, Deserialize};
use log::error;
use serde_json::json;
use futures::StreamExt;
use actix_web::web::Bytes;
use crate::services::ragflow_service::RAGFlowError;

#[derive(Debug, Deserialize)]
pub struct InitChatRequest {
    pub user_id: String,
}

#[derive(Debug, Serialize)]
pub struct InitChatResponse {
    pub success: bool,
    pub conversation_id: String,
    pub message: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SendMessageRequest {
    pub message: String,
    pub quote: Option<bool>,
    pub doc_ids: Option<Vec<String>>,
    pub stream: Option<bool>,
}

// Implement ResponseError for RAGFlowError
impl ResponseError for RAGFlowError {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::InternalServerError()
            .json(json!({"error": self.to_string()}))
    }
}

/// Handler for sending a message to the RAGFlow service.
pub async fn send_message(
    state: web::Data<AppState>,
    request: web::Json<SendMessageRequest>,
) -> impl Responder {
    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "RAGFlow service is not available"
        }))
    };

    let conversation_id = state.ragflow_conversation_id.clone();
    match ragflow_service.send_message(
        conversation_id,
        request.message.clone(),
        request.quote.unwrap_or(false),
        request.doc_ids.clone(),
        request.stream.unwrap_or(false),
    ).await {
        Ok(response_stream) => {
            let mapped_stream = response_stream.map(|result| {
                result.map(|answer| {
                    let json_response = json!({
                        "answer": answer,
                        "success": true
                    });
                    Bytes::from(json_response.to_string())
                })
                .map_err(|e| actix_web::error::ErrorInternalServerError(e))
            });
            HttpResponse::Ok().streaming(mapped_stream)
        },
        Err(e) => {
            error!("Error sending message: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to send message: {}", e)
            }))
        }
    }
}

/// Handler for initiating a new chat conversation.
pub async fn init_chat(
    state: web::Data<AppState>,
    request: web::Json<InitChatRequest>,
) -> impl Responder {
    let user_id = request.user_id.clone();
    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "RAGFlow service is not available"
        }))
    };

    match ragflow_service.create_conversation(user_id.clone()).await {
        Ok(conversation_id) => HttpResponse::Ok().json(InitChatResponse {
            success: true,
            conversation_id,
            message: None,
        }),
        Err(e) => {
            error!("Failed to initialize chat: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to initialize chat: {}", e)
            }))
        }
    }
}

/// Handler for retrieving chat history.
pub async fn get_chat_history(
    _state: web::Data<AppState>,
    _conversation_id: web::Path<String>,
) -> impl Responder {
    HttpResponse::NotImplemented().json(json!({
        "message": "Chat history retrieval is not implemented"
    }))
}

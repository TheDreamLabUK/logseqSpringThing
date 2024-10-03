use actix_web::{web, HttpResponse};
use crate::AppState;
use serde::{Serialize, Deserialize};
use crate::services::ragflow_service::{Message, ChatResponse};
use log::{info, error};

#[derive(Serialize, Deserialize)]
pub struct MessageRequest {
    pub conversation_id: String,
    pub messages: Vec<Message>,
    pub quote: Option<bool>,
    pub doc_ids: Option<Vec<String>>,
    pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct InitChatRequest {
    pub user_id: String,
}

/// Response structure for initiating a chat.
#[derive(Serialize)]
pub struct InitChatResponse {
    pub success: bool,
    pub conversation_id: String,
    pub message: Option<String>,
}

/// Handler for sending a message to the RAGFlow service.
pub async fn send_message(state: web::Data<AppState>, msg: web::Json<MessageRequest>) -> HttpResponse {
    let message_content = &msg.messages.last().unwrap().content;
    let quote = msg.quote.unwrap_or(false);
    let doc_ids = msg.doc_ids.clone();
    let stream = msg.stream.unwrap_or(false);

    info!("Sending message to RAGFlow: {}", message_content);
    info!("Quote: {}, Stream: {}, Doc IDs: {:?}", quote, stream, doc_ids);

    match state.ragflow_service.send_message(msg.conversation_id.clone(), message_content.clone(), quote, doc_ids, stream).await {
        Ok(response) => HttpResponse::Ok().json(response),
        Err(e) => {
            error!("Error sending message: {}", e);
            HttpResponse::InternalServerError().json(ChatResponse {
                retcode: 1,
                data: crate::services::ragflow_service::ChatResponseData {
                    message: vec![Message {
                        role: "system".to_string(),
                        content: format!("Failed to send message: {}", e),
                    }],
                },
            })
        }
    }
}

/// Handler for initiating a new chat conversation.
pub async fn init_chat(state: web::Data<AppState>, req: web::Json<InitChatRequest>) -> HttpResponse {
    let user_id = &req.user_id;

    info!("Initializing chat for user: {}", user_id);

    match state.ragflow_service.create_conversation(user_id.clone()).await {
        Ok(conversation_id) => HttpResponse::Ok().json(InitChatResponse {
            success: true,
            conversation_id,
            message: None,
        }),
        Err(e) => {
            error!("Error initiating chat: {}", e);
            HttpResponse::InternalServerError().json(InitChatResponse {
                success: false,
                conversation_id: "".to_string(),
                message: Some(format!("Failed to initialize chat: {}", e)),
            })
        }
    }
}

/// Handler for retrieving chat history.
pub async fn get_chat_history(state: web::Data<AppState>, path: web::Path<String>) -> HttpResponse {
    let conversation_id = path.into_inner();
    info!("Retrieving chat history for conversation: {}", conversation_id);

    match state.ragflow_service.get_chat_history(conversation_id).await {
        Ok(history) => HttpResponse::Ok().json(history),
        Err(e) => {
            error!("Error retrieving chat history: {}", e);
            HttpResponse::InternalServerError().json(ChatResponse {
                retcode: 1,
                data: crate::services::ragflow_service::ChatResponseData {
                    message: vec![Message {
                        role: "system".to_string(),
                        content: format!("Failed to fetch chat history: {}", e),
                    }],
                },
            })
        }
    }
}

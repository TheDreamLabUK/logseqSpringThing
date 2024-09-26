use actix_web::{web, HttpResponse};
use crate::AppState;
use serde::{Serialize, Deserialize};
use crate::services::ragflow_service::{Message, ChatResponse};
use log::{info, error};

#[derive(Serialize, Deserialize)]
pub struct MessageRequest {
    pub message: String,
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
}

/// Handler for sending a message to the RAGFlow service.
pub async fn send_message(state: web::Data<AppState>, msg: web::Json<MessageRequest>, conversation_id: web::Path<String>) -> HttpResponse {
    let message_content = msg.message.clone();

    info!("Sending message to RAGFlow: {}", message_content);

    match state.ragflow_service.send_message(conversation_id.into_inner(), message_content).await {
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
pub async fn init_chat(state: web::Data<AppState>, user_id: web::Json<InitChatRequest>) -> HttpResponse {
    let user_id = user_id.into_inner().user_id;

    info!("Initializing chat for user: {}", user_id);

    match state.ragflow_service.create_conversation(user_id).await {
        Ok(conversation_id) => HttpResponse::Ok().json(InitChatResponse {
            success: true,
            conversation_id,
        }),
        Err(e) => {
            error!("Error initiating chat: {}", e);
            HttpResponse::InternalServerError().json(InitChatResponse {
                success: false,
                conversation_id: "".to_string(),
            })
        }
    }
}

/// Handler for retrieving chat history.
pub async fn get_chat_history(state: web::Data<AppState>, conversation_id: web::Path<String>) -> HttpResponse {
    info!("Retrieving chat history for conversation: {}", conversation_id);

    match state.ragflow_service.get_chat_history(conversation_id.into_inner()).await {
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

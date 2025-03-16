use actix_web::{web, HttpResponse, ResponseError, Responder};
use crate::AppState;
use serde::{Serialize, Deserialize};
use log::{error, info};
use serde_json::json;
use futures::StreamExt;
use actix_web::web::Bytes;
use crate::services::ragflow_service::RAGFlowError;
use actix_web::web::ServiceConfig;
use crate::types::speech::SpeechOptions;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionRequest {
    pub user_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionResponse {
    pub success: bool,
    pub session_id: String,
    pub message: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageRequest {
    pub question: String,
    pub stream: Option<bool>,
    pub session_id: Option<String>,
    pub enable_tts: Option<bool>,
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

    // Get session ID from request or use the default one from app state if not provided
    let session_id = match &request.session_id {
        Some(id) => id.clone(),
        None => state.ragflow_session_id.clone(),
    };

    let enable_tts = request.enable_tts.unwrap_or(false);
    // The quote and doc_ids parameters are not used in the new API
    match ragflow_service.send_message(
        session_id,
        request.question.clone(),
        false, // quote parameter (unused)
        None,  // doc_ids parameter (unused)
        request.stream.unwrap_or(true),
    ).await {
        Ok(response_stream) => {
            // Check if TTS is enabled and speech service exists
            if enable_tts {
                if let Some(speech_service) = &state.speech_service {
                    let speech_service = speech_service.clone();
                    // Clone the question to pass to TTS
                    let question = request.question.clone();
                    // Spawn a task to process TTS in the background
                    actix_web::rt::spawn(async move {
                        let speech_options = SpeechOptions::default();
                        // The exact question will be sent to TTS
                        if let Err(e) = speech_service.text_to_speech(question, speech_options).await {
                            error!("Error processing TTS: {:?}", e);
                        }
                    });
                }
            }
            
            // Continue with normal text response handling
            let enable_tts = enable_tts; // Clone for capture in closure
            let mapped_stream = response_stream.map(move |result| {
                result.map(|answer| {
                    // Skip empty messages (like the end marker)
                    if answer.is_empty() {
                        return Bytes::new();
                    }
                    
                    // If TTS is enabled, send answer to speech service
                    if enable_tts {
                        if let Some(speech_service) = &state.speech_service {
                            let speech_service = speech_service.clone();
                            let speech_options = SpeechOptions::default();
                            let answer_clone = answer.clone();
                            actix_web::rt::spawn(async move {
                                if let Err(e) = speech_service.text_to_speech(answer_clone, speech_options).await {
                                    error!("Error processing TTS for answer: {:?}", e);
                                }
                            });
                        }
                    }
                    
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

/// Handler for initiating a new session with RAGFlow agent.
pub async fn create_session(
    state: web::Data<AppState>,
    request: web::Json<CreateSessionRequest>,
) -> impl Responder {
    let user_id = request.user_id.clone();
    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "RAGFlow service is not available"
        }))
    };

    match ragflow_service.create_session(user_id.clone()).await {
        Ok(session_id) => {
            // Store the session ID in the AppState for future use
            // We can't directly modify AppState through an Arc, but we can clone it and create a new state
            // For now, we'll log this situation but not update the shared state
            // In a production environment, you'd want a better solution like using RwLock for the session_id
            info!(
                "Created new RAGFlow session: {}. Note: session ID cannot be stored in shared AppState.",
                session_id
            );
            // Use the session_id directly from the request in subsequent calls
            
            HttpResponse::Ok().json(CreateSessionResponse {
                success: true,
                session_id,
                message: None,
            })
        },
        Err(e) => {
            error!("Failed to initialize chat: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to initialize chat: {}", e)
            }))
        }
    }
}

/// Handler for retrieving session history.
pub async fn get_session_history(
    state: web::Data<AppState>,
    session_id: web::Path<String>,
) -> impl Responder {
    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "RAGFlow service is not available"
        }))
    };

    match ragflow_service.get_session_history(session_id.to_string()).await {
        Ok(history) => HttpResponse::Ok().json(history),
        Err(e) => {
            error!("Failed to get session history: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to get chat history: {}", e)
            }))
        }
    }
}

/// Configure RAGFlow API routes
pub fn config(cfg: &mut ServiceConfig) {
    cfg.service(
        web::scope("/ragflow")
            .route("/session", web::post().to(create_session))
            .route("/message", web::post().to(send_message))
            .route("/history/{session_id}", web::get().to(get_session_history))
    );
}

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
use crate::models::ragflow_chat::{RagflowChatRequest, RagflowChatResponse};
use actix_web::HttpRequest;

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
async fn handle_ragflow_chat(
    state: web::Data<AppState>,
    req: HttpRequest, // To get headers for auth
    payload: web::Json<RagflowChatRequest>,
) -> impl Responder {
    // Authentication: Check for power user
    let pubkey = match req.headers().get("X-Nostr-Pubkey").and_then(|v| v.to_str().ok()) {
        Some(pk) => pk.to_string(),
        None => return HttpResponse::Unauthorized().json(json!({"error": "Missing X-Nostr-Pubkey header"})),
    };
    let token = match req.headers().get("Authorization").and_then(|v| v.to_str().ok().map(|s| s.trim_start_matches("Bearer "))) {
        Some(t) => t.to_string(),
        None => return HttpResponse::Unauthorized().json(json!({"error": "Missing Authorization token"})),
    };

    if let Some(nostr_service) = &state.nostr_service {
        if !nostr_service.validate_session(&pubkey, &token).await {
            return HttpResponse::Unauthorized().json(json!({"error": "Invalid session token"}));
        }
        // Accessing feature checks through AppState methods
        let has_ragflow_specific_access = state.has_feature_access(&pubkey, "ragflow");
        let is_power_user = state.is_power_user(&pubkey);

        if !is_power_user && !has_ragflow_specific_access {
            return HttpResponse::Forbidden().json(json!({"error": "This feature requires power user access or specific RAGFlow permission"}));
        }
    } else {
        // This case should ideally not be reached if nostr_service is integral
        // and initialized properly. Consider logging a warning or error.
        error!("Nostr service not available during chat handling for pubkey: {}", pubkey);
        return HttpResponse::InternalServerError().json(json!({"error": "Nostr service not available"}));
    }

    info!("[handle_ragflow_chat] Checking RAGFlow service availability. Is Some: {}", state.ragflow_service.is_some()); // ADDED LOG

    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => {
            error!("[handle_ragflow_chat] RAGFlow service is None, returning 503."); // ADDED LOG
            return HttpResponse::ServiceUnavailable().json(json!({"error": "RAGFlow service not available"}));
        }
    };

    info!("[handle_ragflow_chat] RAGFlow service is Some. Proceeding."); // ADDED LOG

    let mut session_id = payload.session_id.clone();
    if session_id.is_none() {
        // Create a new session if none provided. Using pubkey as user_id for RAGFlow session.
        match ragflow_service.create_session(pubkey.clone()).await {
            Ok(new_sid) => {
                info!("Created new RAGFlow session {} for pubkey {}", new_sid, pubkey);
                session_id = Some(new_sid);
            }
            Err(e) => {
                error!("Failed to create RAGFlow session for pubkey {}: {}", pubkey, e);
                return HttpResponse::InternalServerError().json(json!({"error": format!("Failed to create RAGFlow session: {}", e)}));
            }
        }
    }

    // We've ensured it's Some by now, or returned an error.
    let current_session_id = session_id.expect("Session ID should be Some at this point");

    let stream_preference = payload.stream.unwrap_or(false); // Default to false if not provided
    match ragflow_service.send_chat_message(current_session_id.clone(), payload.question.clone(), stream_preference).await {
        Ok((answer, final_session_id)) => {
            HttpResponse::Ok().json(RagflowChatResponse {
                answer,
                session_id: final_session_id, // RAGFlow service send_chat_message returns the session_id it used
            })
        }
        Err(e) => {
            error!("Error communicating with RAGFlow for session {}: {}", current_session_id, e);
            HttpResponse::InternalServerError().json(json!({"error": format!("RAGFlow communication error: {}", e)}))
        }
    }
}
pub fn config(cfg: &mut ServiceConfig) {
    cfg.service(
        web::scope("/ragflow")
            .route("/session", web::post().to(create_session)) // Existing
            .route("/message", web::post().to(send_message))   // Existing (streaming)
            .route("/chat", web::post().to(handle_ragflow_chat)) // New REST chat endpoint
            .route("/history/{session_id}", web::get().to(get_session_history)) // Existing
    );
}

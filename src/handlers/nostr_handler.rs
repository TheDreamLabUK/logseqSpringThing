use crate::app_state::AppState;
use crate::models::protected_settings::{NostrUser, ApiKeys};
use crate::services::nostr_service::{NostrService, AuthEvent, NostrError};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub user: NostrUser,
    pub token: String,
    pub expires_at: i64,
}

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub is_power_user: bool,
}

#[derive(Debug, Deserialize)]
pub struct ApiKeysRequest {
    pub perplexity: Option<String>,
    pub openai: Option<String>,
    pub ragflow: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ValidateRequest {
    pub pubkey: String,
    pub token: String,
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/auth/nostr")
            .route("", web::post().to(login))
            .route("", web::delete().to(logout))
            .route("/verify", web::post().to(verify))
            .route("/refresh", web::post().to(refresh))
            .route("/api-keys", web::post().to(update_api_keys))
            .route("/api-keys", web::get().to(get_api_keys))
    );
}

async fn login(
    event: web::Json<AuthEvent>,
    nostr_service: web::Data<NostrService>,
) -> Result<HttpResponse, Error> {
    match nostr_service.verify_auth_event(event.into_inner()).await {
        Ok(user) => {
            let token = user.session_token.clone().unwrap_or_default();
            let expires_at = user.last_seen + std::env::var("NOSTR_TOKEN_EXPIRY")
                .unwrap_or_else(|_| "3600".to_string())
                .parse::<i64>()
                .unwrap_or(3600);

            Ok(HttpResponse::Ok().json(AuthResponse {
                user,
                token,
                expires_at,
            }))
        }
        Err(NostrError::InvalidSignature) => {
            Ok(HttpResponse::Unauthorized().json(json!({
                "error": "Invalid signature"
            })))
        }
        Err(e) => {
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Authentication error: {}", e)
            })))
        }
    }
}

async fn logout(
    req: web::Json<ValidateRequest>,
    nostr_service: web::Data<NostrService>,
) -> Result<HttpResponse, Error> {
    // Validate session before logout
    if !nostr_service.validate_session(&req.pubkey, &req.token).await {
        return Ok(HttpResponse::Unauthorized().json(json!({
            "error": "Invalid session"
        })));
    }

    match nostr_service.logout(&req.pubkey).await {
        Ok(_) => Ok(HttpResponse::Ok().json(json!({
            "message": "Logged out successfully"
        }))),
        Err(e) => Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Logout error: {}", e)
        }))),
    }
}

async fn verify(
    req: web::Json<ValidateRequest>,
    nostr_service: web::Data<NostrService>,
) -> Result<HttpResponse, Error> {
    let is_valid = nostr_service.validate_session(&req.pubkey, &req.token).await;
    let is_power_user = if is_valid {
        nostr_service.is_power_user(&req.pubkey).await
    } else {
        false
    };

    Ok(HttpResponse::Ok().json(VerifyResponse {
        valid: is_valid,
        is_power_user,
    }))
}

async fn refresh(
    req: web::Json<ValidateRequest>,
    nostr_service: web::Data<NostrService>,
) -> Result<HttpResponse, Error> {
    // First validate the current session
    if !nostr_service.validate_session(&req.pubkey, &req.token).await {
        return Ok(HttpResponse::Unauthorized().json(json!({
            "error": "Invalid session"
        })));
    }

    match nostr_service.refresh_session(&req.pubkey).await {
        Ok(new_token) => {
            if let Some(user) = nostr_service.get_user(&req.pubkey).await {
                let expires_at = user.last_seen + std::env::var("NOSTR_TOKEN_EXPIRY")
                    .unwrap_or_else(|_| "3600".to_string())
                    .parse::<i64>()
                    .unwrap_or(3600);

                Ok(HttpResponse::Ok().json(AuthResponse {
                    user,
                    token: new_token,
                    expires_at,
                }))
            } else {
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "User not found after refresh"
                })))
            }
        }
        Err(e) => Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Session refresh error: {}", e)
        }))),
    }
}

async fn update_api_keys(
    req: web::Json<ApiKeysRequest>,
    nostr_service: web::Data<NostrService>,
    pubkey: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let api_keys = ApiKeys {
        perplexity: req.perplexity.clone(),
        openai: req.openai.clone(),
        ragflow: req.ragflow.clone(),
    };

    match nostr_service.update_user_api_keys(&pubkey, api_keys).await {
        Ok(user) => Ok(HttpResponse::Ok().json(user)),
        Err(NostrError::UserNotFound) => {
            Ok(HttpResponse::NotFound().json(json!({
                "error": "User not found"
            })))
        }
        Err(NostrError::PowerUserOperation) => {
            Ok(HttpResponse::Forbidden().json(json!({
                "error": "Cannot update API keys for power users"
            })))
        }
        Err(e) => {
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to update API keys: {}", e)
            })))
        }
    }
}

async fn get_api_keys(
    state: web::Data<AppState>,
    pubkey: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let protected_settings = state.protected_settings.read().await;
    let api_keys = protected_settings.get_api_keys(&pubkey);
    
    Ok(HttpResponse::Ok().json(api_keys))
}

// Add the handler to app_state initialization
pub fn init_nostr_service(app_state: &mut AppState) {
    let nostr_service = NostrService::new();
    
    // Start session cleanup task
    let service_clone = nostr_service.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600)); // Every hour
        loop {
            interval.tick().await;
            service_clone.cleanup_sessions(24).await; // Clean up sessions older than 24 hours
        }
    });

    app_state.nostr_service = Some(web::Data::new(nostr_service));
}
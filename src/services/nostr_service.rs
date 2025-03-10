use crate::models::protected_settings::{NostrUser, ApiKeys};
use crate::config::feature_access::FeatureAccess;
use nostr_sdk::{
    prelude::*,
    event::Error as EventError
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::Utc;
use thiserror::Error;
use log::{debug, error, info};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum NostrError {
    #[error("Invalid event: {0}")]
    InvalidEvent(String),
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("User not found")]
    UserNotFound,
    #[error("Invalid token")]
    InvalidToken,
    #[error("Session expired")]
    SessionExpired,
    #[error("Power user operation not allowed")]
    PowerUserOperation,
    #[error("Nostr event error: {0}")]
    NostrError(#[from] EventError),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthEvent {
    pub id: String,
    pub pubkey: String,
    pub content: String,
    pub sig: String,
    pub created_at: i64,
    pub kind: i32,
    pub tags: Vec<Vec<String>>,
}

#[derive(Clone)]
pub struct NostrService {
    users: Arc<RwLock<HashMap<String, NostrUser>>>,
    power_user_pubkeys: Vec<String>,
    token_expiry: i64,
    feature_access: Arc<RwLock<FeatureAccess>>,
}

impl NostrService {
    pub fn new() -> Self {
        let power_users = std::env::var("POWER_USER_PUBKEYS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let token_expiry = std::env::var("AUTH_TOKEN_EXPIRY")
            .unwrap_or_else(|_| "3600".to_string())
            .parse()
            .unwrap_or(3600);

        let feature_access = Arc::new(RwLock::new(FeatureAccess::from_env()));
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
            power_user_pubkeys: power_users,
            feature_access,
            token_expiry,
        }
    }

    pub async fn verify_auth_event(&self, event: AuthEvent) -> Result<NostrUser, NostrError> {
        // Convert to Nostr Event for verification
        // Convert to JSON string and parse as Nostr Event
        debug!("Verifying auth event with id: {} and pubkey: {}", event.id, event.pubkey);

        let json_str = match serde_json::to_string(&event) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to serialize auth event with id {}: {}", event.id, e);
                return Err(NostrError::JsonError(e));
            }
        };

        debug!("Event JSON for verification (truncated): {}...", 
               if json_str.len() > 100 { &json_str[0..100] } else { &json_str });

        let nostr_event = match Event::from_json(&json_str) {
            Ok(e) => e,
            Err(e) => {
                error!("Failed to parse Nostr event for pubkey {}: {}", event.pubkey, e);
                return Err(NostrError::InvalidEvent(format!("Parse error for event {}: {}", event.id, e)));
            }
        };

        if let Err(e) = nostr_event.verify() {
            error!("Signature verification failed for pubkey {}: {}", event.pubkey, e);
            return Err(NostrError::InvalidSignature);
        }

        // Register new user if not already registered
        let mut feature_access = self.feature_access.write().await;
        if feature_access.register_new_user(&event.pubkey) {
            info!("Registered new user with basic access: {}", event.pubkey);
        }

        let now = Utc::now();
        let is_power_user = self.power_user_pubkeys.contains(&event.pubkey);

        // Generate session token
        let session_token = Uuid::new_v4().to_string();

        let user = NostrUser {
            pubkey: event.pubkey.clone(),
            npub: nostr_event.pubkey.to_bech32()
                .map_err(|_| NostrError::NostrError(EventError::InvalidId))?,
            is_power_user,
            api_keys: ApiKeys::default(),
            last_seen: now.timestamp(),
            session_token: Some(session_token),
        };

        // Log successful user creation
        info!("Created/updated user: pubkey={}, is_power_user={}", user.pubkey, user.is_power_user);

        // Store or update user
        let mut users = self.users.write().await;
        users.insert(user.pubkey.clone(), user.clone());

        Ok(user)
    }

    pub async fn get_user(&self, pubkey: &str) -> Option<NostrUser> {
        let users = self.users.read().await;
        users.get(pubkey).cloned()
    }

    pub async fn update_user_api_keys(&self, pubkey: &str, api_keys: ApiKeys) -> Result<NostrUser, NostrError> {
        let mut users = self.users.write().await;
        
        if let Some(user) = users.get_mut(pubkey) {
            if user.is_power_user {
                return Err(NostrError::PowerUserOperation);
            }
            user.api_keys = api_keys;
            user.last_seen = Utc::now().timestamp();
            Ok(user.clone())
        } else {
            Err(NostrError::UserNotFound)
        }
    }

    pub async fn validate_session(&self, pubkey: &str, token: &str) -> bool {
        if let Some(user) = self.get_user(pubkey).await {
            if let Some(session_token) = user.session_token {
                let now = Utc::now().timestamp();
                if now - user.last_seen <= self.token_expiry {
                    return session_token == token;
                }
            }
        }
        false
    }

    pub async fn refresh_session(&self, pubkey: &str) -> Result<String, NostrError> {
        let mut users = self.users.write().await;
        
        if let Some(user) = users.get_mut(pubkey) {
            let now = Utc::now().timestamp();
            let new_token = Uuid::new_v4().to_string();
            user.session_token = Some(new_token.clone());
            user.last_seen = now;
            Ok(new_token)
        } else {
            Err(NostrError::UserNotFound)
        }
    }

    pub async fn logout(&self, pubkey: &str) -> Result<(), NostrError> {
        let mut users = self.users.write().await;
        
        if let Some(user) = users.get_mut(pubkey) {
            user.session_token = None;
            user.last_seen = Utc::now().timestamp();
            Ok(())
        } else {
            Err(NostrError::UserNotFound)
        }
    }

    pub async fn cleanup_sessions(&self, max_age_hours: i64) {
        let now = Utc::now();
        let mut users = self.users.write().await;
        
        users.retain(|_, user| {
            let age = now.timestamp() - user.last_seen;
            age < (max_age_hours * 3600)
        });
    }

    pub async fn is_power_user(&self, pubkey: &str) -> bool {
        if let Some(user) = self.get_user(pubkey).await {
            user.is_power_user
        } else {
            false
        }
    }
}

impl Default for NostrService {
    fn default() -> Self {
        Self::new()
    }
}
use crate::app_state::AppState;
use crate::models::{UISettings, UserSettings};
use crate::config::Settings; // <-- Import the full Settings struct
use crate::handlers::socket_flow_handler::ClientManager; // For WebSocket broadcasting
use actix_web::{web, Error, HttpResponse, HttpRequest};
use chrono::Utc;
use serde_json::{Value, json}; // Value for payload, json for the json! macro
use crate::config::feature_access::FeatureAccess;
use log::{info, error, warn, debug};
use std::time::Instant;

// Add a new endpoint to clear the settings cache for a user
async fn clear_user_settings_cache(
    req: HttpRequest,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    // Get pubkey from header
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    
    // Check if user has permission
    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted to clear settings cache without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }
    
    UserSettings::clear_cache(&pubkey);
    info!("Cleared settings cache for user {}", pubkey);
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "message": "Settings cache cleared"
    })))
}

// Add a new endpoint for admin to clear all settings caches
async fn clear_all_settings_cache(
    req: HttpRequest,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    // Get pubkey from header
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    
    // Only power users can clear all caches
    if !feature_access.is_power_user(&pubkey) {
        warn!("Non-power user {} attempted to clear all settings caches", pubkey);
        return Ok(HttpResponse::Forbidden().body("Only power users can clear all settings caches"));
    }
    
    UserSettings::clear_all_cache();
    info!("Power user {} cleared all settings caches", pubkey);
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "message": "All settings caches cleared"
    })))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/user-settings")
            .route(web::get().to(get_public_settings))
            .route(web::post().to(update_settings))
    ).service(
        web::resource("/user-settings/sync")
            .route(web::get().to(get_user_settings))
            .route(web::post().to(update_user_settings)) // This now points to the new function
    ).service(
        web::resource("/user-settings/clear-cache")
            .route(web::post().to(clear_user_settings_cache))
    ).service(
        web::resource("/admin/settings/clear-all-cache")
            .route(web::post().to(clear_all_settings_cache))
    );
}

pub async fn get_public_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = state.settings.read().await;
    
    // Convert to UI settings
    let ui_settings = UISettings::from(&*settings_guard);
    
    Ok(HttpResponse::Ok().json(&ui_settings))
}

async fn get_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let start_time = Instant::now();
    
    // Get pubkey from header
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    
    debug!("Processing settings request for user: {}", pubkey);
    
    // Check if user has permission using FeatureAccess
    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted to sync settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    let is_power_user = feature_access.is_power_user(&pubkey);
    let result;

    if is_power_user {
        // Power users get settings from the global settings file
        let settings_guard = state.settings.read().await;
        let ui_settings = UISettings::from(&*settings_guard);
        debug!("Returning global settings for power user {}", pubkey);
        result = Ok(HttpResponse::Ok().json(ui_settings));
    } else {
        // Regular users get their personal settings or defaults
        // This will use the cache if available due to our UserSettings::load implementation
        let user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {} with default settings", pubkey);
            UserSettings::new(&pubkey, UISettings::default())
        });
        result = Ok(HttpResponse::Ok().json(&user_settings.settings));
    }
    
    // Log the time taken to process this request
    let elapsed = start_time.elapsed();
    debug!("Settings request for {} processed in {:?}", pubkey, elapsed);
    
    result
}

// --- START OF USER PROVIDED update_user_settings ---
async fn update_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    payload: web::Json<Value>, // Still receive raw JSON Value
) -> Result<HttpResponse, Error> {
    let start_time = Instant::now();

    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            // If no pubkey, can't determine user type or save user-specific settings
            warn!("Update settings request received without Nostr pubkey.");
            // This behavior differs from the old version which returned default settings.
            // Returning BadRequest as per user's new code.
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey for settings update"));
        }
    };

    debug!("Processing settings update for user: {}", pubkey);

    // Check if user has permission to sync settings (power users implicitly can)
    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted to sync settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    // *** CHANGE: Deserialize into the full Settings struct ***
    let received_settings: Settings = match serde_json::from_value(payload.into_inner()) {
        Ok(settings) => settings,
        Err(e) => {
            error!("Failed to deserialize incoming settings payload for user {}: {}", pubkey, e);
            // Log the payload that failed to deserialize for debugging
            // Be cautious logging sensitive data like API keys if they are present
            // debug!("Failing payload: {:?}", payload); // Uncomment cautiously for debugging
            return Ok(HttpResponse::BadRequest().body(format!("Invalid settings format: {}", e)));
        }
    };

    let is_power_user = feature_access.is_power_user(&pubkey);
    let result;
    let settings_to_broadcast: Option<UISettings>;

    if is_power_user {
        // Power users update the global settings file
        let mut settings_guard = state.settings.write().await;

        // *** CAREFUL MERGE: Update only the parts the client should modify ***
        // Overwrite visualization and XR completely as they are UI-driven
        settings_guard.visualization = received_settings.visualization;
        settings_guard.xr = received_settings.xr;

        // Selectively merge parts of system settings (avoid overwriting sensitive network/security)
        settings_guard.system.websocket = received_settings.system.websocket;
        settings_guard.system.debug = received_settings.system.debug;
        // DO NOT merge system.network or system.security from client

        // Update AI service settings if present in received data
        // Use if let to handle potentially missing fields gracefully
        if let Some(ragflow) = received_settings.ragflow { settings_guard.ragflow = Some(ragflow); }
        if let Some(perplexity) = received_settings.perplexity { settings_guard.perplexity = Some(perplexity); }
        if let Some(openai) = received_settings.openai { settings_guard.openai = Some(openai); }
        if let Some(kokoro) = received_settings.kokoro { settings_guard.kokoro = Some(kokoro); }
        // DO NOT update auth settings from client

        // Save the updated full settings
        if let Err(e) = settings_guard.save() { // Save the merged settings
            error!("Failed to save global settings after update from {}: {}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
        }

        info!("Power user {} updated global settings", pubkey);
        // Extract UISettings *after* saving the merged full settings
        let updated_ui_settings = UISettings::from(&*settings_guard);
        settings_to_broadcast = Some(updated_ui_settings.clone());
        result = Ok(HttpResponse::Ok().json(updated_ui_settings)); // Respond with UI subset

    } else {
        // Regular users update their personal settings file
        // They send their desired UISettings, which we deserialized into the full Settings struct above.
        // We need to extract the relevant UISettings part for saving their personal file.
        let ui_settings_from_payload = UISettings::from(&received_settings); // Extract the relevant UI parts

        let mut user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {}", pubkey);
            UserSettings::new(&pubkey, UISettings::default()) // Create with default UI settings
        });

        user_settings.settings = ui_settings_from_payload; // Update with the settings sent by the user
        user_settings.last_modified = Utc::now().timestamp();

        if let Err(e) = user_settings.save() { // Save personal settings
            error!("Failed to save user settings for {}: {}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save user settings: {}", e)));
        }

        debug!("User {} updated their settings", pubkey);
        settings_to_broadcast = None; // Don't broadcast personal settings changes globally
        result = Ok(HttpResponse::Ok().json(&user_settings.settings)); // Respond with their saved settings
    }

    // --- Broadcast Logic ---
    if let Some(settings_payload) = settings_to_broadcast {
        if let Some(client_manager_ref) = state.client_manager.as_ref() { // Use as_ref() for Option<Arc<T>>
            let broadcast_message = json!({
                "type": "settings_updated",
                "payload": settings_payload
            });
            match serde_json::to_string(&broadcast_message) {
                Ok(msg_str) => {
                    info!("Broadcasting settings update to all clients.");
                    client_manager_ref.broadcast_text_message(msg_str).await;
                }
                Err(e) => {
                    error!("Failed to serialize settings broadcast message: {}", e);
                }
            }
        } else {
            warn!("ClientManager not found in AppState, cannot broadcast settings.");
        }
    }
    // --- End Broadcast Logic ---

    let elapsed = start_time.elapsed();
    debug!("Settings update for {} processed in {:?}", pubkey, elapsed);

    result
}
// --- END OF USER PROVIDED update_user_settings ---

async fn update_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    // Get pubkey from header
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Attempt to update settings without authentication");
            // For updates, we do require authentication
            // This prevents unauthenticated users from modifying settings
            // They can still read public settings via get endpoints
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };

    // Check if user is a power user
    let is_power_user = feature_access.is_power_user(&pubkey);

    if !is_power_user {
        warn!("Non-power user {} attempted to modify global settings", pubkey);
        return Ok(HttpResponse::Forbidden().body("Only power users can modify global settings"));
    }

    // Parse and validate settings
    // This endpoint /user-settings (without /sync) was likely intended for UISettings only.
    // If power users are now sending full Settings to /user-settings/sync,
    // this /user-settings endpoint might need review or deprecation if it's redundant.
    // For now, keeping its existing logic of expecting UISettings.
    let ui_settings: UISettings = match serde_json::from_value(payload.into_inner()) {
        Ok(settings) => settings,
        Err(e) => return Ok(HttpResponse::BadRequest().body(format!("Invalid settings format: {}", e)))
    };

    let mut settings_guard = state.settings.write().await;
    ui_settings.merge_into_settings(&mut settings_guard);
    
    if let Err(e) = settings_guard.save() {
        error!("Failed to save global settings: {}", e);
        return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
    }
    
    info!("Power user {} updated global settings via /user-settings endpoint", pubkey);
    let updated_ui_settings = UISettings::from(&*settings_guard);
    // Consider broadcasting here as well if this endpoint is still actively used for updates.
    Ok(HttpResponse::Ok().json(updated_ui_settings))
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings = app_state.settings.read().await;
    let ui_settings = UISettings::from(&*settings);
    Ok(HttpResponse::Ok().json(&ui_settings.visualization))
}

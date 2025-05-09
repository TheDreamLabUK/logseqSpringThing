use crate::app_state::AppState;
use crate::models::{UISettings, UserSettings};
use crate::models::ui_settings::UISystemSettings; // Correct import
// Import both settings types and alias the client-facing one
use crate::config::{AppFullSettings, Settings as ClientFacingSettings};
// use crate::handlers::socket_flow_handler::ClientManager; // Needed for broadcast - ClientManager is now accessed via AppState
use actix_web::{web, Error, HttpResponse, HttpRequest};
use chrono::Utc;
use serde_json::json;
use crate::config::feature_access::FeatureAccess;
use log::{info, error, warn, debug};
use std::time::Instant;

// Helper function to convert AppFullSettings to UISettings (requires From impl update)
// This assumes the From impl is updated in models/ui_settings.rs
fn convert_to_ui_settings(full_settings: &AppFullSettings) -> UISettings {
    UISettings::from(full_settings) // Rely on the From trait implementation
}


// --- Cache Clearing Endpoints (Unaffected by Settings struct changes) ---

async fn clear_user_settings_cache(
    req: HttpRequest,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted to clear settings cache without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }
    UserSettings::clear_cache(&pubkey);
    info!("Cleared settings cache for user {}", pubkey);
    Ok(HttpResponse::Ok().json(json!({ "status": "success", "message": "Settings cache cleared" })))
}

async fn clear_all_settings_cache(
    req: HttpRequest,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    if !feature_access.is_power_user(&pubkey) {
        warn!("Non-power user {} attempted to clear all settings caches", pubkey);
        return Ok(HttpResponse::Forbidden().body("Only power users can clear all settings caches"));
    }
    UserSettings::clear_all_cache();
    info!("Power user {} cleared all settings caches", pubkey);
    Ok(HttpResponse::Ok().json(json!({ "status": "success", "message": "All settings caches cleared" })))
}

// --- Configuration ---

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/user-settings")
            .route(web::get().to(get_public_settings))
            .route(web::post().to(update_settings))
    ).service(
        web::resource("/user-settings/sync")
            .route(web::get().to(get_user_settings))
            .route(web::post().to(update_user_settings)) // This now points to the updated function
    ).service(
        web::resource("/user-settings/clear-cache")
            .route(web::post().to(clear_user_settings_cache))
    ).service(
        web::resource("/admin/settings/clear-all-cache")
            .route(web::post().to(clear_all_settings_cache))
    );
}

// --- GET Endpoints ---

pub async fn get_public_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = state.settings.read().await;
    let ui_settings = convert_to_ui_settings(&*settings_guard);
    Ok(HttpResponse::Ok().json(&ui_settings))
}

async fn get_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let start_time = Instant::now();
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers for get_user_settings");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    debug!("Processing get_user_settings request for user: {}", pubkey);

    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted get_user_settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    let is_power_user = feature_access.is_power_user(&pubkey);
    let result;

    if is_power_user {
        let settings_guard = state.settings.read().await;
        let ui_settings = convert_to_ui_settings(&*settings_guard);
        debug!("Returning global UI settings for power user {}", pubkey);
        result = Ok(HttpResponse::Ok().json(ui_settings));
    } else {
        let user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {} with default settings", pubkey);
            UserSettings::new(&pubkey, UISettings::default())
        });
        result = Ok(HttpResponse::Ok().json(&user_settings.settings));
    }

    let elapsed = start_time.elapsed();
    debug!("Settings request for {} processed in {:?}", pubkey, elapsed);
    result
}

// --- POST Endpoints ---

// Handles updates from the main settings UI (/user-settings/sync)
async fn update_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    // Use Actix's extractor directly for ClientFacingSettings
    payload: web::Json<ClientFacingSettings>,
) -> Result<HttpResponse, Error> {
    let start_time = Instant::now();
    // Extract the settings from the payload wrapper
    let received_client_settings = payload.into_inner();

    // Log the received settings AFTER successful extraction/deserialization by Actix
    // Use debug level and potentially truncate or selectively log fields to avoid excessive noise/sensitive data
    debug!("Successfully deserialized settings payload: {:?}", received_client_settings);


    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Update settings request received without Nostr pubkey.");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey for settings update"));
        }
    };
    debug!("Processing update_user_settings for user: {}", pubkey);

    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted update_user_settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    // No longer need manual deserialization here
    // let received_client_settings: ClientFacingSettings = ...

    let is_power_user = feature_access.is_power_user(&pubkey);
    let result;
    // let mut settings_to_broadcast: Option<UISettings> = None; // Removed as broadcast logic is removed

    if is_power_user {
        let mut settings_guard = state.settings.write().await; // Locks Arc<RwLock<AppFullSettings>>

        // --- Careful Merge from ClientFacingSettings into AppFullSettings ---
        settings_guard.visualisation = received_client_settings.visualisation;
        settings_guard.xr = received_client_settings.xr;
        settings_guard.auth = received_client_settings.auth;

        let client_sys = &received_client_settings.system;
        let server_sys = &mut settings_guard.system;

        // Map ClientWebSocketSettings into ServerFullWebSocketSettings fields
        let client_ws = &client_sys.websocket;
        let server_ws = &mut server_sys.websocket;
        server_ws.reconnect_attempts = client_ws.reconnect_attempts;
        server_ws.reconnect_delay = client_ws.reconnect_delay;
        server_ws.binary_chunk_size = client_ws.binary_chunk_size;
        server_ws.compression_enabled = client_ws.compression_enabled;
        server_ws.compression_threshold = client_ws.compression_threshold;
        server_ws.update_rate = client_ws.update_rate;

        server_sys.debug = client_sys.debug.clone(); // Clone DebugSettings
        server_sys.persist_settings = client_sys.persist_settings;

        settings_guard.ragflow = received_client_settings.ragflow;
        settings_guard.perplexity = received_client_settings.perplexity;
        settings_guard.openai = received_client_settings.openai;
        settings_guard.kokoro = received_client_settings.kokoro;
        // --- End Merge ---

        if let Err(e) = settings_guard.save() {
            error!("Failed to save global AppFullSettings after update from {}: {}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
        }

        info!("Power user {} updated global settings", pubkey);
        let updated_ui_settings = convert_to_ui_settings(&*settings_guard);
        // settings_to_broadcast = Some(updated_ui_settings.clone()); // Removed as broadcast logic is removed
        result = Ok(HttpResponse::Ok().json(updated_ui_settings));

    } else {
        // Regular users update their personal UserSettings file
        let ui_settings_from_payload = UISettings {
            visualisation: received_client_settings.visualisation,
            system: UISystemSettings { // Use the imported struct directly
                websocket: received_client_settings.system.websocket,
                debug: received_client_settings.system.debug,
            },
            xr: received_client_settings.xr,
        };

        let mut user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {}", pubkey);
            UserSettings::new(&pubkey, UISettings::default())
        });

        user_settings.settings = ui_settings_from_payload;
        user_settings.last_modified = Utc::now().timestamp();

        if let Err(e) = user_settings.save() {
            error!("Failed to save user settings for {}: {}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save user settings: {}", e)));
        }

        debug!("User {} updated their settings", pubkey);
        result = Ok(HttpResponse::Ok().json(&user_settings.settings));
    }

    // --- Broadcast Logic Removed as per new guideline (WebSockets for position/audio only) ---
    // if let Some(settings_payload) = settings_to_broadcast {
    //     // Get ClientManager from AppState
    //     let client_manager = state.ensure_client_manager().await;
    //     let broadcast_message = json!({
    //         "type": "settings_updated",
    //         "payload": settings_payload
    //     });
    //     match serde_json::to_string(&broadcast_message) {
    //         Ok(msg_str) => {
    //             info!("Broadcasting settings update to all clients.");
    //             // Dereference Arc to call method on ClientManager
    //             (*client_manager).broadcast_text_message(msg_str).await;
    //         }
    //         Err(e) => {
    //             error!("Failed to serialize settings broadcast message: {}", e);
    //         }
    //     }
    // }
    // --- End Broadcast Logic ---

    let elapsed = start_time.elapsed();
    debug!("Settings update for {} processed in {:?}", pubkey, elapsed);
    result
}

// Handles updates from the older /user-settings endpoint (needs review/deprecation?)
async fn update_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    // Use Actix's extractor directly here too, assuming it should also accept ClientFacingSettings
    payload: web::Json<ClientFacingSettings>,
) -> Result<HttpResponse, Error> {
    warn!("Received settings update via deprecated /user-settings endpoint. Use /user-settings/sync instead.");
    let received_client_settings = payload.into_inner();
    debug!("Successfully deserialized settings payload via /user-settings: {:?}", received_client_settings);

    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Attempt to update settings via /user-settings without authentication");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };

    if !feature_access.is_power_user(&pubkey) {
        warn!("Non-power user {} attempted to modify global settings via /user-settings", pubkey);
        return Ok(HttpResponse::Forbidden().body("Only power users can modify global settings"));
    }

    // Perform the same careful merge as in update_user_settings
    let mut settings_guard = state.settings.write().await; // Locks AppFullSettings

    // --- Careful Merge ---
    settings_guard.visualisation = received_client_settings.visualisation;
    settings_guard.xr = received_client_settings.xr;
    settings_guard.auth = received_client_settings.auth;

    let client_sys = &received_client_settings.system;
    let server_sys = &mut settings_guard.system;
    let client_ws = &client_sys.websocket;
    let server_ws = &mut server_sys.websocket;
    server_ws.reconnect_attempts = client_ws.reconnect_attempts;
    server_ws.reconnect_delay = client_ws.reconnect_delay;
    server_ws.binary_chunk_size = client_ws.binary_chunk_size;
    server_ws.compression_enabled = client_ws.compression_enabled;
    server_ws.compression_threshold = client_ws.compression_threshold;
    server_ws.update_rate = client_ws.update_rate;
    server_sys.debug = client_sys.debug.clone(); // Clone DebugSettings
    server_sys.persist_settings = client_sys.persist_settings;

    settings_guard.ragflow = received_client_settings.ragflow;
    settings_guard.perplexity = received_client_settings.perplexity;
    settings_guard.openai = received_client_settings.openai;
    settings_guard.kokoro = received_client_settings.kokoro;
    // --- End Merge ---

    if let Err(e) = settings_guard.save() {
        error!("Failed to save global AppFullSettings after update from {}: {}", pubkey, e);
        return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
    }

    info!("Power user {} updated global settings via /user-settings endpoint", pubkey);
    let updated_ui_settings = convert_to_ui_settings(&*settings_guard);
    // Consider broadcasting here too if this endpoint remains active
    Ok(HttpResponse::Ok().json(updated_ui_settings))
}

// --- GET Graph Specific Settings ---

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = app_state.settings.read().await; // Reads AppFullSettings
    Ok(HttpResponse::Ok().json(&settings_guard.visualisation))
}

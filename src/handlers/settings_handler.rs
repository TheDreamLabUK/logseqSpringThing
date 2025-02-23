use crate::app_state::AppState;
use crate::models::{UISettings, UserSettings};
use actix_web::{web, Error, HttpResponse, HttpRequest};
use chrono::Utc;
use serde_json::Value;
use crate::config::feature_access::FeatureAccess;
use log::{info, error, warn, debug};

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/user-settings")
            .route(web::get().to(get_public_settings))
            .route(web::post().to(update_settings))
    ).service(
        web::resource("/user-settings/sync")
            .route(web::get().to(get_user_settings))
            .route(web::post().to(update_user_settings))
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
    // Get pubkey from header
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    
    // Check if user has permission using FeatureAccess
    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted to sync settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    let is_power_user = feature_access.is_power_user(&pubkey);

    if is_power_user {
        // Power users get settings from the global settings file
        let settings_guard = state.settings.read().await;
        let ui_settings = UISettings::from(&*settings_guard);
        debug!("Returning global settings for power user {}", pubkey);
        Ok(HttpResponse::Ok().json(ui_settings))
    } else {
        // Regular users get their personal settings or defaults
        let user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {} with default settings", pubkey);
            UserSettings::new(&pubkey, UISettings::default())
        });
        Ok(HttpResponse::Ok().json(&user_settings.settings))
    }
}

async fn update_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    // Get pubkey from header
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            debug!("No Nostr pubkey in headers, returning default settings");
            let settings_guard = state.settings.read().await;
            let ui_settings = UISettings::from(&*settings_guard);
            return Ok(HttpResponse::Ok().json(ui_settings));
        }
    };

    // Check if user has permission to sync settings
    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted to sync settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    // Parse and validate settings
    let ui_settings: UISettings = match serde_json::from_value(payload.into_inner()) {
        Ok(settings) => settings,
        Err(e) => return Ok(HttpResponse::BadRequest().body(format!("Invalid settings format: {}", e)))
    };

    // Check if user is a power user
    let is_power_user = feature_access.is_power_user(&pubkey);

    if is_power_user {
        // Power users update the global settings file
        let mut settings_guard = state.settings.write().await;
        ui_settings.merge_into_settings(&mut settings_guard);
        
        if let Err(e) = settings_guard.save() {
            error!("Failed to save global settings: {}", e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
        }
        
        info!("Power user {} updated global settings", pubkey);
        let updated_ui_settings = UISettings::from(&*settings_guard);
        Ok(HttpResponse::Ok().json(updated_ui_settings))
    } else {
        // Regular users update their personal settings file
        let mut user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {}", pubkey);
            UserSettings::new(&pubkey, UISettings::default())
        });
        user_settings.settings = ui_settings;
        user_settings.last_modified = Utc::now().timestamp();
        
        if let Err(e) = user_settings.save() {
            error!("Failed to save user settings for {}: {}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save user settings: {}", e)));
        }
        
        debug!("User {} updated their settings", pubkey);
        Ok(HttpResponse::Ok().json(&user_settings.settings))
    }
}

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
    
    info!("Power user {} updated global settings", pubkey);
    let updated_ui_settings = UISettings::from(&*settings_guard);
    Ok(HttpResponse::Ok().json(updated_ui_settings))
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings = app_state.settings.read().await;
    let ui_settings = UISettings::from(&*settings);
    Ok(HttpResponse::Ok().json(&ui_settings.visualization))
}

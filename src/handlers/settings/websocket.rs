use actix_web::{get, put, web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{debug, error};

use crate::config::Settings;
use crate::handlers::settings::common::{SettingResponse, get_setting_value, update_setting_value};

// Note: Connection keep-alive is handled by WebSocket protocol-level ping/pong frames
// automatically by the actix-web-actors framework on the server and browser WebSocket API
// on the client. No custom heartbeat implementation is needed.
#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketSettings {
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub update_rate: u32,
}

#[get("")]
async fn get_websocket_settings(settings: web::Data<Arc<RwLock<Settings>>>) -> HttpResponse {
    debug!("Getting WebSocket settings");
    let settings = settings.read().await;
    
    match serde_json::to_value(&settings.system.websocket) {
        Ok(settings_value) => {
            debug!("Successfully retrieved WebSocket settings: {:?}", settings_value);
            HttpResponse::Ok().json(SettingResponse {
                category: "websocket".to_string(),
                setting: "all".to_string(),
                value: settings_value,
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to serialize WebSocket settings: {}", e);
            HttpResponse::InternalServerError().json(SettingResponse {
                category: "websocket".to_string(),
                setting: "all".to_string(),
                value: Value::Null,
                success: false,
                error: Some(format!("Failed to serialize WebSocket settings: {}", e)),
            })
        }
    }
}

#[get("/{setting}")]
async fn get_websocket_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let setting = path.into_inner();
    let settings = settings.read().await;
    
    debug!("Getting WebSocket setting: {}", setting);
    match get_setting_value(&settings, "websocket", &setting) {
        Ok(value) => {
            debug!("Successfully retrieved WebSocket setting {}: {:?}", setting, value);
            HttpResponse::Ok().json(SettingResponse {
                category: "websocket".to_string(),
                setting: setting.clone(),
                value,
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to get WebSocket setting {}: {}", setting, e);
            HttpResponse::BadRequest().json(SettingResponse {
                category: "websocket".to_string(),
                setting: setting.clone(),
                value: Value::Null,
                success: false,
                error: Some(e),
            })
        }
    }
}

#[put("/{setting}")]
async fn update_websocket_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<Value>,
) -> HttpResponse {
    let setting = path.into_inner();
    let mut settings = settings.write().await;
    
    debug!("Updating WebSocket setting: {} = {:?}", setting, value);
    
    // Validate setting value based on type
    let validation_error = match setting.as_str() {
        "reconnectAttempts" => {
            if let Some(v) = value.as_u64() {
                if v == 0 || v > 10 {
                    Some("reconnectAttempts must be between 1 and 10")
                } else {
                    None
                }
            } else {
                Some("reconnectAttempts must be a positive integer")
            }
        },
        "reconnectDelay" => {
            if let Some(v) = value.as_u64() {
                if v < 1000 || v > 60000 {
                    Some("reconnectDelay must be between 1000 and 60000 milliseconds")
                } else {
                    None
                }
            } else {
                Some("reconnectDelay must be a positive integer")
            }
        },
        "updateRate" => {
            if let Some(v) = value.as_u64() {
                if v < 1 || v > 120 {
                    Some("updateRate must be between 1 and 120")
                } else {
                    None
                }
            } else {
                Some("updateRate must be a positive integer")
            }
        },
        _ => None
    };

    if let Some(error_msg) = validation_error {
        error!("WebSocket setting validation failed: {}", error_msg);
        return HttpResponse::BadRequest().json(SettingResponse {
            category: "websocket".to_string(),
            setting: setting.clone(),
            value: Value::Null,
            success: false,
            error: Some(error_msg.to_string()),
        });
    }
    
    match update_setting_value(&mut settings, "websocket", &setting, &value) {
        Ok(_) => {
            debug!("Successfully updated WebSocket setting {}: {:?}", setting, value);
            HttpResponse::Ok().json(SettingResponse {
                category: "websocket".to_string(),
                setting: setting.clone(),
                value: value.into_inner(),
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to update WebSocket setting {}: {}", setting, e);
            HttpResponse::BadRequest().json(SettingResponse {
                category: "websocket".to_string(),
                setting: setting.clone(),
                value: Value::Null,
                success: false,
                error: Some(e),
            })
        }
    }
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg
        .service(get_websocket_settings)
        .service(get_websocket_setting)
        .service(update_websocket_setting);
}

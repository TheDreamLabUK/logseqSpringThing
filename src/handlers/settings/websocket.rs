use actix_web::{get, put, web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::debug;

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
    let settings = settings.read().await;
    let settings_value = serde_json::to_value(&settings.system.websocket)
        .unwrap_or_default();
    
    HttpResponse::Ok().json(SettingResponse {
        category: "websocket".to_string(),
        setting: "all".to_string(),
        value: settings_value,
        success: true,
        error: None,
    })
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
        Ok(value) => HttpResponse::Ok().json(SettingResponse {
            category: "websocket".to_string(),
            setting: setting.clone(),
            value,
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::BadRequest().json(SettingResponse {
            category: "websocket".to_string(),
            setting: setting.clone(),
            value: Value::Null,
            success: false,
            error: Some(e),
        }),
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
    match update_setting_value(&mut settings, "websocket", &setting, &value) {
        Ok(_) => HttpResponse::Ok().json(SettingResponse {
            category: "websocket".to_string(),
            setting: setting.clone(),
            value: value.into_inner(),
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::BadRequest().json(SettingResponse {
            category: "websocket".to_string(),
            setting: setting.clone(),
            value: Value::Null,
            success: false,
            error: Some(e),
        }),
    }
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg
        .service(get_websocket_settings)
        .service(get_websocket_setting)
        .service(update_websocket_setting);
}

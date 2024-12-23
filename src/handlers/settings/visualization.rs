use actix_web::{get, put, web, HttpResponse};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{debug, error};

use crate::config::Settings;
use super::common::{SettingResponse, CategorySettingsResponse, CategorySettingsUpdate, get_setting_value, update_setting_value};

#[get("/{category}/{setting}")]
async fn get_visualization_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    let settings = settings.read().await;
    
    match get_setting_value(&settings, &category, &setting) {
        Ok(value) => HttpResponse::Ok().json(SettingResponse {
            category: category.clone(),
            setting: setting.clone(),
            value,
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::BadRequest().json(SettingResponse {
            category,
            setting,
            value: Value::Null,
            success: false,
            error: Some(e),
        }),
    }
}

#[put("/{category}/{setting}")]
async fn update_visualization_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    let mut settings = settings.write().await;
    
    match update_setting_value::<serde_json::Value>(&mut settings, &category, &setting, &value) {
        Ok(_) => HttpResponse::Ok().json(SettingResponse {
            category: category.clone(),
            setting: setting.clone(),
            value: value.into_inner(),
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::BadRequest().json(SettingResponse {
            category,
            setting,
            value: value.into_inner(),
            success: false,
            error: Some(e),
        }),
    }
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(get_visualization_setting)
       .service(update_visualization_setting);
}

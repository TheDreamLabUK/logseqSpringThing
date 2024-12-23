use actix_web::{get, put, web, HttpResponse};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{debug, error};

use crate::config::Settings;
use super::common::{SettingResponse, CategorySettingsResponse, CategorySettingsUpdate, get_setting_value, update_setting_value, get_category_settings, update_category_settings};

#[get("/{category}")]
async fn get_visualization_category(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let category = path.into_inner();
    let settings = settings.read().await;
    
    match get_category_settings(&settings, &category) {
        Ok(settings) => HttpResponse::Ok().json(CategorySettingsResponse {
            category: category.clone(),
            settings,
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::BadRequest().json(CategorySettingsResponse {
            category,
            settings: Value::Null,
            success: false,
            error: Some(e),
        }),
    }
}

#[put("/{category}")]
async fn update_visualization_category(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    update: web::Json<CategorySettingsUpdate>,
) -> HttpResponse {
    let category = path.into_inner();
    let mut settings = settings.write().await;
    
    match update_category_settings(&mut settings, &category, update.into_inner()) {
        Ok(updated_settings) => HttpResponse::Ok().json(CategorySettingsResponse {
            category: category.clone(),
            settings: updated_settings,
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::BadRequest().json(CategorySettingsResponse {
            category,
            settings: Value::Null,
            success: false,
            error: Some(e),
        }),
    }
}

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
    cfg.service(get_visualization_category)
       .service(update_visualization_category)
       .service(get_visualization_setting)
       .service(update_visualization_setting);
}

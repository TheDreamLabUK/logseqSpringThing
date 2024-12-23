use actix_web::{get, put, web, HttpResponse};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

use crate::config::Settings;
use super::common::{SettingResponse, CategorySettingsResponse, CategorySettingsUpdate, get_setting_value, update_setting_value, get_category_settings_value};

#[get("")]
async fn get_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    let category = "visualization".to_string();
    let settings = settings.read().await;
    
    match get_category_settings_value(&settings, &category) {
        Ok(value) => {
            let settings_map: HashMap<String, Value> = value.as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();
            
            HttpResponse::Ok().json(CategorySettingsResponse {
                category: category.clone(),
                settings: settings_map,
                success: true,
                error: None,
            })
        },
        Err(e) => HttpResponse::BadRequest().json(CategorySettingsResponse {
            category,
            settings: HashMap::new(),
            success: false,
            error: Some(e),
        }),
    }
}

#[put("")]
async fn update_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    update: web::Json<CategorySettingsUpdate>,
) -> HttpResponse {
    let category = "visualization".to_string();
    let mut settings = settings.write().await;
    let mut success = true;
    let mut error_msg = None;

    for (setting, value) in update.settings.iter() {
        if let Err(e) = update_setting_value(&mut settings, &category, setting, value) {
            success = false;
            error_msg = Some(e);
            break;
        }
    }

    HttpResponse::Ok().json(CategorySettingsResponse {
        category,
        settings: update.settings.clone(),
        success,
        error: error_msg,
    })
}

#[get("/{setting}")]
async fn get_visualization_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let setting = path.into_inner();
    let category = "visualization".to_string();
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

#[put("/{setting}")]
async fn update_visualization_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<Value>,
) -> HttpResponse {
    let setting = path.into_inner();
    let category = "visualization".to_string();
    let mut settings = settings.write().await;
    
    match update_setting_value(&mut settings, &category, &setting, &value) {
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
    cfg.service(get_visualization_settings)
       .service(update_visualization_settings)
       .service(get_visualization_setting)
       .service(update_visualization_setting);
}

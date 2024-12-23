use actix_web::{get, put, web, HttpResponse};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use log::{error, debug};

use crate::config::Settings;
use super::common::{SettingResponse, CategorySettingsResponse, CategorySettingsUpdate, get_setting_value, update_setting_value};

// List of categories that make up visualization settings
const VISUALIZATION_CATEGORIES: [&str; 7] = [
    "nodes",
    "edges",
    "bloom",
    "physics",
    "rendering",
    "labels",
    "animations"
];

#[get("")]
async fn get_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    debug!("Getting all visualization settings");
    
    let settings_guard = settings.read().await;
    
    let mut combined_settings = HashMap::new();
    
    // Convert settings to Value for easier access
    let settings_value = match serde_json::to_value(&*settings_guard) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to serialize settings: {}", e);
            return HttpResponse::InternalServerError().json(CategorySettingsResponse {
                category: "visualization".to_string(),
                settings: HashMap::new(),
                success: false,
                error: Some(format!("Failed to serialize settings: {}", e)),
            });
        }
    };

    // Combine all visualization-related categories
    for category in VISUALIZATION_CATEGORIES.iter() {
        debug!("Processing category: {}", category);
        if let Some(category_settings) = settings_value.get(category) {
            if let Some(map) = category_settings.as_object() {
                for (key, value) in map {
                    let combined_key = format!("{}_{}", category, key);
                    debug!("Adding setting: {}", combined_key);
                    combined_settings.insert(combined_key, value.clone());
                }
            } else {
                error!("Category {} settings is not an object", category);
            }
        } else {
            error!("Category {} not found in settings", category);
        }
    }

    debug!("Returning {} combined settings", combined_settings.len());
    HttpResponse::Ok().json(CategorySettingsResponse {
        category: "visualization".to_string(),
        settings: combined_settings,
        success: true,
        error: None,
    })
}

#[put("")]
async fn update_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    update: web::Json<CategorySettingsUpdate>,
) -> HttpResponse {
    debug!("Updating visualization settings");
    let mut settings_guard = settings.write().await;
    let mut success = true;
    let mut error_msg = None;

    for (key, value) in update.settings.iter() {
        // Split the key into category and setting
        let parts: Vec<&str> = key.split('_').collect();
        if parts.len() < 2 {
            error!("Invalid setting key format: {}", key);
            success = false;
            error_msg = Some(format!("Invalid setting key format: {}", key));
            break;
        }

        let category = parts[0];
        let setting = parts[1..].join("_");

        if !VISUALIZATION_CATEGORIES.contains(&category) {
            error!("Invalid category: {}", category);
            success = false;
            error_msg = Some(format!("Invalid category: {}", category));
            break;
        }

        debug!("Updating setting {}.{}", category, setting);
        if let Err(e) = update_setting_value(&mut settings_guard, category, &setting, value) {
            error!("Failed to update setting {}.{}: {}", category, setting, e);
            success = false;
            error_msg = Some(e);
            break;
        }
    }

    HttpResponse::Ok().json(CategorySettingsResponse {
        category: "visualization".to_string(),
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
    let setting_path = path.into_inner();
    debug!("Getting visualization setting: {}", setting_path);
    
    let parts: Vec<&str> = setting_path.split('_').collect();
    if parts.len() < 2 {
        error!("Invalid setting path format: {}", setting_path);
        return HttpResponse::BadRequest().json(SettingResponse {
            category: "visualization".to_string(),
            setting: setting_path.clone(),
            value: Value::Null,
            success: false,
            error: Some("Invalid setting path format".to_string()),
        });
    }

    let category = parts[0];
    let setting = parts[1..].join("_");

    if !VISUALIZATION_CATEGORIES.contains(&category) {
        error!("Invalid category: {}", category);
        return HttpResponse::BadRequest().json(SettingResponse {
            category: "visualization".to_string(),
            setting: setting_path.clone(),
            value: Value::Null,
            success: false,
            error: Some(format!("Invalid category: {}", category)),
        });
    }

    let settings_guard = settings.read().await;

    match get_setting_value(&settings_guard, category, &setting) {
        Ok(value) => {
            debug!("Successfully retrieved setting value");
            HttpResponse::Ok().json(SettingResponse {
                category: "visualization".to_string(),
                setting: setting_path,
                value,
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to get setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category: "visualization".to_string(),
                setting: setting_path,
                value: Value::Null,
                success: false,
                error: Some(e),
            })
        }
    }
}

#[put("/{setting}")]
async fn update_visualization_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<Value>,
) -> HttpResponse {
    let setting_path = path.into_inner();
    debug!("Updating visualization setting: {}", setting_path);
    
    let parts: Vec<&str> = setting_path.split('_').collect();
    if parts.len() < 2 {
        error!("Invalid setting path format: {}", setting_path);
        return HttpResponse::BadRequest().json(SettingResponse {
            category: "visualization".to_string(),
            setting: setting_path.clone(),
            value: value.into_inner(),
            success: false,
            error: Some("Invalid setting path format".to_string()),
        });
    }

    let category = parts[0];
    let setting = parts[1..].join("_");

    if !VISUALIZATION_CATEGORIES.contains(&category) {
        error!("Invalid category: {}", category);
        return HttpResponse::BadRequest().json(SettingResponse {
            category: "visualization".to_string(),
            setting: setting_path.clone(),
            value: value.into_inner(),
            success: false,
            error: Some(format!("Invalid category: {}", category)),
        });
    }

    let mut settings_guard = settings.write().await;

    match update_setting_value(&mut settings_guard, category, &setting, &value) {
        Ok(_) => {
            debug!("Successfully updated setting value");
            HttpResponse::Ok().json(SettingResponse {
                category: "visualization".to_string(),
                setting: setting_path,
                value: value.into_inner(),
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to update setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category: "visualization".to_string(),
                setting: setting_path,
                value: value.into_inner(),
                success: false,
                error: Some(e),
            })
        }
    }
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(get_visualization_settings)
       .service(update_visualization_settings)
       .service(get_visualization_setting)
       .service(update_visualization_setting);
}

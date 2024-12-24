use actix_web::{get, put, web, HttpResponse};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use log::{error, debug};

use crate::config::Settings;
use super::common::{SettingResponse, CategorySettingsResponse, CategorySettingsUpdate, get_setting_value, update_setting_value};

// List of categories that make up visualization settings
const VISUALIZATION_CATEGORIES: [&str; 10] = [
    "animations",
    "ar",
    "audio",
    "bloom",
    "edges",
    "hologram",
    "labels",
    "nodes",
    "physics",
    "rendering"
];

#[get("/visualization")]
async fn get_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    debug!("Getting all visualization settings");
    
    let settings_guard = settings.read().await;
    let mut combined_settings = HashMap::new();
    
    // Get visualization settings directly
    let vis_settings = &settings_guard.visualization;
    
    // Process each category
    for category in VISUALIZATION_CATEGORIES.iter() {
        debug!("Processing category: {}", category);
        let category_value = match *category {
            "animations" => serde_json::to_value(&vis_settings.animations),
            "ar" => serde_json::to_value(&vis_settings.ar),
            "audio" => serde_json::to_value(&vis_settings.audio),
            "bloom" => serde_json::to_value(&vis_settings.bloom),
            "edges" => serde_json::to_value(&vis_settings.edges),
            "hologram" => serde_json::to_value(&vis_settings.hologram),
            "labels" => serde_json::to_value(&vis_settings.labels),
            "nodes" => serde_json::to_value(&vis_settings.nodes),
            "physics" => serde_json::to_value(&vis_settings.physics),
            "rendering" => serde_json::to_value(&vis_settings.rendering),
            _ => continue,
        };

        if let Ok(value) = category_value {
            if let Some(map) = value.as_object() {
                for (key, value) in map {
                    let combined_key = format!("{}_{}", category, key);
                    debug!("Adding setting: {}", combined_key);
                    combined_settings.insert(combined_key, value.clone());
                }
            }
        } else {
            error!("Failed to serialize {} settings", category);
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

#[put("/visualization")]
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

#[get("/visualization/{category}/{setting}")]
async fn get_visualization_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    debug!("Getting visualization setting: {}.{}", category, setting);

    if !VISUALIZATION_CATEGORIES.contains(&category.as_str()) {
        error!("Invalid category: {}", category);
        return HttpResponse::BadRequest().json(SettingResponse {
            category: "visualization".to_string(),
            setting: format!("{}.{}", category, setting),
            value: Value::Null,
            success: false,
            error: Some(format!("Invalid category: {}", category)),
        });
    }

    let settings_guard = settings.read().await;

    match get_setting_value(&settings_guard, &category, &setting) {
        Ok(value) => {
            debug!("Successfully retrieved setting value");
            HttpResponse::Ok().json(SettingResponse {
                category: "visualization".to_string(),
                setting: format!("{}.{}", category, setting),
                value,
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to get setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category: "visualization".to_string(),
                setting: format!("{}.{}", category, setting),
                value: Value::Null,
                success: false,
                error: Some(e),
            })
        }
    }
}

#[put("/visualization/{category}/{setting}")]
async fn update_visualization_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    debug!("Updating visualization setting: {}.{}", category, setting);

    if !VISUALIZATION_CATEGORIES.contains(&category.as_str()) {
        error!("Invalid category: {}", category);
        return HttpResponse::BadRequest().json(SettingResponse {
            category: "visualization".to_string(),
            setting: format!("{}.{}", category, setting),
            value: value.into_inner(),
            success: false,
            error: Some(format!("Invalid category: {}", category)),
        });
    }

    let mut settings_guard = settings.write().await;

    match update_setting_value(&mut settings_guard, &category, &setting, &value) {
        Ok(_) => {
            debug!("Successfully updated setting value");
            HttpResponse::Ok().json(SettingResponse {
                category: "visualization".to_string(),
                setting: format!("{}.{}", category, setting),
                value: value.into_inner(),
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to update setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category: "visualization".to_string(),
                setting: format!("{}.{}", category, setting),
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

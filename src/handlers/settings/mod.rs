use actix_web::{web, get, put, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use log::debug;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::collections::HashMap;
use crate::config::Settings;
use crate::utils::case_conversion::to_snake_case;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingResponse {
    pub category: String,
    pub setting: String,
    pub value: Value,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CategorySettingsResponse {
    pub category: String,
    pub settings: HashMap<String, Value>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// Helper function to get all settings
fn get_all_settings(settings: &Settings) -> Value {
    debug!("Getting all settings");
    serde_json::to_value(settings.clone()).unwrap_or_default()
}

// Helper function to get setting value from settings object
fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Getting setting value for {}.{}", category, setting);
    
    // Convert kebab-case to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Convert settings to Value for easier access
    let settings_value = serde_json::to_value(settings.clone())
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    // Get category object
    let category_value = settings_value.get(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    // Get setting value
    let setting_value = category_value.get(&setting_snake)
        .ok_or_else(|| format!("Setting '{}' not found in category '{}'", setting, category))?;
    
    Ok(setting_value.clone())
}

// Helper function to update setting value in settings object
fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Updating setting value for {}.{}", category, setting);
    
    // Convert kebab-case to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Convert settings to Value for manipulation
    let mut settings_value = serde_json::to_value(settings.clone())
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    // Get category object
    let category_value = settings_value.get_mut(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    // Update setting value
    if let Some(obj) = category_value.as_object_mut() {
        obj.insert(setting_snake.clone(), value.clone());
        
        // Convert back to Settings
        *settings = serde_json::from_value(settings_value)
            .map_err(|e| format!("Failed to deserialize settings: {}", e))?;
        Ok(())
    } else {
        Err(format!("Category '{}' is not an object", category))
    }
}

// Helper function to get all settings for a category
fn get_category_settings(settings: &Settings, category: &str) -> Result<Value, String> {
    debug!("Getting settings for category: {}", category);
    
    // Convert kebab-case to snake_case
    let category_snake = to_snake_case(category);
    
    // Convert settings to Value
    let settings_value = serde_json::to_value(settings.clone())
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    // Get category object
    let category_value = settings_value.get(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    Ok(category_value.clone())
}

#[get("")]
async fn get_all_settings_handler(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    let settings_guard = settings.read().await;
    let settings_value = get_all_settings(&settings_guard);
    HttpResponse::Ok().json(settings_value)
}

#[get("/{category}/{setting}")]
async fn get_setting_handler(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    let settings_guard = settings.read().await;
    
    match get_setting_value(&settings_guard, &category, &setting) {
        Ok(value) => HttpResponse::Ok().json(SettingResponse {
            category,
            setting,
            value,
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::NotFound().json(SettingResponse {
            category,
            setting,
            value: Value::Null,
            success: false,
            error: Some(e),
        }),
    }
}

#[put("/{category}/{setting}")]
async fn update_setting_handler(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    let mut settings_guard = settings.write().await;
    
    match update_setting_value(&mut settings_guard, &category, &setting, &value) {
        Ok(_) => HttpResponse::Ok().json(SettingResponse {
            category,
            setting,
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

#[get("/{category}")]
async fn get_category_settings_handler(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let category = path.into_inner();
    let settings_guard = settings.read().await;
    
    match get_category_settings(&settings_guard, &category) {
        Ok(settings) => HttpResponse::Ok().json(CategorySettingsResponse {
            category,
            settings: settings.as_object()
                .map(|obj| obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<HashMap<String, Value>>())
                .unwrap_or_default(),
            success: true,
            error: None,
        }),
        Err(e) => HttpResponse::NotFound().json(CategorySettingsResponse {
            category,
            settings: HashMap::new(),
            success: false,
            error: Some(e),
        }),
    }
}

pub mod common;
pub mod websocket;
pub mod visualization;

// Register all settings handlers
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg
        .service(get_all_settings_handler)
        .service(get_category_settings_handler)
        .service(get_setting_handler)
        .service(update_setting_handler)
        .service(
            web::scope("/websocket")
                .configure(websocket::config)
        );
}

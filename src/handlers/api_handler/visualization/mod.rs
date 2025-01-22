use crate::config::Settings;
use actix_web::{web, HttpResponse, error::ErrorInternalServerError, Error, Result};
use log::{error, info, debug};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::utils::case_conversion::to_snake_case;
use crate::AppState;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SettingResponse {
    pub category: String,
    pub setting: String,
    pub value: Value,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CategorySettingsResponse {
    pub category: String,
    pub settings: HashMap<String, Value>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SettingValue {
    pub value: Value,
}

fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Attempting to get setting value for category: {}, setting: {}", category, setting);
    
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!("Converted category '{}' to snake_case: '{}'", category, category_snake);
    debug!("Converted setting '{}' to snake_case: '{}'", setting, setting_snake);
    
    let settings_value = match serde_json::to_value(&settings) {
        Ok(v) => {
            debug!("Successfully serialized settings to JSON");
            v
        },
        Err(e) => {
            error!("Failed to serialize settings to JSON: {}", e);
            return Err(format!("Failed to serialize settings: {}", e));
        }
    };
    
    debug!("Settings JSON structure: {}", settings_value);
    
    let category_value = match settings_value.get(&category_snake) {
        Some(v) => {
            debug!("Found category '{}' in settings", category_snake);
            v
        },
        None => {
            error!("Category '{}' not found in settings", category_snake);
            return Err(format!("Category '{}' not found", category));
        }
    };
    
    let setting_value = match category_value.get(&setting_snake) {
        Some(v) => {
            debug!("Found setting '{}' in category '{}'", setting_snake, category_snake);
            v
        },
        None => {
            error!("Setting '{}' not found in category '{}'", setting_snake, category_snake);
            return Err(format!("Setting '{}' not found in category '{}'", setting, category));
        }
    };
    
    debug!("Found setting value: {:?}", setting_value);
    Ok(setting_value.clone())
}

fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Attempting to update setting value for category: {}, setting: {}", category, setting);
    
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!("Converted category '{}' to snake_case: '{}'", category, category_snake);
    debug!("Converted setting '{}' to snake_case: '{}'", setting, setting_snake);
    
    let mut settings_value = match serde_json::to_value(&*settings) {
        Ok(v) => {
            debug!("Successfully serialized settings to JSON");
            v
        },
        Err(e) => {
            error!("Failed to serialize settings to JSON: {}", e);
            return Err(format!("Failed to serialize settings: {}", e));
        }
    };
    
    debug!("Settings JSON structure: {}", settings_value);
    
    let category_value = match settings_value.get_mut(&category_snake) {
        Some(v) => {
            debug!("Found category '{}' in settings", category_snake);
            v
        },
        None => {
            error!("Category '{}' not found in settings", category_snake);
            return Err(format!("Category '{}' not found", category));
        }
    };
    
    if let Some(obj) = category_value.as_object_mut() {
        obj.insert(setting_snake.to_string(), value.clone());
        debug!("Updated setting value successfully");
        
        match serde_json::from_value(settings_value) {
            Ok(new_settings) => {
                debug!("Successfully converted updated JSON back to Settings");
                *settings = new_settings;
                Ok(())
            },
            Err(e) => {
                error!("Failed to convert JSON back to Settings: {}", e);
                Err(format!("Failed to deserialize settings: {}", e))
            }
        }
    } else {
        error!("Category '{}' is not an object", category_snake);
        Err(format!("Category '{}' is not an object", category))
    }
}

fn get_category_settings_value(settings: &Settings, category: &str) -> Result<Value, String> {
    debug!("Getting settings for category: {}", category);
    let value = match category {
        "nodes" => serde_json::to_value(&settings.nodes),
        "edges" => serde_json::to_value(&settings.edges),
        "physics" => serde_json::to_value(&settings.physics),
        "rendering" => serde_json::to_value(&settings.rendering),
        "animations" => serde_json::to_value(&settings.animations),
        "labels" => serde_json::to_value(&settings.labels),
        "bloom" => serde_json::to_value(&settings.bloom),
        "hologram" => serde_json::to_value(&settings.hologram),
        "network" => serde_json::to_value(&settings.network),
        "websocket" => serde_json::to_value(&settings.websocket),
        "security" => serde_json::to_value(&settings.security),
        "debug" => serde_json::to_value(&settings.debug),
        _ => return Err(format!("Invalid category: {}", category)),
    }.map_err(|e| format!("Failed to serialize {} settings: {}", category, e))?;
    
    debug!("Successfully retrieved settings for category: {}", category);
    Ok(value)
}

pub async fn get_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!("Getting setting for category: {}, setting: {}", category, setting);
    
    let settings_guard = match settings.read().await {
        guard => {
            debug!("Successfully acquired settings read lock");
            guard
        }
    };

    match get_setting_value(&*settings_guard, &category, &setting) {
        Ok(value) => {
            debug!("Successfully retrieved setting value: {:?}", value);
            HttpResponse::Ok().json(SettingResponse {
                category,
                setting,
                value,
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to get setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some(e),
            })
        }
    }
}

pub async fn update_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!("Updating setting for category: {}, setting: {}", category, setting);
    
    let mut settings_guard = match settings.write().await {
        guard => {
            debug!("Successfully acquired settings write lock");
            guard
        }
    };

    match update_setting_value(&mut *settings_guard, &category, &setting, &value) {
        Ok(_) => {
            if let Err(e) = save_settings_to_file(&*settings_guard) {
                error!("Failed to save settings to file: {}", e);
                return HttpResponse::InternalServerError().json(SettingResponse {
                    category,
                    setting,
                    value: value.into_inner(),
                    success: false,
                    error: Some("Failed to persist settings".to_string()),
                });
            }
            HttpResponse::Ok().json(SettingResponse {
                category,
                setting,
                value: value.into_inner(),
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to update setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category,
                setting,
                value: value.into_inner(),
                success: false,
                error: Some(e),
            })
        }
    }
}

pub async fn get_category_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let settings_read = settings.read().await;
    let debug_enabled = settings_read.debug.enabled;
    let log_json = debug_enabled && settings_read.debug.log_full_json;
    
    let category = path.into_inner();
    match get_category_settings_value(&settings_read, &category) {
        Ok(value) => {
            if log_json {
                debug!("Category '{}' settings: {}", category, serde_json::to_string_pretty(&value).unwrap_or_default());
            }
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
        Err(e) => {
            error!("Failed to get category settings for '{}': {}", category, e);
            HttpResponse::NotFound().json(CategorySettingsResponse {
                category: category.clone(),
                settings: HashMap::new(),
                success: false,
                error: Some(e),
            })
        }
    }
}

pub async fn get_visualization_settings(
    app_state: web::Data<AppState>,
    category: web::Path<String>,
) -> Result<HttpResponse, actix_web::Error> {
    debug!("Getting settings for category: {}", category);
    
    if category.as_str() == "client_debug" {
        debug!("Checking UI container status for debugging");
    }
    
    let settings = app_state.settings.read().await;
    Ok(HttpResponse::Ok().json(&*settings))
}

pub async fn get_settings_category(
    category: web::Path<String>,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let settings = app_state.settings.read().await;
    
    let value = match category.as_str() {
        "nodes" => serde_json::to_value(&settings.nodes),
        "edges" => serde_json::to_value(&settings.edges),
        "physics" => serde_json::to_value(&settings.physics),
        "rendering" => serde_json::to_value(&settings.rendering),
        "animations" => serde_json::to_value(&settings.animations),
        "labels" => serde_json::to_value(&settings.labels),
        "bloom" => serde_json::to_value(&settings.bloom),
        "hologram" => serde_json::to_value(&settings.hologram),
        "network" => serde_json::to_value(&settings.network),
        "websocket" => serde_json::to_value(&settings.websocket),
        "security" => serde_json::to_value(&settings.security),
        "debug" => serde_json::to_value(&settings.debug),
        _ => return Ok(HttpResponse::NotFound().finish()),
    }.map_err(ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(value))
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    debug!("Attempting to save settings to file");
    
    let settings_path = std::env::var("SETTINGS_FILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/app/settings.toml"));
    
    info!("Attempting to save settings to: {:?}", settings_path);
    
    if let Some(parent) = settings_path.parent() {
        match fs::create_dir_all(parent) {
            Ok(_) => debug!("Created parent directories: {:?}", parent),
            Err(e) => {
                error!("Failed to create parent directories: {}", e);
                return Err(e);
            }
        }
    }
    
    if settings_path.exists() {
        match fs::metadata(&settings_path) {
            Ok(metadata) => {
                if metadata.permissions().readonly() {
                    error!("Settings file is read-only: {:?}", settings_path);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Settings file is read-only"
                    ));
                }
            }
            Err(e) => {
                error!("Failed to check settings file permissions: {}", e);
                return Err(e);
            }
        }
    }
    
    let toml_string = match toml::to_string_pretty(&settings) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to serialize settings to TOML: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };
    
    match fs::write(&settings_path, toml_string) {
        Ok(_) => {
            info!("Settings saved successfully to: {:?}", settings_path);
            Ok(())
        }
        Err(e) => {
            error!("Failed to write settings file: {}", e);
            Err(e)
        }
    }
}

// Configure routes using snake_case
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/visualization")
            .route("/settings/{category}/{setting}", web::get().to(get_setting))
            .route("/settings/{category}/{setting}", web::put().to(update_setting))
            .route("/settings/{category}", web::get().to(get_category_settings))
            .route("/get_settings/{category}", web::get().to(get_visualization_settings))
    );
}

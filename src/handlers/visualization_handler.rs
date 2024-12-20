use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::fs;
use std::path::PathBuf;
use toml;
use log::{error, info, debug};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
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

// Request/Response structures for individual settings
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingValue {
    pub value: Value,
}

// Helper function to get setting value from settings object
fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Attempting to get setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to appropriate cases
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!("Converted category '{}' to snake_case: '{}'", category, category_snake);
    debug!("Converted setting '{}' to snake_case: '{}'", setting, setting_snake);
    
    // Convert settings to Value for easier access
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
    
    // Get category object using snake_case for internal lookup
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
    
    // Get setting value using snake_case for internal lookup
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

// Helper function to update setting value in settings object
fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Attempting to update setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case for internal lookup
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!("Converted category '{}' to snake_case: '{}'", category, category_snake);
    debug!("Converted setting '{}' to snake_case: '{}'", setting, setting_snake);
    
    // Convert settings to Value for manipulation, using a reference to avoid moving
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
    
    // Get category object
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
    
    // Update setting value
    if let Some(obj) = category_value.as_object_mut() {
        obj.insert(setting_snake.to_string(), value.clone());
        debug!("Updated setting value successfully");
        
        // Convert back to Settings
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

// Helper function to get all settings for a category
fn get_category_settings_value(settings: &Settings, category: &str) -> Result<Value, String> {
    debug!("Getting settings for category: {}", category);
    let value = match category {
        "nodes" => serde_json::to_value(&settings.nodes)
            .map_err(|e| format!("Failed to serialize node settings: {}", e))?,
        "edges" => serde_json::to_value(&settings.edges)
            .map_err(|e| format!("Failed to serialize edge settings: {}", e))?,
        "rendering" => serde_json::to_value(&settings.rendering)
            .map_err(|e| format!("Failed to serialize rendering settings: {}", e))?,
        "labels" => serde_json::to_value(&settings.labels)
            .map_err(|e| format!("Failed to serialize labels settings: {}", e))?,
        "bloom" => serde_json::to_value(&settings.bloom)
            .map_err(|e| format!("Failed to serialize bloom settings: {}", e))?,
        "animations" => serde_json::to_value(&settings.animations)
            .map_err(|e| format!("Failed to serialize animations settings: {}", e))?,
        "ar" => serde_json::to_value(&settings.ar)
            .map_err(|e| format!("Failed to serialize ar settings: {}", e))?,
        "audio" => serde_json::to_value(&settings.audio)
            .map_err(|e| format!("Failed to serialize audio settings: {}", e))?,
        "physics" => serde_json::to_value(&settings.physics)
            .map_err(|e| format!("Failed to serialize physics settings: {}", e))?,
        "clientDebug" => serde_json::to_value(&settings.client_debug)
            .map_err(|e| format!("Failed to serialize client debug settings: {}", e))?,
        "serverDebug" => serde_json::to_value(&settings.server_debug)
            .map_err(|e| format!("Failed to serialize server debug settings: {}", e))?,
        "security" => serde_json::to_value(&settings.security)
            .map_err(|e| format!("Failed to serialize security settings: {}", e))?,
        "websocket" => serde_json::to_value(&settings.websocket)
            .map_err(|e| format!("Failed to serialize websocket settings: {}", e))?,
        "network" => serde_json::to_value(&settings.network)
            .map_err(|e| format!("Failed to serialize network settings: {}", e))?,
        "default" => serde_json::to_value(&settings.default)
            .map_err(|e| format!("Failed to serialize default settings: {}", e))?,
        "github" => serde_json::to_value(&settings.github)
            .map_err(|e| format!("Failed to serialize github settings: {}", e))?,
        _ => return Err(format!("Invalid category: {}", category)),
    };
    debug!("Successfully retrieved settings for category: {}", category);
    Ok(value)
}

// GET /api/visualization/settings/{category}/{setting}
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

// PUT /api/visualization/settings/{category}/{setting}
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

// GET /api/visualization/settings/{category}
pub async fn get_category_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let category = path.into_inner();
    debug!("Getting all settings for category: {}", category);

    let settings_guard = match settings.read().await {
        guard => {
            debug!("Successfully acquired settings read lock");
            guard
        }
    };

    match get_category_settings_value(&*settings_guard, &category) {
        Ok(settings_value) => {
            debug!("Successfully retrieved category settings: {:?}", settings_value);
            let settings_hash: HashMap<String, Value> = settings_value
                .as_object()
                .map(|map| map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();
            
            HttpResponse::Ok().json(CategorySettingsResponse {
                category,
                settings: settings_hash,
                success: true,
                error: None,
            })
        },
        Err(e) => {
            error!("Failed to get category settings: {}", e);
            HttpResponse::NotFound().json(CategorySettingsResponse {
                category,
                settings: HashMap::new(),
                success: false,
                error: Some(e),
            })
        }
    }
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/visualization")
            .route("/settings/{category}/{setting}", web::get().to(get_setting))
            .route("/settings/{category}/{setting}", web::put().to(update_setting))
            .route("/settings/{category}", web::get().to(get_category_settings))
        );
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    debug!("Attempting to save settings to file");
    
    // Use absolute path from environment or default to /app/settings.toml
    let settings_path = std::env::var("SETTINGS_FILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/app/settings.toml"));
    
    info!("Attempting to save settings to: {:?}", settings_path);
    
    // Ensure parent directory exists and is writable
    if let Some(parent) = settings_path.parent() {
        match fs::create_dir_all(parent) {
            Ok(_) => debug!("Created parent directories: {:?}", parent),
            Err(e) => {
                error!("Failed to create parent directories: {}", e);
                return Err(e);
            }
        }
    }
    
    // Check if file exists and is writable
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
    
    // Convert settings to TOML
    let toml_string = match toml::to_string_pretty(&settings) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to serialize settings to TOML: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };
    
    // Write to settings.toml
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

use crate::config::Settings;
use actix_web::{get, put, web, HttpResponse};
use log::{error, info, debug};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::utils::case_conversion::{to_snake_case, to_camel_case};

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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CategorySettingsUpdate {
    pub settings: HashMap<String, Value>,
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
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!("Converted category '{}' to snake_case: '{}'", category, category_snake);
    debug!("Converted setting '{}' to snake_case: '{}'", setting, setting_snake);
    
    // Convert settings to Value for easier access
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;

    // Determine the root category (visualization, xr, or system)
    let (root_category, sub_category) = if category_snake.starts_with("visualization_") {
        ("visualization", &category_snake[13..])
    } else if category_snake.starts_with("xr_") {
        ("xr", &category_snake[3..])
    } else if category_snake.starts_with("system_") {
        ("system", &category_snake[7..])
    } else {
        return Err(format!("Invalid category format: {}", category));
    };

    // Get the root category object
    let root_value = match settings_value.get(root_category) {
        Some(v) => v,
        None => {
            error!("Root category '{}' not found", root_category);
            return Err(format!("Root category '{}' not found", root_category));
        }
    };

    // Get the sub-category object
    let sub_value = match root_value.get(sub_category) {
        Some(v) => v,
        None => {
            error!("Sub-category '{}' not found in {}", sub_category, root_category);
            return Err(format!("Sub-category '{}' not found in {}", sub_category, root_category));
        }
    };

    // Get the setting value
    match sub_value.get(&setting_snake) {
        Some(v) => {
            debug!("Found setting '{}' in {}.{}", setting_snake, root_category, sub_category);
            Ok(v.clone())
        },
        None => {
            error!("Setting '{}' not found in {}.{}", setting_snake, root_category, sub_category);
            Err(format!("Setting '{}' not found in {}.{}", setting, root_category, sub_category))
        }
    }
}

// Helper function to update setting value in settings object
fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Attempting to update setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Determine the root category and sub-category
    let (root_category, sub_category) = if category_snake.starts_with("visualization_") {
        ("visualization", &category_snake[13..])
    } else if category_snake.starts_with("xr_") {
        ("xr", &category_snake[3..])
    } else if category_snake.starts_with("system_") {
        ("system", &category_snake[7..])
    } else {
        return Err(format!("Invalid category format: {}", category));
    };

    // Convert the value to the appropriate type and update the settings
    match serde_json::from_value(value.clone()) {
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
        // Get the current value to determine its type
        let current_value = obj.get(&setting_snake);
        
        // Convert value based on the current value's type
        let converted_value = match current_value {
            Some(current) if current.is_boolean() => {
                // For boolean settings, handle various input formats
                if value.is_boolean() {
                    value.clone()
                } else if value.is_string() {
                    Value::Bool(value.as_str().unwrap_or("false").to_lowercase() == "true")
                } else if value.is_number() {
                    Value::Bool(value.as_i64().unwrap_or(0) != 0)
                } else {
                    value.clone()
                }
            },
            Some(current) if current.is_number() => {
                // For numeric settings, handle string inputs
                if value.is_number() {
                    value.clone()
                } else if value.is_string() {
                    if let Ok(num) = value.as_str().unwrap_or("0").trim().parse::<f64>() {
                        Value::Number(serde_json::Number::from_f64(num).unwrap_or_else(|| serde_json::Number::from(0)))
                    } else {
                        value.clone()
                    }
                } else if value.is_boolean() {
                    Value::Number(serde_json::Number::from(if value.as_bool().unwrap_or(false) { 1 } else { 0 }))
                } else {
                    value.clone()
                }
            },
            _ => value.clone()
        };

        obj.insert(setting_snake.to_string(), converted_value);
        debug!("Updated setting value successfully");
        
        // Convert back to Settings
        match serde_json::from_value(settings_value) {
            Ok(new_settings) => {
                debug!("Successfully converted updated JSON back to Settings");
                *settings = new_settings;
                Ok(())
            } else {
                Err(format!("Invalid settings structure for {}.{}", root_category, sub_category))
            }
        },
        Err(e) => Err(format!("Invalid value for setting: {}", e))
    }
}

// GET /api/visualization/settings/{category}
#[get("/api/visualization/settings/{category}")]
pub async fn get_category_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>
) -> HttpResponse {
    let category = path.into_inner();
    debug!("Getting settings for category: {}", category);
    // Convert incoming category to snake_case for internal lookup
    let category_snake = to_snake_case(&category);
    debug!("Looking up settings for category: {} (snake_case: {})", category, category_snake);

    let value = match category_snake.as_str() {
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
        "client_debug" => serde_json::to_value(&settings.client_debug)
            .map_err(|e| format!("Failed to serialize client debug settings: {}", e))?,
        "server_debug" => serde_json::to_value(&settings.server_debug)
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
            HttpResponse::Ok().json(value)
        },
        Err(e) => {
            error!("Failed to get setting value: {}", e);
            HttpResponse::NotFound().json(Value::Null)
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
    debug!("Raw value from client: {:?}", value);
    info!("Updating setting for category: {}, setting: {}", category, setting);
    
    let mut settings_guard = match settings.write().await {
        guard => {
            debug!("Successfully acquired settings write lock");
            guard
        }
    };

    // Extract the actual value from the client's JSON structure
    let actual_value = if let Some(obj) = value.as_object() {
        if let Some(val) = obj.get("value") {
            val
        } else {
            &value
        }
    } else {
        &value
    };

    match update_setting_value(&mut *settings_guard, &category, &setting, actual_value) {
        Ok(_) => {
            // Get the actual updated value for the response
            match get_setting_value(&*settings_guard, &category, &setting) {
                Ok(updated_value) => {
                    if let Err(e) = save_settings_to_file(&*settings_guard) {
                        error!("Failed to save settings to file: {}", e);
                        return HttpResponse::InternalServerError().json(Value::Null);
                    }
                    // Return just the updated value as the client expects
                    HttpResponse::Ok().json(updated_value)
                },
                Err(e) => {
                    error!("Failed to get updated setting value: {}", e);
                    HttpResponse::InternalServerError().json(Value::Null)
                }
            }
        },
        Err(e) => {
            error!("Failed to update setting value: {}", e);
            HttpResponse::NotFound().json(Value::Null)
        }
    }
}

// GET /api/visualization/settings/{category}
pub async fn get_category_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let settings_read = settings.read().await;
    let debug_enabled = settings_read.server_debug.enabled;
    let log_json = debug_enabled && settings_read.server_debug.log_full_json;
    
    let category = path.into_inner();
    match get_category_settings_value(&settings_read, &category) {
        Ok(value) => {
            if log_json {
                debug!("Category '{}' settings: {}", category, serde_json::to_string_pretty(&value).unwrap_or_default());
            }
            // The client expects the settings directly, not wrapped in a response object
            let settings_map: HashMap<String, Value> = value.as_object()
                .map(|m| m.iter().map(|(k, v)| {
                    // Convert snake_case keys to camelCase for client
                    (to_camel_case(k), v.clone())
                }).collect())
                .unwrap_or_default();
            
            HttpResponse::Ok().json(settings_map)
        },
        Err(e) => {
            error!("Failed to get category settings for '{}': {}", category, e);
            // Return empty object for 404s as client expects
            HttpResponse::NotFound().json(HashMap::<String, Value>::new())
        }
    }
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(get_category_settings)
       .service(update_category_settings)
       .service(
           web::resource("/api/visualization/settings/{category}/{setting}")
               .route(web::get().to(get_setting))
               .route(web::put().to(update_setting))
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

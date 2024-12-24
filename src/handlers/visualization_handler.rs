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

    // Get the category value directly from settings
    let value = match category_snake.as_str() {
        "animations" => serde_json::to_value(&settings.visualization.animations)?,
        "ar" => serde_json::to_value(&settings.visualization.ar)?,
        "audio" => serde_json::to_value(&settings.visualization.audio)?,
        "bloom" => serde_json::to_value(&settings.visualization.bloom)?,
        "edges" => serde_json::to_value(&settings.visualization.edges)?,
        "hologram" => serde_json::to_value(&settings.visualization.hologram)?,
        "labels" => serde_json::to_value(&settings.visualization.labels)?,
        "nodes" => serde_json::to_value(&settings.visualization.nodes)?,
        "physics" => serde_json::to_value(&settings.visualization.physics)?,
        "rendering" => serde_json::to_value(&settings.visualization.rendering)?,
        "network" => serde_json::to_value(&settings.system.network)?,
        "websocket" => serde_json::to_value(&settings.system.websocket)?,
        "security" => serde_json::to_value(&settings.system.security)?,
        "client_debug" => serde_json::to_value(&settings.system.debug)?,
        "server_debug" => serde_json::to_value(&settings.system.debug)?,
        _ => return Err(format!("Invalid category: {}", category)),
    };

    // Get the setting value directly from the category value
    match value.get(&setting_snake) {
        Some(v) => {
            debug!("Found setting '{}' in {}", setting_snake, category);
            Ok(v.clone())
        },
        None => {
            error!("Setting '{}' not found in {}", setting_snake, category);
            Err(format!("Setting '{}' not found in {}", setting, category))
        }
    }
}

// Helper function to update setting value in settings object
fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Attempting to update setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Convert settings to Value for modification
    let mut settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    // Get the current value from the appropriate category
    let current_value = match category_snake.as_str() {
        "animations" => serde_json::to_value(&settings.visualization.animations)?,
        "ar" => serde_json::to_value(&settings.visualization.ar)?,
        "audio" => serde_json::to_value(&settings.visualization.audio)?,
        "bloom" => serde_json::to_value(&settings.visualization.bloom)?,
        "edges" => serde_json::to_value(&settings.visualization.edges)?,
        "hologram" => serde_json::to_value(&settings.visualization.hologram)?,
        "labels" => serde_json::to_value(&settings.visualization.labels)?,
        "nodes" => serde_json::to_value(&settings.visualization.nodes)?,
        "physics" => serde_json::to_value(&settings.visualization.physics)?,
        "rendering" => serde_json::to_value(&settings.visualization.rendering)?,
        "network" => serde_json::to_value(&settings.system.network)?,
        "websocket" => serde_json::to_value(&settings.system.websocket)?,
        "security" => serde_json::to_value(&settings.system.security)?,
        "client_debug" => serde_json::to_value(&settings.system.debug)?,
        "server_debug" => serde_json::to_value(&settings.system.debug)?,
        _ => return Err(format!("Invalid category: {}", category)),
    }.get(&setting_snake).ok_or_else(|| format!("Setting '{}' not found in {}", setting, category))?.clone();

    // Convert value based on the current value's type
    let converted_value = if current_value.is_boolean() {
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
    } else if current_value.is_number() {
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
    } else {
        value.clone()
    };

    // Update the setting in the appropriate category
    match category_snake.as_str() {
        "animations" => {
            let mut animations = settings.visualization.animations.clone();
            let mut value_map = serde_json::to_value(&animations)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.animations = serde_json::from_value(value_map)?;
            }
        },
        "ar" => {
            let mut ar = settings.visualization.ar.clone();
            let mut value_map = serde_json::to_value(&ar)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.ar = serde_json::from_value(value_map)?;
            }
        },
        "audio" => {
            let mut audio = settings.visualization.audio.clone();
            let mut value_map = serde_json::to_value(&audio)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.audio = serde_json::from_value(value_map)?;
            }
        },
        "bloom" => {
            let mut bloom = settings.visualization.bloom.clone();
            let mut value_map = serde_json::to_value(&bloom)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.bloom = serde_json::from_value(value_map)?;
            }
        },
        "edges" => {
            let mut edges = settings.visualization.edges.clone();
            let mut value_map = serde_json::to_value(&edges)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.edges = serde_json::from_value(value_map)?;
            }
        },
        "hologram" => {
            let mut hologram = settings.visualization.hologram.clone();
            let mut value_map = serde_json::to_value(&hologram)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.hologram = serde_json::from_value(value_map)?;
            }
        },
        "labels" => {
            let mut labels = settings.visualization.labels.clone();
            let mut value_map = serde_json::to_value(&labels)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.labels = serde_json::from_value(value_map)?;
            }
        },
        "nodes" => {
            let mut nodes = settings.visualization.nodes.clone();
            let mut value_map = serde_json::to_value(&nodes)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.nodes = serde_json::from_value(value_map)?;
            }
        },
        "physics" => {
            let mut physics = settings.visualization.physics.clone();
            let mut value_map = serde_json::to_value(&physics)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.physics = serde_json::from_value(value_map)?;
            }
        },
        "rendering" => {
            let mut rendering = settings.visualization.rendering.clone();
            let mut value_map = serde_json::to_value(&rendering)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.rendering = serde_json::from_value(value_map)?;
            }
        },
        "network" => {
            let mut network = settings.system.network.clone();
            let mut value_map = serde_json::to_value(&network)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.network = serde_json::from_value(value_map)?;
            }
        },
        "websocket" => {
            let mut websocket = settings.system.websocket.clone();
            let mut value_map = serde_json::to_value(&websocket)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.websocket = serde_json::from_value(value_map)?;
            }
        },
        "security" => {
            let mut security = settings.system.security.clone();
            let mut value_map = serde_json::to_value(&security)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.security = serde_json::from_value(value_map)?;
            }
        },
        "client_debug" | "server_debug" => {
            let mut debug = settings.system.debug.clone();
            let mut value_map = serde_json::to_value(&debug)?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.debug = serde_json::from_value(value_map)?;
            }
        },
        _ => return Err(format!("Invalid category: {}", category)),
    };
    
    debug!("Updated setting value successfully");
    Ok(())
}

// Helper function to get category settings value 
fn get_category_settings_value(settings: &Settings, category: &str) -> Result<Value, String> {
    let category_snake = to_snake_case(category);
    debug!("Getting settings for category: {} (snake_case: {})", category, category_snake);

    match category_snake.as_str() {
        "animations" => serde_json::to_value(&settings.visualization.animations)
            .map_err(|e| format!("Failed to serialize animations settings: {}", e)),
        "ar" => serde_json::to_value(&settings.visualization.ar)
            .map_err(|e| format!("Failed to serialize ar settings: {}", e)),
        "audio" => serde_json::to_value(&settings.visualization.audio)
            .map_err(|e| format!("Failed to serialize audio settings: {}", e)),
        "bloom" => serde_json::to_value(&settings.visualization.bloom)
            .map_err(|e| format!("Failed to serialize bloom settings: {}", e)),
        "edges" => serde_json::to_value(&settings.visualization.edges)
            .map_err(|e| format!("Failed to serialize edges settings: {}", e)),
        "hologram" => serde_json::to_value(&settings.visualization.hologram)
            .map_err(|e| format!("Failed to serialize hologram settings: {}", e)),
        "labels" => serde_json::to_value(&settings.visualization.labels)
            .map_err(|e| format!("Failed to serialize labels settings: {}", e)),
        "nodes" => serde_json::to_value(&settings.visualization.nodes)
            .map_err(|e| format!("Failed to serialize nodes settings: {}", e)),
        "physics" => serde_json::to_value(&settings.visualization.physics)
            .map_err(|e| format!("Failed to serialize physics settings: {}", e)),
        "rendering" => serde_json::to_value(&settings.visualization.rendering)
            .map_err(|e| format!("Failed to serialize rendering settings: {}", e)),
        "network" => serde_json::to_value(&settings.system.network)
            .map_err(|e| format!("Failed to serialize network settings: {}", e)),
        "websocket" => serde_json::to_value(&settings.system.websocket)
            .map_err(|e| format!("Failed to serialize websocket settings: {}", e)),
        "security" => serde_json::to_value(&settings.system.security)
            .map_err(|e| format!("Failed to serialize security settings: {}", e)),
        "client_debug" | "server_debug" => serde_json::to_value(&settings.system.debug)
            .map_err(|e| format!("Failed to serialize debug settings: {}", e)),
        _ => Err(format!("Invalid category: {}", category)),
    }
}

// GET /api/visualization/settings/{category}
#[get("/api/visualization/settings/{category}")]
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

// GET /api/visualization/settings/{category}/{setting}
pub async fn get_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!("Getting setting for category: {}, setting: {}", category, setting);
    
    let settings_guard = settings.read().await;

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
    
    let mut settings_guard = settings.write().await;

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

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(get_category_settings)
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

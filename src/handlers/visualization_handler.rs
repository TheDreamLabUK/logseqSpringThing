use crate::config::Settings;
use actix_web::{get, put, web, HttpResponse, Responder};
use log::{error, info, debug};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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

// Helper function to get setting value from settings object
fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Getting setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Get the category value directly from settings
    let category_value = match category_snake.as_str() {
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
    }?;

    // If no setting is specified, return the entire category
    if setting.is_empty() {
        return Ok(category_value);
    }

    // Get the setting value
    match category_value.get(&setting_snake) {
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
    debug!("Updating setting {}.{}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Get current value to determine type
    let current_value = get_setting_value(settings, &category_snake, &setting_snake)?;
    
    // Convert value based on the current value's type
    let converted_value = if current_value.is_boolean() {
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

    // Update the appropriate category
    match category_snake.as_str() {
        "animations" => {
            let animations = settings.visualization.animations.clone();
            let mut value_map = serde_json::to_value(&animations)
                .map_err(|e| format!("Failed to serialize animations: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.animations = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update animations: {}", e))?;
            }
        },
        "ar" => {
            let ar = settings.visualization.ar.clone();
            let mut value_map = serde_json::to_value(&ar)
                .map_err(|e| format!("Failed to serialize ar settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.ar = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update ar settings: {}", e))?;
            }
        },
        "audio" => {
            let audio = settings.visualization.audio.clone();
            let mut value_map = serde_json::to_value(&audio)
                .map_err(|e| format!("Failed to serialize audio settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.audio = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update audio settings: {}", e))?;
            }
        },
        "bloom" => {
            let bloom = settings.visualization.bloom.clone();
            let mut value_map = serde_json::to_value(&bloom)
                .map_err(|e| format!("Failed to serialize bloom settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.bloom = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update bloom settings: {}", e))?;
            }
        },
        "edges" => {
            let edges = settings.visualization.edges.clone();
            let mut value_map = serde_json::to_value(&edges)
                .map_err(|e| format!("Failed to serialize edges settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.edges = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update edges settings: {}", e))?;
            }
        },
        "hologram" => {
            let hologram = settings.visualization.hologram.clone();
            let mut value_map = serde_json::to_value(&hologram)
                .map_err(|e| format!("Failed to serialize hologram settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.hologram = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update hologram settings: {}", e))?;
            }
        },
        "labels" => {
            let labels = settings.visualization.labels.clone();
            let mut value_map = serde_json::to_value(&labels)
                .map_err(|e| format!("Failed to serialize labels settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.labels = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update labels settings: {}", e))?;
            }
        },
        "nodes" => {
            let nodes = settings.visualization.nodes.clone();
            let mut value_map = serde_json::to_value(&nodes)
                .map_err(|e| format!("Failed to serialize nodes settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.nodes = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update nodes settings: {}", e))?;
            }
        },
        "physics" => {
            let physics = settings.visualization.physics.clone();
            let mut value_map = serde_json::to_value(&physics)
                .map_err(|e| format!("Failed to serialize physics settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.physics = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update physics settings: {}", e))?;
            }
        },
        "rendering" => {
            let rendering = settings.visualization.rendering.clone();
            let mut value_map = serde_json::to_value(&rendering)
                .map_err(|e| format!("Failed to serialize rendering settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.rendering = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update rendering settings: {}", e))?;
            }
        },
        "network" => {
            let network = settings.system.network.clone();
            let mut value_map = serde_json::to_value(&network)
                .map_err(|e| format!("Failed to serialize network settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.network = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update network settings: {}", e))?;
            }
        },
        "websocket" => {
            let websocket = settings.system.websocket.clone();
            let mut value_map = serde_json::to_value(&websocket)
                .map_err(|e| format!("Failed to serialize websocket settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.websocket = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update websocket settings: {}", e))?;
            }
        },
        "security" => {
            let security = settings.system.security.clone();
            let mut value_map = serde_json::to_value(&security)
                .map_err(|e| format!("Failed to serialize security settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.security = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update security settings: {}", e))?;
            }
        },
        "client_debug" | "server_debug" => {
            let debug = settings.system.debug.clone();
            let mut value_map = serde_json::to_value(&debug)
                .map_err(|e| format!("Failed to serialize debug settings: {}", e))?;
            if let Some(obj) = value_map.as_object_mut() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.debug = serde_json::from_value(value_map)
                    .map_err(|e| format!("Failed to update debug settings: {}", e))?;
            }
        },
        _ => return Err(format!("Invalid category: {}", category)),
    };
    
    debug!("Successfully updated setting {}.{}", category_snake, setting_snake);
    Ok(())
}

// Flattened API endpoints for settings
#[get("/settings")]
pub async fn get_all_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    let settings_read = settings.read().await;
    let all_settings = serde_json::to_value(&*settings_read).unwrap_or_default();
    
    // Convert all keys to camelCase for client
    let flattened = flatten_and_convert_case(&all_settings);
    HttpResponse::Ok().json(flattened)
}

#[get("/settings/{path:.*}")]
pub async fn get_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let settings_read = settings.read().await;
    let path_str = path.into_inner();
    
    // Convert camelCase path to snake_case for internal lookup
    let internal_path = to_snake_case(&path_str);
    match get_setting_by_path(&settings_read, &internal_path) {
        Ok(value) => {
            let converted = convert_value_case_to_camel(&value);
            HttpResponse::Ok().json(converted)
        },
        Err(e) => HttpResponse::NotFound().json(json!({
            "error": format!("Setting not found: {}", e)
        }))
    }
}

#[put("/api/settings/{category}/{setting}")]
async fn update_visualization_setting(
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> impl Responder {
    let (category, setting) = path.into_inner();
    let path_str = format!("{}.{}", category, setting);
    
    // Get write lock for settings update
    let mut settings_guard = settings.write().await;
    match update_setting_by_path(&mut *settings_guard, &path_str, &value) {
        Ok(_) => {
            // Convert settings to Value without moving it
            match serde_json::to_value(&*settings_guard) {
                Ok(settings_value) => HttpResponse::Ok().json(settings_value),
                Err(e) => {
                    error!("Failed to serialize settings: {}", e);
                    HttpResponse::InternalServerError().json(json!({
                        "error": format!("Failed to serialize settings: {}", e)
                    }))
                }
            }
        },
        Err(e) => {
            error!("Failed to update setting: {}", e);
            HttpResponse::BadRequest().json(json!({
                "error": format!("Failed to update setting: {}", e)
            }))
        }
    }
}

// Helper function to flatten settings and convert to camelCase
fn flatten_and_convert_case(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut flattened = serde_json::Map::new();
            flatten_object_recursive(map, "", &mut flattened);
            Value::Object(flattened)
        },
        _ => value.clone(),
    }
}

// Recursively flatten nested objects with dot notation
fn flatten_object_recursive(
    obj: &serde_json::Map<String, Value>,
    prefix: &str,
    output: &mut serde_json::Map<String, Value>
) {
    for (key, value) in obj {
        let new_key = if prefix.is_empty() {
            to_camel_case(key)
        } else {
            format!("{}.{}", prefix, to_camel_case(key))
        };

        match value {
            Value::Object(nested) => {
                flatten_object_recursive(nested, &new_key, output);
            },
            _ => {
                output.insert(new_key, value.clone());
            }
        }
    }
}

// Helper function to get a setting by its dot-notation path
fn get_setting_by_path(settings: &Settings, path: &str) -> Result<Value, String> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = serde_json::to_value(settings).unwrap();
    
    for part in parts {
        match current {
            Value::Object(map) => {
                current = map.get(part)
                    .ok_or_else(|| format!("Setting not found: {}", part))?
                    .clone();
            },
            _ => return Err(format!("Invalid path: {}", path)),
        }
    }
    
    Ok(current)
}

// Helper function to update a setting by its dot-notation path
fn update_setting_by_path(settings: &mut Settings, path: &str, value: &Value) -> Result<(), String> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = serde_json::to_value(&*settings).unwrap();
    
    // Navigate to the parent object
    let parent_path = &parts[..parts.len()-1];
    let mut parent = &mut current;
    for part in parent_path {
        parent = parent.get_mut(part)
            .ok_or_else(|| format!("Invalid path: {}", path))?;
    }
    
    // Update the value
    if let Value::Object(map) = parent {
        let last_key = parts.last().unwrap();
        map.insert(last_key.to_string(), value.clone());
        
        // Deserialize back into Settings
        *settings = serde_json::from_value(current)
            .map_err(|e| format!("Failed to deserialize settings: {}", e))?;
        Ok(())
    } else {
        Err(format!("Invalid path: {}", path))
    }
}

// Helper function to convert all object keys in a Value to camelCase
fn convert_value_case_to_camel(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (key, val) in map {
                let camel_key = to_camel_case(key);
                new_map.insert(camel_key, convert_value_case_to_camel(val));
            }
            Value::Object(new_map)
        },
        Value::Array(arr) => {
            Value::Array(arr.iter().map(convert_value_case_to_camel).collect())
        },
        _ => value.clone(),
    }
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(get_all_settings)
       .service(get_setting)
       .service(update_visualization_setting);
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

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use log::{error, debug};
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CategorySettingsUpdate {
    pub settings: HashMap<String, Value>,
}

pub fn get_category_settings_value(settings: &Settings, category: &str) -> Result<Value, String> {
    debug!("Getting settings for category: {}", category);
    let category_snake = to_snake_case(category);
    
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

pub fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Getting setting value for {}.{}", category, setting);
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    let category_value = get_category_settings_value(settings, &category_snake)?;
    
    match category_value.get(&setting_snake) {
        Some(v) => {
            debug!("Found value for {}.{}", category_snake, setting_snake);
            Ok(v.clone())
        },
        None => {
            error!("Setting '{}' not found in {}", setting_snake, category);
            Err(format!("Setting '{}' not found in {}", setting, category))
        }
    }
}

pub fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Updating setting {}.{}", category, setting);
    
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
            let value_map = serde_json::to_value(&animations)
                .map_err(|e| format!("Failed to serialize animations: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.animations = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update animations: {}", e))?;
            }
        },
        "ar" => {
            let ar = settings.visualization.ar.clone();
            let value_map = serde_json::to_value(&ar)
                .map_err(|e| format!("Failed to serialize ar settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.ar = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update ar settings: {}", e))?;
            }
        },
        "audio" => {
            let audio = settings.visualization.audio.clone();
            let value_map = serde_json::to_value(&audio)
                .map_err(|e| format!("Failed to serialize audio settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.audio = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update audio settings: {}", e))?;
            }
        },
        "bloom" => {
            let bloom = settings.visualization.bloom.clone();
            let value_map = serde_json::to_value(&bloom)
                .map_err(|e| format!("Failed to serialize bloom settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.bloom = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update bloom settings: {}", e))?;
            }
        },
        "edges" => {
            let edges = settings.visualization.edges.clone();
            let value_map = serde_json::to_value(&edges)
                .map_err(|e| format!("Failed to serialize edges settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.edges = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update edges settings: {}", e))?;
            }
        },
        "hologram" => {
            let hologram = settings.visualization.hologram.clone();
            let value_map = serde_json::to_value(&hologram)
                .map_err(|e| format!("Failed to serialize hologram settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.hologram = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update hologram settings: {}", e))?;
            }
        },
        "labels" => {
            let labels = settings.visualization.labels.clone();
            let value_map = serde_json::to_value(&labels)
                .map_err(|e| format!("Failed to serialize labels settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.labels = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update labels settings: {}", e))?;
            }
        },
        "nodes" => {
            let nodes = settings.visualization.nodes.clone();
            let value_map = serde_json::to_value(&nodes)
                .map_err(|e| format!("Failed to serialize nodes settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.nodes = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update nodes settings: {}", e))?;
            }
        },
        "physics" => {
            let physics = settings.visualization.physics.clone();
            let value_map = serde_json::to_value(&physics)
                .map_err(|e| format!("Failed to serialize physics settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.physics = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update physics settings: {}", e))?;
            }
        },
        "rendering" => {
            let rendering = settings.visualization.rendering.clone();
            let value_map = serde_json::to_value(&rendering)
                .map_err(|e| format!("Failed to serialize rendering settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.visualization.rendering = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update rendering settings: {}", e))?;
            }
        },
        "network" => {
            let network = settings.system.network.clone();
            let value_map = serde_json::to_value(&network)
                .map_err(|e| format!("Failed to serialize network settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.network = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update network settings: {}", e))?;
            }
        },
        "websocket" => {
            let websocket = settings.system.websocket.clone();
            let value_map = serde_json::to_value(&websocket)
                .map_err(|e| format!("Failed to serialize websocket settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.websocket = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update websocket settings: {}", e))?;
            }
        },
        "security" => {
            let security = settings.system.security.clone();
            let value_map = serde_json::to_value(&security)
                .map_err(|e| format!("Failed to serialize security settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.security = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update security settings: {}", e))?;
            }
        },
        "client_debug" | "server_debug" => {
            let debug = settings.system.debug.clone();
            let value_map = serde_json::to_value(&debug)
                .map_err(|e| format!("Failed to serialize debug settings: {}", e))?;
            if let Some(mut obj) = value_map.as_object().cloned() {
                obj.insert(setting_snake.clone(), converted_value);
                settings.system.debug = serde_json::from_value(Value::Object(obj))
                    .map_err(|e| format!("Failed to update debug settings: {}", e))?;
            }
        },
        _ => return Err(format!("Invalid category: {}", category)),
    };
    
    debug!("Successfully updated setting {}.{}", category_snake, setting_snake);
    Ok(())
}

pub fn set_field_value<T>(obj: &mut T, field: &str, value: Value) -> Result<(), String> 
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let map = serde_json::to_value(&*obj)
        .map_err(|e| format!("Failed to serialize object: {}", e))?
        .as_object()
        .ok_or_else(|| "Failed to convert object to map".to_string())?
        .clone();

    let mut updated_map = map.clone();
    updated_map.insert(field.to_string(), value);

    *obj = serde_json::from_value(Value::Object(updated_map))
        .map_err(|e| format!("Failed to deserialize updated object: {}", e))?;

    Ok(())
}

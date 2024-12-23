use serde::{Deserialize, Serialize};
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CategorySettingsUpdate {
    pub settings: HashMap<String, Value>,
}

pub fn get_category_settings_value(settings: &Settings, category: &str) -> Result<Value, String> {
    let category_snake = to_snake_case(category);
    
    // Convert settings to Value for easier access
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;

    // Special handling for nested settings
    match category_snake.as_str() {
        "hologram" => {
            if let Some(hologram) = settings_value.get("hologram") {
                return Ok(hologram.clone());
            }
        },
        "websocket" => {
            if let Some(websocket) = settings_value.get("websocket") {
                return Ok(websocket.clone());
            }
        },
        _ => {}
    }

    // Regular category lookup
    settings_value.get(&category_snake)
        .cloned()
        .ok_or_else(|| format!("Category '{}' not found", category))
}

pub fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    match category_snake.as_str() {
        "hologram" => {
            if let Some(hologram) = settings_value.get("hologram") {
                if let Some(setting_value) = hologram.get(&setting_snake) {
                    return Ok(setting_value.clone());
                }
            }
        },
        "websocket" => {
            if let Some(websocket) = settings_value.get("websocket") {
                if let Some(setting_value) = websocket.get(&setting_snake) {
                    return Ok(setting_value.clone());
                }
            }
        },
        _ => {}
    }
    
    let category_value = settings_value.get(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    category_value.get(&setting_snake)
        .cloned()
        .ok_or_else(|| format!("Setting '{}' not found in category '{}'", setting, category))
}

pub fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    match category_snake.as_str() {
        "websocket" => {
            match setting_snake.as_str() {
                "heartbeat_interval" => {
                    if let Some(v) = value.as_u64() {
                        settings.websocket.heartbeat_interval = v;
                        return Ok(());
                    }
                },
                "heartbeat_timeout" => {
                    if let Some(v) = value.as_u64() {
                        settings.websocket.heartbeat_timeout = v;
                        return Ok(());
                    }
                },
                "reconnect_attempts" => {
                    if let Some(v) = value.as_u64() {
                        settings.websocket.reconnect_attempts = v as u32;
                        return Ok(());
                    }
                },
                "reconnect_delay" => {
                    if let Some(v) = value.as_u64() {
                        settings.websocket.reconnect_delay = v;
                        return Ok(());
                    }
                },
                "update_rate" => {
                    if let Some(v) = value.as_u64() {
                        settings.websocket.update_rate = v as u32;
                        return Ok(());
                    }
                },
                _ => {}
            }
        },
        _ => {}
    }
    
    match serde_json::from_value(value.clone()) {
        Ok(v) => {
            match category_snake.as_str() {
                "hologram" => {
                    let hologram = &mut settings.hologram;
                    if let Err(e) = set_field_value(hologram, &setting_snake, v) {
                        return Err(format!("Failed to set hologram setting: {}", e));
                    }
                },
                _ => {
                    if let Err(e) = set_field_value(settings, &category_snake, v) {
                        return Err(format!("Failed to set setting: {}", e));
                    }
                }
            }
            Ok(())
        },
        Err(e) => Err(format!("Invalid value for setting: {}", e))
    }
}

pub fn set_field_value<T>(obj: &mut T, field: &str, value: Value) -> Result<(), String> 
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let mut map = serde_json::to_value(&*obj)
        .map_err(|e| format!("Failed to serialize object: {}", e))?
        .as_object()
        .ok_or_else(|| "Failed to convert object to map".to_string())?
        .clone();

    map.insert(field.to_string(), value);

    let value = Value::Object(map);
    *obj = serde_json::from_value(value)
        .map_err(|e| format!("Failed to deserialize updated object: {}", e))?;

    Ok(())
}

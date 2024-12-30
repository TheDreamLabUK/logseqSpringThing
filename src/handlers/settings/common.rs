use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use log::debug;
use crate::config::Settings;
use convert_case::{Case, Casing};

// Convert kebab-case from API to snake_case for internal use
pub fn to_snake_case(s: &str) -> String {
    s.to_case(Case::Snake)
}

// Convert snake_case to kebab-case for API responses
pub fn to_kebab_case(s: &str) -> String {
    s.to_case(Case::Kebab)
}

// Convert nested object keys from snake_case to kebab-case
pub fn convert_to_kebab_case(value: Value) -> Value {
    match value {
        Value::Object(obj) => {
            let mut new_obj = serde_json::Map::new();
            for (k, v) in obj {
                let kebab_key = to_kebab_case(&k);
                new_obj.insert(kebab_key, convert_to_kebab_case(v));
            }
            Value::Object(new_obj)
        }
        Value::Array(arr) => Value::Array(
            arr.into_iter()
               .map(convert_to_kebab_case)
               .collect()
        ),
        v => v,
    }
}

// Convert nested object keys from kebab-case to snake_case
pub fn convert_to_snake_case(value: Value) -> Value {
    match value {
        Value::Object(obj) => {
            let mut new_obj = serde_json::Map::new();
            for (k, v) in obj {
                let snake_key = to_snake_case(&k);
                new_obj.insert(snake_key, convert_to_snake_case(v));
            }
            Value::Object(new_obj)
        }
        Value::Array(arr) => Value::Array(
            arr.into_iter()
               .map(convert_to_snake_case)
               .collect()
        ),
        v => v,
    }
}

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
    
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    let category_value = settings_value.get(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    // Convert response to kebab-case
    Ok(convert_to_kebab_case(category_value.clone()))
}

pub fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Getting setting value for {}.{}", category, setting);
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    let category_value = get_category_settings_value(settings, &category_snake)?;
    
    let setting_value = category_value.get(&setting_snake)
        .ok_or_else(|| format!("Setting '{}' not found", setting))?;
    
    // Convert response to kebab-case
    Ok(convert_to_kebab_case(setting_value.clone()))
}

pub fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Updating setting {}.{}", category, setting);
    
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Convert incoming value's keys to snake_case
    let snake_value = convert_to_snake_case(value.clone());
    
    let mut settings_value = serde_json::to_value(&*settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    if let Some(obj) = settings_value.get_mut(&category_snake)
        .and_then(|v| v.as_object_mut())
    {
        obj.insert(setting_snake.clone(), snake_value);
        *settings = serde_json::from_value(settings_value)
            .map_err(|e| format!("Failed to deserialize settings: {}", e))?;
        Ok(())
    } else {
        Err(format!("Category '{}' not found or invalid", category))
    }
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

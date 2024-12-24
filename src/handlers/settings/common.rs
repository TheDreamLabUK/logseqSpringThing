use serde::{Deserialize, Serialize};
use serde_json::{Value, Map};
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
    
    // Convert settings to Value for easier access
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| {
            error!("Failed to serialize settings: {}", e);
            format!("Failed to serialize settings: {}", e)
        })?;

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

    // Get the root category object
    let root_value = match settings_value.get(root_category) {
        Some(v) => v,
        None => {
            error!("Root category '{}' not found", root_category);
            return Err(format!("Root category '{}' not found", root_category));
        }
    };

    // Get the sub-category object
    match root_value.get(sub_category) {
        Some(v) => {
            debug!("Found settings for {}.{}", root_category, sub_category);
            Ok(v.clone())
        },
        None => {
            error!("Sub-category '{}' not found in {}", sub_category, root_category);
            Err(format!("Sub-category '{}' not found in {}", sub_category, root_category))
        }
    }
}

pub fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Getting setting value for {}.{}", category, setting);
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| {
            error!("Failed to serialize settings: {}", e);
            format!("Failed to serialize settings: {}", e)
        })?;
    
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
            debug!("Found value for {}.{}.{}", root_category, sub_category, setting_snake);
            Ok(v.clone())
        },
        None => {
            error!("Setting '{}' not found in {}.{}", setting_snake, root_category, sub_category);
            Err(format!("Setting '{}' not found in {}.{}", setting, root_category, sub_category))
        }
    }
}

pub fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Updating setting {}.{}", category, setting);
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

    // Convert settings to Value for modification
    let mut settings_value = serde_json::to_value(&*settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;

    // Get mutable access to the root category
    let root_value = settings_value.get_mut(root_category)
        .ok_or_else(|| format!("Root category '{}' not found", root_category))?;

    // Get mutable access to the sub-category
    let sub_value = root_value.get_mut(sub_category)
        .ok_or_else(|| format!("Sub-category '{}' not found in {}", sub_category, root_category))?;

    // Update the setting value
    if let Some(obj) = sub_value.as_object_mut() {
        obj.insert(setting_snake.clone(), value.clone());
        
        // Convert back to Settings
        *settings = serde_json::from_value(settings_value)
            .map_err(|e| format!("Failed to update settings: {}", e))?;
        
        debug!("Successfully updated setting {}.{}.{}", root_category, sub_category, setting_snake);
        Ok(())
    } else {
        Err(format!("Invalid settings structure for {}.{}", root_category, sub_category))
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

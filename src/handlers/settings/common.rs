use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use crate::config::Settings;
use log::{debug, error};
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

// Helper function to get all settings in a category
pub fn get_category_settings(settings: &Settings, category: &str) -> Result<HashMap<String, Value>, String> {
    debug!("Attempting to get all settings for category: {}", category);
    
    // Convert kebab-case URL parameter to snake_case
    let category_snake = to_snake_case(category);
    
    // Convert settings to Value for easier access
    let settings_value = serde_json::to_value(&settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    // Get category object
    let category_value = settings_value.get(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    // Convert category value to HashMap
    if let Some(obj) = category_value.as_object() {
        Ok(obj.clone())
    } else {
        Err(format!("Category '{}' is not an object", category))
    }
}

// Helper function to update all settings in a category
pub fn update_category_settings(
    settings: &mut Settings,
    category: &str,
    update: CategorySettingsUpdate
) -> Result<HashMap<String, Value>, String> {
    debug!("Attempting to update all settings for category: {}", category);
    
    // Convert kebab-case URL parameter to snake_case
    let category_snake = to_snake_case(category);
    
    // Convert settings to Value for manipulation
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    let mut settings_map = settings_value.as_object()
        .ok_or("Settings is not an object")?
        .clone();
    
    // Get category object
    let category_value = settings_map.get_mut(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    if let Some(obj) = category_value.as_object_mut() {
        // Update each setting in the category
        for (key, value) in update.settings {
            let key_snake = to_snake_case(&key);
            if obj.contains_key(&key_snake) {
                obj.insert(key_snake, value);
            } else {
                return Err(format!("Setting '{}' not found in category '{}'", key, category));
            }
        }
        
        // Update the settings object
        *settings = serde_json::from_value(serde_json::Value::Object(settings_map))?;
        
        // Return the updated category settings
        Ok(obj.clone())
    } else {
        Err(format!("Category '{}' is not an object", category))
    }
}

// Helper function to get setting value from settings object
pub fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Attempting to get setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Convert settings to Value for easier access
    let settings_value = serde_json::to_value(&settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    // Get category object
    let category_value = settings_value.get(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    // Get setting value
    category_value.get(&setting_snake)
        .ok_or_else(|| format!("Setting '{}' not found in category '{}'", setting, category))
        .map(|v| v.clone())
}

// Helper function to update setting value in settings object
pub fn update_setting_value<T: serde::de::DeserializeOwned>(
    settings: &mut Settings,
    category: &str,
    setting: &str,
    value: &Value
) -> Result<(), String> {
    debug!("Attempting to update setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    
    // Deserialize the value
    let typed_value: T = serde_json::from_value(value.clone())
        .map_err(|e| format!("Invalid value for setting: {}", e))?;
    
    // Update the setting using reflection
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    let mut settings_map = settings_value.as_object()
        .ok_or("Settings is not an object")?
        .clone();
    
    let category_value = settings_map.get_mut(&category_snake)
        .ok_or_else(|| format!("Category '{}' not found", category))?;
    
    if let Some(obj) = category_value.as_object_mut() {
        obj.insert(setting_snake, serde_json::to_value(typed_value)?);
        *settings = serde_json::from_value(serde_json::Value::Object(settings_map))?;
        Ok(())
    } else {
        Err(format!("Category '{}' is not an object", category))
    }
}

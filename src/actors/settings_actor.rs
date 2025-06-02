//! Settings Actor to replace Arc<RwLock<AppFullSettings>>

use actix::prelude::*;
use serde_json::Value;
use log::{debug, info};

use crate::actors::messages::*;
use crate::config::AppFullSettings;

pub struct SettingsActor {
    settings: AppFullSettings,
}

impl SettingsActor {
    pub fn new(settings: AppFullSettings) -> Self {
        Self { settings }
    }

    pub fn get_settings(&self) -> &AppFullSettings {
        &self.settings
    }

    pub fn update_settings(&mut self, new_settings: AppFullSettings) {
        self.settings = new_settings;
        debug!("Settings updated");
    }

    pub fn get_setting_by_path(&self, path: &str) -> Result<Value, String> {
        // Navigate the settings object using the path
        let path_parts: Vec<&str> = path.split('.').collect();
        
        // Convert settings to JSON value for easier navigation
        let settings_value = serde_json::to_value(&self.settings)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;
        
        let mut current = &settings_value;
        
        for part in path_parts {
            match current.get(part) {
                Some(value) => current = value,
                None => return Err(format!("Setting path '{}' not found", path)),
            }
        }
        
        Ok(current.clone())
    }

    pub fn set_setting_by_path(&mut self, path: &str, value: Value) -> Result<(), String> {
        // Convert current settings to mutable JSON value
        let mut settings_value = serde_json::to_value(&self.settings)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;
        
        let path_parts: Vec<&str> = path.split('.').collect();
        
        // Navigate to the parent object
        let mut current_val_mut_ref = &mut settings_value;

        for part_key_str_ref in &path_parts[..path_parts.len().saturating_sub(1)] {
            // Ensure current_val_mut_ref points to an object.
            if !current_val_mut_ref.is_object() {
                if current_val_mut_ref.is_null() {
                    // If it's null, we can replace it with an object.
                    *current_val_mut_ref = Value::Object(serde_json::Map::new());
                } else {
                    // Otherwise, it's an existing non-object value in the path, which is an error.
                    let type_str = match current_val_mut_ref {
                        Value::Null => "null", // Should have been caught by is_null
                        Value::Bool(_) => "boolean",
                        Value::Number(_) => "number",
                        Value::String(_) => "string",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object", // Should not happen here due to !is_object()
                    };
                    return Err(format!(
                        "Path component '{}' is a non-object type ({}) and cannot be traversed.",
                        *part_key_str_ref, type_str
                    ));
                }
            }

            // Now, current_val_mut_ref is definitely a mutable reference to an Object Value.
            // We get its underlying &mut Map. Unwrap is safe due to the check above.
            let current_map_mut = current_val_mut_ref.as_object_mut().unwrap();

            // Get or insert the next part. `entry` returns an `Entry`.
            // `or_insert_with` ensures the value is an object if inserted.
            current_val_mut_ref = current_map_mut
                .entry((*part_key_str_ref).to_string())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            
            // Ensure this new current_val_mut_ref is also an object for the next iteration.
            // This handles cases where an existing path component was not an object.
            if !current_val_mut_ref.is_object() {
                 return Err(format!(
                    "Existing path component '{}' was expected to be an object, but it's not.",
                    *part_key_str_ref
                ));
            }
        }

        // Set the final value
        if let Some(final_part_key_str) = path_parts.last() {
            if let Value::Object(map) = current_val_mut_ref {
                map.insert((*final_part_key_str).to_string(), value);
            } else {
                // This should not happen if the loop maintained the object invariant and path_parts is not empty.
                // If path_parts was empty (e.g. path=""), current_val_mut_ref would be &mut settings_value.
                // If settings_value itself is not an object, this is an error.
                if path_parts.is_empty() || (path_parts.len() == 1 && path_parts[0].is_empty()) {
                     return Err("Cannot set value: root settings is not an object.".to_string());
                }
                return Err(format!(
                    "Cannot set value: final path component before '{}' is not an object.",
                    final_part_key_str
                ));
            }
        } else {
            // path_parts is empty, which implies the original path string was problematic
            // or resulted in no parts. This case should ideally be handled by path validation earlier.
            return Err("Path is empty, cannot set value.".to_string());
        }
        
        // Convert back to AppFullSettings
        self.settings = serde_json::from_value(settings_value)
            .map_err(|e| format!("Failed to deserialize updated settings: {}", e))?;
        
        debug!("Setting '{}' updated", path);
        Ok(())
    }
}

impl Actor for SettingsActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("SettingsActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("SettingsActor stopped");
    }
}

impl Handler<GetSettings> for SettingsActor {
    type Result = Result<AppFullSettings, String>;

    fn handle(&mut self, _msg: GetSettings, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.settings.clone())
    }
}

impl Handler<UpdateSettings> for SettingsActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.update_settings(msg.settings);
        Ok(())
    }
}

impl Handler<GetSettingByPath> for SettingsActor {
    type Result = Result<Value, String>;

    fn handle(&mut self, msg: GetSettingByPath, _ctx: &mut Self::Context) -> Self::Result {
        self.get_setting_by_path(&msg.path)
    }
}

impl Handler<SetSettingByPath> for SettingsActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSettingByPath, _ctx: &mut Self::Context) -> Self::Result {
        self.set_setting_by_path(&msg.path, msg.value)
    }
}
//! Settings Actor to replace Arc<RwLock<AppFullSettings>>

use actix::prelude::*;
use serde_json::Value;
use log::{debug, info, warn};

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
        let mut current = &mut settings_value;
        
        for part in &path_parts[..path_parts.len() - 1] {
            match current.get_mut(part) {
                Some(obj) if obj.is_object() => current = obj,
                Some(_) => return Err(format!("Path '{}' is not an object", part)),
                None => {
                    // Create the path if it doesn't exist
                    current[part] = Value::Object(serde_json::Map::new());
                    current = current.get_mut(part).unwrap();
                }
            }
        }
        
        // Set the final value
        let final_part = path_parts[path_parts.len() - 1];
        current[final_part] = value;
        
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
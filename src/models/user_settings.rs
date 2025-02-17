use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use log::{info, error};

use crate::models::UISettings;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    pub pubkey: String,
    pub settings: UISettings,
    pub last_modified: i64,
}

impl UserSettings {
    pub fn new(pubkey: &str, settings: UISettings) -> Self {
        Self {
            pubkey: pubkey.to_string(),
            settings,
            last_modified: chrono::Utc::now().timestamp(),
        }
    }

    pub fn load(pubkey: &str) -> Option<Self> {
        let path = Self::get_settings_path(pubkey);
        match fs::read_to_string(&path) {
            Ok(content) => {
                match serde_yaml::from_str(&content) {
                    Ok(settings) => {
                        info!("Loaded settings for user {}", pubkey);
                        Some(settings)
                    }
                    Err(e) => {
                        error!("Failed to parse settings for user {}: {}", pubkey, e);
                        None
                    }
                }
            }
            Err(_) => None,
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::get_settings_path(&self.pubkey);
        
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return Err(format!("Failed to create settings directory: {}", e));
            }
        }

        // Save settings
        match serde_yaml::to_string(&self) {
            Ok(yaml) => {
                if let Err(e) = fs::write(&path, yaml) {
                    Err(format!("Failed to write settings file: {}", e))
                } else {
                    info!("Saved settings for user {}", self.pubkey);
                    Ok(())
                }
            }
            Err(e) => Err(format!("Failed to serialize settings: {}", e)),
        }
    }

    fn get_settings_path(pubkey: &str) -> PathBuf {
        PathBuf::from("/app/user_settings").join(format!("{}.yaml", pubkey))
    }
}
use super::common::{SettingResult, SettingsError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSocketSettings {
    pub update_rate: u32,
    pub max_connections: u32,
    pub heartbeat_interval: u32,
}

impl WebSocketSettings {
    pub fn validate(&self) -> SettingResult<()> {
        if self.update_rate < 1 || self.update_rate > 120 {
            return Err(SettingsError::InvalidValue(
                "Update rate must be between 1 and 120".to_string()
            ));
        }

        if self.max_connections < 1 || self.max_connections > 1000 {
            return Err(SettingsError::InvalidValue(
                "Max connections must be between 1 and 1000".to_string()
            ));
        }

        if self.heartbeat_interval < 1000 || self.heartbeat_interval > 60000 {
            return Err(SettingsError::InvalidValue(
                "Heartbeat interval must be between 1000 and 60000 ms".to_string()
            ));
        }

        Ok(())
    }
}

pub async fn update_websocket_settings(settings: WebSocketSettings) -> SettingResult<()> {
    settings.validate()?;
    
    log::debug!("Updating WebSocket settings: {:?}", settings);
    
    // Update settings in persistent storage
    save_settings_to_file().await?;
    
    Ok(())
} 
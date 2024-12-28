use super::common::{SettingResult, SettingsError};

pub fn validate_visualization_setting(name: &str, value: &serde_json::Value) -> SettingResult<()> {
    match name {
        "update_rate" => {
            if let Some(rate) = value.as_u64() {
                if rate >= 1 && rate <= 120 {
                    Ok(())
                } else {
                    Err(SettingsError::InvalidValue(
                        "Update rate must be between 1 and 120".to_string()
                    ))
                }
            } else {
                Err(SettingsError::InvalidValue("Invalid update rate type".to_string()))
            }
        },
        "color_scheme" => {
            if value.is_string() {
                Ok(())
            } else {
                Err(SettingsError::InvalidValue("Color scheme must be a string".to_string()))
            }
        },
        // Add validation for other visualization settings
        _ => Ok(()) // Allow unknown settings for forward compatibility
    }
}

pub async fn update_visualization_setting(
    name: String,
    value: serde_json::Value,
) -> SettingResult<()> {
    validate_visualization_setting(&name, &value)?;
    
    // Update the setting
    log::debug!("Updating visualization setting {} to {:?}", name, value);
    
    // Save to persistent storage
    save_settings_to_file().await?;
    
    Ok(())
} 
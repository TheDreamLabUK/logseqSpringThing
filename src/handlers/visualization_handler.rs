use crate::config::Settings;
use crate::AppState;
use crate::actors::messages::{GetSettings, UpdateSettings};
use actix_web::{web, HttpResponse};
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// Internal helper function to convert camelCase or kebab-case to snake_case
// This replaces the dependency on case_conversion.rs
fn to_snake_case(s: &str) -> String {
    // First handle kebab-case by replacing hyphens with underscores
    let s = s.replace('-', "_");
    
    // Then handle camelCase by adding underscores before uppercase letters
    let mut result = String::with_capacity(s.len() + 4);
    let mut chars = s.chars().peekable();
    
    while let Some(c) = chars.next() {
        if c.is_ascii_uppercase() {
            // If this is an uppercase letter, add an underscore before it
            // unless it's at the beginning of the string
            if !result.is_empty() {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
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
pub struct SettingValue {
    pub value: Value,
}

fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!(
        "Attempting to get setting value for category: {}, setting: {}",
        category, setting
    );

    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!(
        "Converted category '{}' to snake_case: '{}'",
        category, category_snake
    );
    debug!(
        "Converted setting '{}' to snake_case: '{}'",
        setting, setting_snake
    );

    // Convert settings to Value for easier access
    let settings_value = match serde_json::to_value(&settings) {
        Ok(v) => {
            debug!("Successfully serialized settings to JSON");
            v
        }
        Err(e) => {
            error!("Failed to serialize settings to JSON: {}", e);
            return Err(format!("Failed to serialize settings: {}", e));
        }
    };

    debug!("Settings JSON structure: {}", settings_value);

    // Handle nested categories
    let parts: Vec<&str> = category_snake.split('.').collect();
    let mut current_value = &settings_value;

    for part in parts {
        current_value = match current_value.get(part) {
            Some(v) => {
                debug!("Found category part '{}' in settings", part);
                v
            }
            None => {
                error!("Category part '{}' not found in settings", part);
                return Err(format!("Category '{}' not found", category));
            }
        };
    }

    // Get setting value using snake_case for internal lookup
    let setting_value = match current_value.get(&setting_snake) {
        Some(v) => {
            debug!(
                "Found setting '{}' in category '{}'",
                setting_snake, category_snake
            );
            v
        }
        None => {
            error!(
                "Setting '{}' not found in category '{}'",
                setting_snake, category_snake
            );
            return Err(format!(
                "Setting '{}' not found in category '{}'",
                setting, category
            ));
        }
    };

    debug!("Found setting value: {:?}", setting_value);
    Ok(setting_value.clone())
}

fn update_setting_value(
    settings: &mut Settings,
    category: &str,
    setting: &str,
    value: &Value,
) -> Result<(), String> {
    debug!(
        "Attempting to update setting value for category: {}, setting: {}",
        category, setting
    );

    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!(
        "Converted category '{}' to snake_case: '{}'",
        category, category_snake
    );
    debug!(
        "Converted setting '{}' to snake_case: '{}'",
        setting, setting_snake
    );

    // Convert settings to Value for manipulation
    let mut settings_value = match serde_json::to_value(&*settings) {
        Ok(v) => {
            debug!("Successfully serialized settings to JSON");
            v
        }
        Err(e) => {
            error!("Failed to serialize settings to JSON: {}", e);
            return Err(format!("Failed to serialize settings: {}", e));
        }
    };

    debug!("Settings JSON structure: {}", settings_value);

    // Handle nested categories
    let parts: Vec<&str> = category_snake.split('.').collect();
    let mut current_value = &mut settings_value;

    for part in parts {
        current_value = match current_value.get_mut(part) {
            Some(v) => {
                debug!("Found category part '{}' in settings", part);
                v
            }
            None => {
                error!("Category part '{}' not found in settings", part);
                return Err(format!("Category '{}' not found", category));
            }
        };
    }

    // Update setting value
    if let Some(obj) = current_value.as_object_mut() {
        obj.insert(setting_snake.to_string(), value.clone());
        debug!("Updated setting value successfully");

        // Convert back to Settings
        match serde_json::from_value(settings_value) {
            Ok(new_settings) => {
                debug!("Successfully converted updated JSON back to Settings");
                *settings = new_settings;
                Ok(())
            }
            Err(e) => {
                error!("Failed to convert JSON back to Settings: {}", e);
                Err(format!("Failed to deserialize settings: {}", e))
            }
        }
    } else {
        error!("Category '{}' is not an object", category_snake);
        Err(format!("Category '{}' is not an object", category))
    }
}

fn get_category_settings_value(settings: &Settings, category: &str) -> Result<Value, String> {
    debug!("Getting settings for category: {}", category);
    let value = match category {
        "visualisation.nodes" => serde_json::to_value(&settings.visualisation.nodes)
            .map_err(|e| format!("Failed to serialize node settings: {}", e))?,
        "visualisation.edges" => serde_json::to_value(&settings.visualisation.edges)
            .map_err(|e| format!("Failed to serialize edge settings: {}", e))?,
        "visualisation.rendering" => serde_json::to_value(&settings.visualisation.rendering)
            .map_err(|e| format!("Failed to serialize rendering settings: {}", e))?,
        "visualisation.labels" => serde_json::to_value(&settings.visualisation.labels)
            .map_err(|e| format!("Failed to serialize labels settings: {}", e))?,
        "visualisation.bloom" => serde_json::to_value(&settings.visualisation.bloom)
            .map_err(|e| format!("Failed to serialize bloom settings: {}", e))?,
        "visualisation.animations" => serde_json::to_value(&settings.visualisation.animations)
            .map_err(|e| format!("Failed to serialize animations settings: {}", e))?,
        "visualisation.physics" => serde_json::to_value(&settings.visualisation.physics)
            .map_err(|e| format!("Failed to serialize physics settings: {}", e))?,
        "visualisation.hologram" => serde_json::to_value(&settings.visualisation.hologram)
            .map_err(|e| format!("Failed to serialize hologram settings: {}", e))?,
        "system.network" => serde_json::to_value(&settings.system.network)
            .map_err(|e| format!("Failed to serialize network settings: {}", e))?,
        "system.websocket" => serde_json::to_value(&settings.system.websocket)
            .map_err(|e| format!("Failed to serialize websocket settings: {}", e))?,
        "system.security" => serde_json::to_value(&settings.system.security)
            .map_err(|e| format!("Failed to serialize security settings: {}", e))?,
        "system.debug" => serde_json::to_value(&settings.system.debug)
            .map_err(|e| format!("Failed to serialize debug settings: {}", e))?,
        "xr" => serde_json::to_value(&settings.xr)
            .map_err(|e| format!("Failed to serialize xr settings: {}", e))?,
        "github" => serde_json::to_value(&settings.github)
            .map_err(|e| format!("Failed to serialize github settings: {}", e))?,
        "ragflow" => serde_json::to_value(&settings.ragflow)
            .map_err(|e| format!("Failed to serialize ragflow settings: {}", e))?,
        "perplexity" => serde_json::to_value(&settings.perplexity)
            .map_err(|e| format!("Failed to serialize perplexity settings: {}", e))?,
        "openai" => serde_json::to_value(&settings.openai)
            .map_err(|e| format!("Failed to serialize openai settings: {}", e))?,
        _ => return Err(format!("Invalid category: {}", category)),
    };
    debug!("Successfully retrieved settings for category: {}", category);
    Ok(value)
}

pub async fn get_setting(
    app_state: web::Data<AppState>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!(
        "Getting setting for category: {}, setting: {}",
        category, setting
    );

    let settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some("Failed to get settings".to_string()),
            });
        }
        Err(e) => {
            error!("Settings actor mailbox error: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some("Settings service unavailable".to_string()),
            });
        }
    };

    // Convert AppFullSettings to Settings for compatibility with existing helper functions
    let converted_settings = Settings {
        debug_mode: settings.system.debug.enabled,
        debug: crate::config::DebugSettings {
            enable_websocket_debug: settings.system.debug.enable_websocket_debug,
            enable_data_debug: settings.system.debug.enable_data_debug,
            log_binary_headers: settings.system.debug.log_binary_headers,
            log_full_json: settings.system.debug.log_full_json,
        },
        visualisation: settings.visualisation.clone(),
        system: settings.system.clone(),
        xr: settings.xr.clone(),
        github: settings.github.clone(),
        ragflow: settings.ragflow.clone(),
        perplexity: settings.perplexity.clone(),
        openai: settings.openai.clone(),
        ..Default::default()
    };

    match get_setting_value(&converted_settings, &category, &setting) {
        Ok(value) => {
            debug!("Successfully retrieved setting value: {:?}", value);
            HttpResponse::Ok().json(SettingResponse {
                category,
                setting,
                value,
                success: true,
                error: None,
            })
        }
        Err(e) => {
            error!("Failed to get setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some(e),
            })
        }
    }
}

pub async fn update_setting(
    app_state: web::Data<AppState>,
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!(
        "Updating setting for category: {}, setting: {}",
        category, setting
    );

    // Get current settings
    let mut settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings for update: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: value.into_inner(),
                success: false,
                error: Some("Failed to get settings".to_string()),
            });
        }
        Err(e) => {
            error!("Settings actor mailbox error for update: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: value.into_inner(),
                success: false,
                error: Some("Settings service unavailable".to_string()),
            });
        }
    };

    // Convert AppFullSettings to Settings for compatibility with existing helper functions
    let mut converted_settings = Settings {
        debug_mode: settings.system.debug.enabled,
        debug: crate::config::DebugSettings {
            enable_websocket_debug: settings.system.debug.enable_websocket_debug,
            enable_data_debug: settings.system.debug.enable_data_debug,
            log_binary_headers: settings.system.debug.log_binary_headers,
            log_full_json: settings.system.debug.log_full_json,
        },
        visualisation: settings.visualisation.clone(),
        system: settings.system.clone(),
        xr: settings.xr.clone(),
        github: settings.github.clone(),
        ragflow: settings.ragflow.clone(),
        perplexity: settings.perplexity.clone(),
        openai: settings.openai.clone(),
        ..Default::default()
    };

    match update_setting_value(&mut converted_settings, &category, &setting, &value) {
        Ok(_) => {
            // Convert back to AppFullSettings and update
            settings.system.debug.enabled = converted_settings.debug_mode;
            settings.system.debug.enable_websocket_debug = converted_settings.debug.enable_websocket_debug;
            settings.system.debug.enable_data_debug = converted_settings.debug.enable_data_debug;
            settings.system.debug.log_binary_headers = converted_settings.debug.log_binary_headers;
            settings.system.debug.log_full_json = converted_settings.debug.log_full_json;
            settings.visualisation = converted_settings.visualisation;
            settings.system = converted_settings.system;
            settings.xr = converted_settings.xr;
            settings.github = converted_settings.github;
            settings.ragflow = converted_settings.ragflow;
            settings.perplexity = converted_settings.perplexity;
            settings.openai = converted_settings.openai;

            // Update settings via actor
            match app_state.settings_addr.send(UpdateSettings { settings }).await {
                Ok(Ok(())) => {
                    HttpResponse::Ok().json(SettingResponse {
                        category,
                        setting,
                        value: value.into_inner(),
                        success: true,
                        error: None,
                    })
                }
                Ok(Err(e)) => {
                    error!("Failed to update settings: {}", e);
                    HttpResponse::InternalServerError().json(SettingResponse {
                        category,
                        setting,
                        value: value.into_inner(),
                        success: false,
                        error: Some("Failed to update settings".to_string()),
                    })
                }
                Err(e) => {
                    error!("Settings actor mailbox error during update: {}", e);
                    HttpResponse::InternalServerError().json(SettingResponse {
                        category,
                        setting,
                        value: value.into_inner(),
                        success: false,
                        error: Some("Settings service unavailable".to_string()),
                    })
                }
            }
        }
        Err(e) => {
            error!("Failed to update setting value: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category,
                setting,
                value: value.into_inner(),
                success: false,
                error: Some(e),
            })
        }
    }
}

pub async fn get_category_settings(
    app_state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings for category: {}", e);
            let category = path.into_inner();
            return HttpResponse::InternalServerError().json(CategorySettingsResponse {
                category,
                settings: HashMap::new(),
                success: false,
                error: Some("Failed to get settings".to_string()),
            });
        }
        Err(e) => {
            error!("Settings actor mailbox error for category: {}", e);
            let category = path.into_inner();
            return HttpResponse::InternalServerError().json(CategorySettingsResponse {
                category,
                settings: HashMap::new(),
                success: false,
                error: Some("Settings service unavailable".to_string()),
            });
        }
    };

    // Convert AppFullSettings to Settings for compatibility with existing helper functions
    let converted_settings = Settings {
        debug_mode: settings.system.debug.enabled,
        debug: crate::config::DebugSettings {
            enable_websocket_debug: settings.system.debug.enable_websocket_debug,
            enable_data_debug: settings.system.debug.enable_data_debug,
            log_binary_headers: settings.system.debug.log_binary_headers,
            log_full_json: settings.system.debug.log_full_json,
        },
        visualisation: settings.visualisation.clone(),
        system: settings.system.clone(),
        xr: settings.xr.clone(),
        github: settings.github.clone(),
        ragflow: settings.ragflow.clone(),
        perplexity: settings.perplexity.clone(),
        openai: settings.openai.clone(),
        ..Default::default()
    };

    let debug_enabled = settings.system.debug.enabled;
    let log_json = debug_enabled && settings.system.debug.log_full_json;

    let category = path.into_inner();
    match get_category_settings_value(&converted_settings, &category) {
        Ok(value) => {
            if log_json {
                debug!(
                    "Category '{}' settings: {}",
                    category,
                    serde_json::to_string_pretty(&value).unwrap_or_default()
                );
            }
            let settings_map: HashMap<String, Value> = value
                .as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();

            HttpResponse::Ok().json(CategorySettingsResponse {
                category: category.clone(),
                settings: settings_map,
                success: true,
                error: None,
            })
        }
        Err(e) => {
            error!("Failed to get category settings for '{}': {}", category, e);
            HttpResponse::NotFound().json(CategorySettingsResponse {
                category: category.clone(),
                settings: HashMap::new(),
                success: false,
                error: Some(e),
            })
        }
    }
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.route("/settings/{category}/{setting}", web::get().to(get_setting))
        .route(
            "/settings/{category}/{setting}",
            web::put().to(update_setting),
        )
        .route("/settings/{category}", web::get().to(get_category_settings));
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    debug!("Attempting to save settings to file");

    let settings_path = std::env::var("SETTINGS_FILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));

    info!("Attempting to save settings to: {:?}", settings_path);

    if let Some(parent) = settings_path.parent() {
        match fs::create_dir_all(parent) {
            Ok(_) => debug!("Created parent directories: {:?}", parent),
            Err(e) => {
                error!("Failed to create parent directories: {}", e);
                return Err(e);
            }
        }
    }

    if settings_path.exists() {
        match fs::metadata(&settings_path) {
            Ok(metadata) => {
                if metadata.permissions().readonly() {
                    error!("Settings file is read-only: {:?}", settings_path);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Settings file is read-only",
                    ));
                }
            }
            Err(e) => {
                error!("Failed to check settings file permissions: {}", e);
                return Err(e);
            }
        }
    }

    let yaml_string = match serde_yaml::to_string(&settings) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to serialize settings to YAML: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };

    match fs::write(&settings_path, yaml_string) {
        Ok(_) => {
            info!("Settings saved successfully to: {:?}", settings_path);
            Ok(())
        }
        Err(e) => {
            error!("Failed to write settings file: {}", e);
            Err(e)
        }
    }
}

pub async fn get_visualisation_settings(
    app_state: web::Data<AppState>,
    category: web::Path<String>,
) -> Result<HttpResponse, actix_web::Error> {
    debug!("Getting settings for category: {}", category);

    if category.as_str() == "clientDebug" {
        debug!("Checking UI container status for debugging");
    }

    let settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings for visualisation: {}", e);
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to get settings"
            })));
        }
        Err(e) => {
            error!("Settings actor mailbox error for visualisation: {}", e);
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    Ok(HttpResponse::Ok().json(&settings))
}

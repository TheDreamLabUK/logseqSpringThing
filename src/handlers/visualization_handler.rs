use crate::config::Settings;
use actix_web::{get, put, web, HttpResponse};
use log::{error, info, debug};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::utils::case_conversion::to_snake_case;
use crate::handlers::settings::common::get_category_settings_value;

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

// Request/Response structures for individual settings
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingValue {
    pub value: Value,
}

// Helper function to get setting value from settings object
fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    debug!("Attempting to get setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
    let category_snake = to_snake_case(category);
    let setting_snake = to_snake_case(setting);
    debug!("Converted category '{}' to snake_case: '{}'", category, category_snake);
    debug!("Converted setting '{}' to snake_case: '{}'", setting, setting_snake);
    
    // Convert settings to Value for easier access
    let settings_value = serde_json::to_value(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;

    // Determine the root category (visualization, xr, or system)
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
            debug!("Found setting '{}' in {}.{}", setting_snake, root_category, sub_category);
            Ok(v.clone())
        },
        None => {
            error!("Setting '{}' not found in {}.{}", setting_snake, root_category, sub_category);
            Err(format!("Setting '{}' not found in {}.{}", setting, root_category, sub_category))
        }
    }
}

// Helper function to update setting value in settings object
fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    debug!("Attempting to update setting value for category: {}, setting: {}", category, setting);
    
    // Convert kebab-case URL parameters to snake_case
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

    // Convert the value to the appropriate type and update the settings
    match serde_json::from_value(value.clone()) {
        Ok(v) => {
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
                obj.insert(setting_snake.clone(), v);
                
                // Convert back to Settings
                *settings = serde_json::from_value(settings_value)
                    .map_err(|e| format!("Failed to update settings: {}", e))?;
                
                Ok(())
            } else {
                Err(format!("Invalid settings structure for {}.{}", root_category, sub_category))
            }
        },
        Err(e) => Err(format!("Invalid value for setting: {}", e))
    }
}

// GET /api/visualization/settings/{category}
#[get("/api/visualization/settings/{category}")]
pub async fn get_category_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>
) -> HttpResponse {
    let category = path.into_inner();
    debug!("Getting settings for category: {}", category);

    let settings_guard = settings.read().await;
    match get_category_settings_value(&settings_guard, &category) {
        Ok(value) => {
            let settings_map: HashMap<String, Value> = value.as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();
            
            HttpResponse::Ok().json(CategorySettingsResponse {
                category,
                settings: settings_map,
                success: true,
                error: None,
            })
        }
        Err(e) => {
            error!("Failed to get category settings: {}", e);
            HttpResponse::BadRequest().json(CategorySettingsResponse {
                category,
                settings: HashMap::new(),
                success: false,
                error: Some(e),
            })
        }
    }
}

// PUT /api/visualization/settings/{category}
#[put("/api/visualization/settings/{category}")]
pub async fn update_category_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    update: web::Json<CategorySettingsUpdate>
) -> HttpResponse {
    let category = path.into_inner();
    debug!("Updating settings for category: {}", category);

    let mut settings_guard = settings.write().await;
    let settings_map: HashMap<String, Value> = update.settings.clone();

    let mut success = true;
    let mut error_msg = None;

    for (setting, value) in settings_map {
        if let Err(e) = update_setting_value(&mut settings_guard, &category, &setting, &value) {
            error!("Failed to update setting {}.{}: {}", category, setting, e);
            success = false;
            error_msg = Some(e);
            break;
        }
    }

    if success {
        if let Err(e) = save_settings_to_file(&settings_guard) {
            error!("Failed to save settings to file: {}", e);
            success = false;
            error_msg = Some(format!("Failed to save settings: {}", e));
        }
    }

    HttpResponse::Ok().json(CategorySettingsResponse {
        category,
        settings: update.settings.clone(),
        success,
        error: error_msg,
    })
}

// GET /api/visualization/settings/{category}/{setting}
pub async fn get_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!("Getting setting for category: {}, setting: {}", category, setting);
    
    let settings_guard = match settings.read().await {
        guard => {
            debug!("Successfully acquired settings read lock");
            guard
        }
    };

    match get_setting_value(&*settings_guard, &category, &setting) {
        Ok(value) => {
            debug!("Successfully retrieved setting value: {:?}", value);
            HttpResponse::Ok().json(SettingResponse {
                category,
                setting,
                value,
                success: true,
                error: None,
            })
        },
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

// PUT /api/visualization/settings/{category}/{setting}
pub async fn update_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!("Updating setting for category: {}, setting: {}", category, setting);
    
    let mut settings_guard = match settings.write().await {
        guard => {
            debug!("Successfully acquired settings write lock");
            guard
        }
    };

    match update_setting_value(&mut *settings_guard, &category, &setting, &value) {
        Ok(_) => {
            if let Err(e) = save_settings_to_file(&*settings_guard) {
                error!("Failed to save settings to file: {}", e);
                return HttpResponse::InternalServerError().json(SettingResponse {
                    category,
                    setting,
                    value: value.into_inner(),
                    success: false,
                    error: Some("Failed to persist settings".to_string()),
                });
            }
            HttpResponse::Ok().json(SettingResponse {
                category,
                setting,
                value: value.into_inner(),
                success: true,
                error: None,
            })
        },
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

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(get_category_settings)
       .service(update_category_settings)
       .service(
           web::resource("/api/visualization/settings/{category}/{setting}")
               .route(web::get().to(get_setting))
               .route(web::put().to(update_setting))
       );
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    debug!("Attempting to save settings to file");
    
    // Use absolute path from environment or default to /app/settings.toml
    let settings_path = std::env::var("SETTINGS_FILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/app/settings.toml"));
    
    info!("Attempting to save settings to: {:?}", settings_path);
    
    // Ensure parent directory exists and is writable
    if let Some(parent) = settings_path.parent() {
        match fs::create_dir_all(parent) {
            Ok(_) => debug!("Created parent directories: {:?}", parent),
            Err(e) => {
                error!("Failed to create parent directories: {}", e);
                return Err(e);
            }
        }
    }
    
    // Check if file exists and is writable
    if settings_path.exists() {
        match fs::metadata(&settings_path) {
            Ok(metadata) => {
                if metadata.permissions().readonly() {
                    error!("Settings file is read-only: {:?}", settings_path);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Settings file is read-only"
                    ));
                }
            }
            Err(e) => {
                error!("Failed to check settings file permissions: {}", e);
                return Err(e);
            }
        }
    }
    
    // Convert settings to TOML
    let toml_string = match toml::to_string_pretty(&settings) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to serialize settings to TOML: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };
    
    // Write to settings.toml
    match fs::write(&settings_path, toml_string) {
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

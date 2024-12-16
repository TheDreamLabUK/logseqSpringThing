use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::fs;
use std::path::PathBuf;
use toml;
use log::{error, info, debug};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::utils::case_conversion::{to_snake_case, to_camel_case};

// Request/Response structures for individual settings
#[derive(Debug, Serialize, Deserialize)]
pub struct SettingValue {
    pub value: Value,
}

// GET /api/visualization/settings/{category}/{setting} - Get setting for a specific category and setting
pub async fn get_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    debug!("Getting setting for category: {}, setting: {}", category, setting);
    
    // Acquire read lock and log any errors
    let settings_guard = match settings.read().await {
        guard => {
            debug!("Successfully acquired settings read lock");
            guard
        }
    };

    // Get setting value from settings object
    match get_setting_value(&*settings_guard, &category, &setting) {
        Ok(value) => {
            debug!("Successfully retrieved setting value: {:?}", value);
            HttpResponse::Ok().json(SettingValue { value })
        },
        Err(e) => {
            error!("Failed to get setting value: {}", e);
            HttpResponse::BadRequest().json(serde_json::json!({
                "error": format!("Failed to get setting: {}", e)
            }))
        }
    }
}

// PUT /api/visualization/settings/{category}/{setting} - Update setting for a specific category and setting
pub async fn update_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<(String, String)>,
    value: web::Json<SettingValue>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    debug!("Updating setting for category: {}, setting: {}", category, setting);
    
    // Acquire write lock and log any errors
    let mut settings_guard = match settings.write().await {
        guard => {
            debug!("Successfully acquired settings write lock");
            guard
        }
    };

    // Update setting value in settings object
    match update_setting_value(&mut *settings_guard, &category, &setting, &value.value) {
        Ok(_) => {
            // Save settings to file after successful update
            if let Err(e) = save_settings_to_file(&*settings_guard) {
                error!("Failed to save settings to file: {}", e);
                return HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Failed to save settings to file"
                }));
            }
            HttpResponse::Ok().json(serde_json::json!({ "status": "success" }))
        },
        Err(e) => {
            error!("Failed to update setting value: {}", e);
            HttpResponse::BadRequest().json(serde_json::json!({
                "error": format!("Failed to update setting: {}", e)
            }))
        }
    }
}

// Helper function to get setting value from settings object
fn get_setting_value(settings: &Settings, category: &str, setting: &str) -> Result<Value, String> {
    let camel_setting = to_camel_case(setting);
    
    // Match on category and get setting value
    let value = match category {
        "nodes" => serde_json::to_value(&settings.nodes)
            .map_err(|e| format!("Failed to serialize node settings: {}", e))?
            .get(&camel_setting)
            .cloned()
            .ok_or_else(|| format!("Setting {} not found in nodes", setting))?,
            
        "edges" => serde_json::to_value(&settings.edges)
            .map_err(|e| format!("Failed to serialize edge settings: {}", e))?
            .get(&camel_setting)
            .cloned()
            .ok_or_else(|| format!("Setting {} not found in edges", setting))?,
            
        "rendering" => serde_json::to_value(&settings.rendering)
            .map_err(|e| format!("Failed to serialize rendering settings: {}", e))?
            .get(&camel_setting)
            .cloned()
            .ok_or_else(|| format!("Setting {} not found in rendering", setting))?,
            
        "labels" => serde_json::to_value(&settings.labels)
            .map_err(|e| format!("Failed to serialize labels settings: {}", e))?
            .get(&camel_setting)
            .cloned()
            .ok_or_else(|| format!("Setting {} not found in labels", setting))?,
            
        "bloom" => serde_json::to_value(&settings.bloom)
            .map_err(|e| format!("Failed to serialize bloom settings: {}", e))?
            .get(&camel_setting)
            .cloned()
            .ok_or_else(|| format!("Setting {} not found in bloom", setting))?,
            
        _ => return Err(format!("Category {} not found", category))
    };

    Ok(value)
}

// Helper function to update setting value in settings object
fn update_setting_value(settings: &mut Settings, category: &str, setting: &str, value: &Value) -> Result<(), String> {
    let camel_setting = to_camel_case(setting);
    
    // Match on category and update setting value
    match category {
        "nodes" => {
            let mut nodes = serde_json::to_value(&settings.nodes)
                .map_err(|e| format!("Failed to serialize node settings: {}", e))?;
            
            if let Some(obj) = nodes.as_object_mut() {
                obj.insert(camel_setting.clone(), value.clone());
                settings.nodes = serde_json::from_value(nodes)
                    .map_err(|e| format!("Failed to deserialize node settings: {}", e))?;
            }
        },
        "edges" => {
            let mut edges = serde_json::to_value(&settings.edges)
                .map_err(|e| format!("Failed to serialize edge settings: {}", e))?;
            
            if let Some(obj) = edges.as_object_mut() {
                obj.insert(camel_setting.clone(), value.clone());
                settings.edges = serde_json::from_value(edges)
                    .map_err(|e| format!("Failed to deserialize edge settings: {}", e))?;
            }
        },
        "rendering" => {
            let mut rendering = serde_json::to_value(&settings.rendering)
                .map_err(|e| format!("Failed to serialize rendering settings: {}", e))?;
            
            if let Some(obj) = rendering.as_object_mut() {
                obj.insert(camel_setting.clone(), value.clone());
                settings.rendering = serde_json::from_value(rendering)
                    .map_err(|e| format!("Failed to deserialize rendering settings: {}", e))?;
            }
        },
        "labels" => {
            let mut labels = serde_json::to_value(&settings.labels)
                .map_err(|e| format!("Failed to serialize labels settings: {}", e))?;
            
            if let Some(obj) = labels.as_object_mut() {
                obj.insert(camel_setting.clone(), value.clone());
                settings.labels = serde_json::from_value(labels)
                    .map_err(|e| format!("Failed to deserialize labels settings: {}", e))?;
            }
        },
        "bloom" => {
            let mut bloom = serde_json::to_value(&settings.bloom)
                .map_err(|e| format!("Failed to serialize bloom settings: {}", e))?;
            
            if let Some(obj) = bloom.as_object_mut() {
                obj.insert(camel_setting.clone(), value.clone());
                settings.bloom = serde_json::from_value(bloom)
                    .map_err(|e| format!("Failed to deserialize bloom settings: {}", e))?;
            }
        },
        _ => return Err(format!("Category {} not found", category))
    }
    
    Ok(())
}

// GET /api/visualization/settings/{category} - Get all settings for a category
pub async fn get_category_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let category = path.into_inner();
    debug!("Getting all settings for category: {}", category);
    
    let settings_guard = match settings.read().await {
        guard => {
            debug!("Successfully acquired settings read lock");
            guard
        }
    };

    // Get all settings for the category
    match get_category_settings_value(&*settings_guard, &category) {
        Ok(value) => {
            debug!("Successfully retrieved category settings: {:?}", value);
            HttpResponse::Ok().json(value)
        },
        Err(e) => {
            error!("Failed to get category settings: {}", e);
            HttpResponse::BadRequest().json(serde_json::json!({
                "error": format!("Failed to get category settings: {}", e)
            }))
        }
    }
}

// Helper function to get all settings for a category
fn get_category_settings_value(settings: &Settings, category: &str) -> Result<Value, String> {
    let category_snake = to_snake_case(category);
    
    // Use reflection to get the category field
    match serde_json::to_value(settings) {
        Ok(settings_value) => {
            if let Some(obj) = settings_value.as_object() {
                if let Some(category_value) = obj.get(&category_snake) {
                    Ok(category_value.clone())
                } else {
                    Err(format!("Category '{}' not found", category))
                }
            } else {
                Err("Settings is not an object".to_string())
            }
        },
        Err(e) => Err(format!("Failed to serialize settings: {}", e))
    }
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/visualization/settings")
            .service(
                web::resource("/{category}")
                    .route(web::get().to(get_category_settings))
            )
            .service(
                web::resource("/{category}/{setting}")
                    .route(web::get().to(get_setting))
                    .route(web::put().to(update_setting))
            )
    );
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    // Convert settings to TOML
    let toml_string = match toml::to_string_pretty(&settings) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to serialize settings to TOML: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };
    
    // Use absolute path from environment or default to /app/settings.toml
    let settings_path = std::env::var("SETTINGS_FILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/app/settings.toml"));
    
    info!("Attempting to save settings to: {:?}", settings_path);
    
    // Ensure parent directory exists and is writable
    if let Some(parent) = settings_path.parent() {
        match fs::create_dir_all(parent) {
            Ok(_) => info!("Created parent directories: {:?}", parent),
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

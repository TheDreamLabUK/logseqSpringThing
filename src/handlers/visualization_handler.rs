use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use toml;

// GET /api/visualization/settings
pub async fn get_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    let settings_guard = settings.read().await;
    let settings_json = serde_json::json!({
        "rendering": settings_guard.rendering,
        "nodes": settings_guard.nodes,
        "edges": settings_guard.edges,
        "labels": settings_guard.labels,
        "bloom": settings_guard.bloom,
        "ar": settings_guard.ar,
        "physics": settings_guard.physics,
        "animations": settings_guard.animations,
        "audio": settings_guard.audio
    });

    HttpResponse::Ok().json(settings_json)
}

// PUT /api/visualization/settings
pub async fn update_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    new_settings: web::Json<Value>,
) -> HttpResponse {
    let mut settings_guard = settings.write().await;
    
    // Update settings in memory
    update_settings(&mut *settings_guard, new_settings.into_inner());
    
    // Save settings to file
    match save_settings_to_file(&*settings_guard) {
        Ok(_) => {
            let updated_json = serde_json::json!({
                "rendering": settings_guard.rendering,
                "nodes": settings_guard.nodes,
                "edges": settings_guard.edges,
                "labels": settings_guard.labels,
                "bloom": settings_guard.bloom,
                "ar": settings_guard.ar,
                "physics": settings_guard.physics,
                "animations": settings_guard.animations,
                "audio": settings_guard.audio
            });
            HttpResponse::Ok().json(updated_json)
        },
        Err(e) => {
            log::error!("Failed to save settings to file: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to save settings"
            }))
        }
    }
}

fn update_settings(settings: &mut Settings, new_settings: Value) {
    if let Some(obj) = new_settings.as_object() {
        for (section, values) in obj {
            match section.as_str() {
                "rendering" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.rendering, v);
                },
                "nodes" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.nodes, v);
                },
                "edges" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.edges, v);
                },
                "labels" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.labels, v);
                },
                "bloom" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.bloom, v);
                },
                "ar" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.ar, v);
                },
                "physics" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.physics, v);
                },
                "animations" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.animations, v);
                },
                "audio" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.audio, v);
                },
                _ => {}
            }
        }
    }
}

fn update_section<T: serde::de::DeserializeOwned + serde::Serialize>(
    section: &mut T,
    values: &serde_json::Map<String, Value>
) {
    // Create a reference to section to avoid moving it
    let current = serde_json::to_value(&*section).unwrap_or(Value::Null);
    if let Value::Object(mut current_map) = current {
        // Update only provided values
        for (key, value) in values {
            current_map.insert(key.clone(), value.clone());
        }
        // Convert back to the original type
        if let Ok(updated) = serde_json::from_value(Value::Object(current_map)) {
            *section = updated;
        }
    }
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    // Convert settings to TOML
    let toml_string = toml::to_string_pretty(&settings)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    // Get the absolute path to settings.toml
    let settings_path = PathBuf::from(std::env::current_dir()?)
        .join("settings.toml");
    
    // Write to settings.toml with absolute path
    fs::write(&settings_path, toml_string)?;
    
    log::info!("Settings saved to: {:?}", settings_path);
    
    Ok(())
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/visualization")
            .route("/settings", web::get().to(get_visualization_settings))
            .route("/settings", web::put().to(update_visualization_settings))
    );
}

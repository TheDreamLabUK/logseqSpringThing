use crate::config::Settings;
use crate::utils::socket_flow_messages::{Message, SettingsUpdate, UpdateSettings};
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value;
use std::fs;
use toml;

pub async fn get_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    let settings = settings.read().await;
    let settings_json = serde_json::json!({
        "rendering": settings.rendering,
        "nodes": settings.nodes,
        "edges": settings.edges,
        "labels": settings.labels,
        "bloom": settings.bloom,
        "ar": settings.ar,
        "physics": settings.physics,
        "animations": settings.animations,
        "audio": settings.audio
    });

    HttpResponse::Ok().json(settings_json)
}

pub async fn handle_settings_message(
    message: Message,
    settings: Arc<RwLock<Settings>>,
) -> Option<Message> {
    match message {
        Message::UpdateSettings(UpdateSettings { settings: new_settings }) => {
            // Update settings in memory
            let mut settings_lock = settings.write().await;
            update_settings(&mut *settings_lock, new_settings);
            
            // Save settings to file
            if let Err(e) = save_settings_to_file(&*settings_lock) {
                log::error!("Failed to save settings to file: {}", e);
            }
            
            // Send updated settings back to all clients
            Some(Message::SettingsUpdated(SettingsUpdate {
                settings: serde_json::json!({
                    "rendering": settings_lock.rendering,
                    "nodes": settings_lock.nodes,
                    "edges": settings_lock.edges,
                    "labels": settings_lock.labels,
                    "bloom": settings_lock.bloom,
                    "ar": settings_lock.ar,
                    "physics": settings_lock.physics,
                    "animations": settings_lock.animations,
                    "audio": settings_lock.audio
                })
            }))
        },
        _ => None
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
    
    // Write to settings.toml
    fs::write("settings.toml", toml_string)?;
    
    Ok(())
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/visualization")
            .route("/settings", web::get().to(get_visualization_settings))
    );
}

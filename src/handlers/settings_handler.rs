use actix_web::{web, HttpResponse, Error};
use serde_json::{Value, json};
use std::sync::Arc;
use log::{debug, error};

use crate::app_state::AppState;
use crate::utils::case_conversion::{to_camel_case, to_snake_case};

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")  // Matches /api/settings from parent scope
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings))
    );
}

async fn get_settings(state: web::Data<Arc<AppState>>) -> Result<HttpResponse, Error> {
    debug!("Handling GET settings request");
    let settings = state.settings.read().await;
    
    // Transform settings using explicit structure
    let client_settings = json!({
        "visualization": {
            "nodes": convert_struct_to_camel_case(&settings.nodes),
            "edges": convert_struct_to_camel_case(&settings.edges),
            "physics": convert_struct_to_camel_case(&settings.physics),
            "rendering": convert_struct_to_camel_case(&settings.rendering),
            "animations": convert_struct_to_camel_case(&settings.animations),
            "labels": convert_struct_to_camel_case(&settings.labels),
            "bloom": convert_struct_to_camel_case(&settings.bloom),
            "hologram": convert_struct_to_camel_case(&settings.hologram),
            "xr": convert_struct_to_camel_case(&settings.xr),
        },
        "system": {
            "network": convert_struct_to_camel_case(&settings.network),
            "websocket": convert_struct_to_camel_case(&settings.websocket),
            "debug": convert_struct_to_camel_case(&settings.debug)
        }
    });

    debug!("Sending settings response");
    Ok(HttpResponse::Ok().json(client_settings))
}

async fn update_settings(
    state: web::Data<Arc<AppState>>,
    payload: web::Json<Value>
) -> Result<HttpResponse, Error> {
    debug!("Handling POST settings update: {:?}", payload);

    // Validate the incoming settings
    if let Err(e) = validate_settings(&payload) {
        error!("Settings validation failed: {}", e);
        return Ok(HttpResponse::BadRequest().body(format!("Invalid settings: {}", e)));
    }

    // Convert incoming camelCase to snake_case
    let snake_case_settings = convert_to_snake_case(&payload);

    // Get write lock and dereference to access Settings methods
    let mut settings_guard = state.settings.write().await;
    let settings = &mut *settings_guard;
    
    // Merge the new settings with existing ones
    if let Err(e) = settings.merge(snake_case_settings) {
        error!("Settings merge failed: {}", e);
        return Ok(HttpResponse::BadRequest().body(format!("Failed to merge settings: {}", e)));
    }
    
    // Save the updated settings
    if let Err(e) = settings.save() {
        error!("Settings save failed: {}", e);
        return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
    }
    
    debug!("Settings updated successfully");
    
    // Convert settings to camelCase for client response
    let client_settings = convert_struct_to_camel_case(settings);
    Ok(HttpResponse::Ok().json(client_settings))
}

fn validate_settings(settings: &Value) -> Result<(), String> {
    if !settings.is_object() {
        return Err("Settings must be an object".to_string());
    }

    // Add additional validation as needed
    // For example, check required fields, value types, etc.

    Ok(())
}

// Helper function to convert struct fields to camelCase
fn convert_struct_to_camel_case<T: serde::Serialize>(value: &T) -> Value {
    let json_value = serde_json::to_value(value).unwrap_or_default();
    convert_to_camel_case(&json_value)
}

// Recursive function to convert all object keys to camelCase
fn convert_to_camel_case(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                let camel_key = to_camel_case(k);
                new_map.insert(camel_key, convert_to_camel_case(v));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(convert_to_camel_case).collect())
        }
        _ => value.clone()
    }
}

// Recursive function to convert all object keys to snake_case
fn convert_to_snake_case(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                let snake_key = to_snake_case(k);
                new_map.insert(snake_key, convert_to_snake_case(v));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(convert_to_snake_case).collect())
        }
        _ => value.clone()
    }
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    debug!("Handling GET graph settings request");
    let settings = app_state.settings.read().await;
    
    // Create visualization settings object and convert to camelCase
    let visualization = json!({
        "visualization": {
            "nodes": convert_struct_to_camel_case(&settings.nodes),
            "edges": convert_struct_to_camel_case(&settings.edges),
            "physics": convert_struct_to_camel_case(&settings.physics),
            "labels": convert_struct_to_camel_case(&settings.labels)
        }
    });
    
    Ok(HttpResponse::Ok().json(visualization))
}
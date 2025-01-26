use actix_web::{web, HttpResponse, Error};
use serde_json::{Value, json};
use std::sync::Arc;

use crate::app_state::AppState;
use crate::utils::case_conversion::to_camel_case;

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/settings")
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings))
    );
}

async fn get_settings(state: web::Data<Arc<AppState>>) -> Result<HttpResponse, Error> {
    // RwLock::read() returns the guard directly
    let settings = state.settings.read().await;
    // Convert settings to camelCase for client
    let client_settings = convert_struct_to_camel_case(&*settings);
    Ok(HttpResponse::Ok().json(client_settings))
}

async fn update_settings(
    state: web::Data<Arc<AppState>>,
    payload: web::Json<Value>
) -> Result<HttpResponse, Error> {
    // Validate the incoming settings
    if let Err(e) = validate_settings(&payload) {
        return Ok(HttpResponse::BadRequest().body(format!("Invalid settings: {}", e)));
    }

    // Get write lock and dereference to access Settings methods
    let mut settings_guard = state.settings.write().await;
    let settings = &mut *settings_guard;
    
    // Merge the new settings with existing ones
    if let Err(e) = settings.merge(payload.into_inner()) {
        return Ok(HttpResponse::BadRequest().body(format!("Failed to merge settings: {}", e)));
    }
    
    // Save the updated settings
    if let Err(e) = settings.save() {
        return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
    }
    
    // Convert settings to camelCase for client response
    let client_settings = convert_struct_to_camel_case(settings);
    Ok(HttpResponse::Ok().json(client_settings))
}

fn validate_settings(settings: &Value) -> Result<(), String> {
    // Add validation logic here
    // For now, just ensure it's an object
    if !settings.is_object() {
        return Err("Settings must be an object".to_string());
    }
    Ok(())
}

// Helper function to convert struct fields to camelCase
fn convert_struct_to_camel_case<T: serde::Serialize>(value: &T) -> serde_json::Value {
    let json_value = serde_json::to_value(value).unwrap_or_default();
    
    if let serde_json::Value::Object(map) = json_value {
        let converted: serde_json::Map<String, serde_json::Value> = map
            .into_iter()
            .map(|(k, v)| (to_camel_case(&k), v))
            .collect();
        serde_json::Value::Object(converted)
    } else {
        json_value
    }
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
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
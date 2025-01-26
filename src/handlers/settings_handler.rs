use actix_web::{web, HttpResponse, Error};
use serde_json::{Value, json};
use crate::app_state::AppState;
use crate::utils::case_conversion::to_camel_case;

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/settings")
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings))
    );
}

async fn get_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = state.settings.read().await;
    let settings_json = serde_json::to_value(&*settings_guard)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    Ok(HttpResponse::Ok().json(settings_json))
}

async fn update_settings(
    state: web::Data<AppState>,
    payload: web::Json<Value>
) -> Result<HttpResponse, Error> {
    if let Err(e) = validate_settings(&payload) {
        return Ok(HttpResponse::BadRequest().body(format!("Invalid settings: {}", e)));
    }

    let mut settings_guard = state.settings.write().await;
    
    if let Err(e) = settings_guard.merge(payload.into_inner()) {
        return Ok(HttpResponse::BadRequest().body(format!("Failed to merge settings: {}", e)));
    }
    
    if let Err(e) = settings_guard.save() {
        return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
    }
    
    let settings_json = serde_json::to_value(&*settings_guard)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    Ok(HttpResponse::Ok().json(settings_json))
}

fn validate_settings(settings: &Value) -> Result<(), String> {
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
    Ok(HttpResponse::Ok().json(json!({
        "visualization": {
            "nodes": settings.nodes,
            "edges": settings.edges,
            "physics": settings.physics,
            "labels": settings.labels
        }
    })))
}
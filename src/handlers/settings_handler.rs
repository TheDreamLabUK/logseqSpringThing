use actix_web::{web, HttpResponse, Result, Error};
use crate::AppState;
use serde_json::json;
use crate::utils::case_conversion::to_camel_case;
use serde::{Deserialize, Serialize};

pub async fn get_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings.read().await;
    
    // Transform settings using existing case conversion
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
        },
        "system": {
            "network": convert_struct_to_camel_case(&settings.network),
            "websocket": convert_struct_to_camel_case(&settings.websocket),
            "debug": convert_struct_to_camel_case(&settings.debug)
        }
    });

    Ok(HttpResponse::Ok().json(client_settings))
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

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
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

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings))
    ).service(
        web::resource("/graph")
            .route(web::get().to(get_graph_settings))
    );
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SettingsUpdate {
    pub path: String,
    pub value: serde_json::Value,
}

pub async fn update_settings(
    app_state: web::Data<AppState>,
    updates: web::Json<Vec<SettingsUpdate>>,
) -> Result<HttpResponse, Error> {
    let mut settings = app_state.settings.write().await;
    
    for update in updates.iter() {
        // Convert camelCase path to snake_case for server-side storage
        let server_path = to_snake_case(&update.path);
        
        // Update the settings using the path
        let path_parts: Vec<&str> = server_path.split('.').collect();
        let mut current = &mut *settings;
        
        for (i, &part) in path_parts.iter().enumerate() {
            if i == path_parts.len() - 1 {
                // Last part - set the value
                if let Some(obj) = current.as_object_mut() {
                    obj.insert(part.to_string(), update.value.clone());
                }
            } else {
                // Navigate to next level
                if let Some(obj) = current.as_object_mut() {
                    if !obj.contains_key(part) {
                        obj.insert(part.to_string(), json!({}));
                    }
                    current = obj.get_mut(part).unwrap();
                }
            }
        }
    }

    Ok(HttpResponse::Ok().json(json!({ "status": "success" })))
}
use crate::app_state::AppState;
use crate::models::UISettings;
use actix_web::{web, Error, HttpResponse};
use serde_json::{json, Value};

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/settings")
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings)),
    );
}

async fn get_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = state.settings.read().await;
    
    // Convert to UI settings
    let ui_settings = UISettings::from(&*settings_guard);
    
    Ok(HttpResponse::Ok().json(ui_settings))
}

async fn update_settings(
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    // Parse and validate the incoming settings as UISettings
    let ui_settings: UISettings = match serde_json::from_value(payload.into_inner()) {
        Ok(settings) => settings,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().body(format!("Invalid settings format: {}", e)));
        }
    };

    // Update the main settings with UI settings
    let mut settings_guard = state.settings.write().await;
    ui_settings.merge_into_settings(&mut settings_guard);

    // Save the updated settings
    if let Err(e) = settings_guard.save() {
        return Ok(
            HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e))
        );
    }

    // Return the updated UI settings
    let updated_ui_settings = UISettings::from(&*settings_guard);
    Ok(HttpResponse::Ok().json(updated_ui_settings))
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings = app_state.settings.read().await;
    Ok(HttpResponse::Ok().json(json!({
        "visualization": {
            "nodes": settings.visualization.nodes,
            "edges": settings.visualization.edges,
            "physics": settings.visualization.physics,
            "labels": settings.visualization.labels
        }
    })))
}

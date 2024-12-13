use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;

pub async fn get_visualization_settings(
    settings: web::Data<Arc<Settings>>,
) -> HttpResponse {
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

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/visualization")
            .route("/settings", web::get().to(get_visualization_settings))
    );
}

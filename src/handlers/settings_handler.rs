use actix_web::{web, HttpResponse, Result};
use crate::AppState;
use serde_json::json;

pub async fn get_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings.read().await;
    Ok(HttpResponse::Ok().json(&*settings))
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings.read().await;
    Ok(HttpResponse::Ok().json(json!({
        "nodes": settings.nodes,
        "edges": settings.edges,
        "physics": settings.physics,
        "labels": settings.labels
    })))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(get_settings))
    ).service(
        web::resource("/graph")
            .route(web::get().to(get_graph_settings))
    );
} 
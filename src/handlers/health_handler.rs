use actix_web::{web, HttpResponse, Result};
use crate::AppState;

pub async fn health_check(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let metadata = app_state.metadata.read().await;
    let graph = app_state.graph_service.graph_data.read().await;
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "metadata_count": metadata.len(),
        "nodes_count": graph.nodes.len(),
        "edges_count": graph.edges.len()
    })))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(health_check))
    );
} 
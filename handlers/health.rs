use actix_web::{web, HttpResponse};
use serde_json::json;
use std::sync::atomic::Ordering;
use crate::state::AppState;

pub async fn health_check(state: web::Data<AppState>) -> HttpResponse {
    let pool_status = state.db_pool.get_pool_status();
    let metrics = state.metrics.get_metrics().await;
    
    HttpResponse::Ok().json(json!({
        "status": "healthy",
        "database": {
            "connections": {
                "total": pool_status.total_connections,
                "available": pool_status.available_connections,
                "max": pool_status.max_connections
            }
        },
        "metrics": metrics,
        "websocket_connections": metrics.websocket_connections,
        "gpu_compute_enabled": state.gpu_compute_enabled.load(Ordering::Relaxed)
    }))
} 
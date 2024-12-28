use actix_web::{web, HttpResponse};
use serde_json::json;
use std::sync::atomic::Ordering;
use crate::state::AppState;

pub async fn get_gpu_compute_status(state: web::Data<AppState>) -> HttpResponse {
    HttpResponse::Ok().json(json!({
        "enabled": state.gpu_compute_enabled.load(Ordering::Relaxed)
    }))
}

pub async fn set_gpu_compute_status(
    state: web::Data<AppState>,
    enabled: web::Json<bool>,
) -> HttpResponse {
    state.gpu_compute_enabled.store(*enabled, Ordering::Relaxed);
    HttpResponse::Ok().finish()
} 
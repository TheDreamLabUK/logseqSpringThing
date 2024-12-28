use actix_web::{web, HttpResponse};
use crate::state::AppState;

pub async fn get_metrics(state: web::Data<AppState>) -> HttpResponse {
    let metrics = state.metrics.get_metrics().await;
    HttpResponse::Ok().json(metrics)
} 
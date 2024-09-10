use actix_web::{web, HttpResponse};
use crate::AppState;

pub async fn get_graph_data(_state: web::Data<AppState>) -> HttpResponse {
    // Implementation
    HttpResponse::Ok().finish()
}

pub async fn refresh_graph(_state: web::Data<AppState>) -> HttpResponse {
    // Implementation
    HttpResponse::Ok().finish()
}

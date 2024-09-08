use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;

// Define your AppState and other required structs here
struct AppState {
    // Your AppState fields
}

pub async fn get_graph_data(state: web::Data<AppState>) -> HttpResponse {
    // Get graph data logic here
    HttpResponse::Ok().finish()
}

pub async fn refresh_graph(state: web::Data<AppState>) -> HttpResponse {
    // Refresh graph data logic here
    HttpResponse::Ok().finish()
}

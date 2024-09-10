use actix_web::{web, App, HttpServer, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::services::file_service::{FileService, RealGitHubService, RealOpenWebUIService};
use crate::models::metadata::Metadata;

mod services;
mod models;
mod handlers;

struct AppState {
    file_cache: Arc<RwLock<std::collections::HashMap<String, String>>>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let app_state = web::Data::new(AppState {
        file_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/fetch-and-process", web::get().to(handlers::file_handler::fetch_and_process_files))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
use actix_web::{web, App, HttpServer};
use webxr_graph::handlers::file_handler::fetch_and_process_files;
use webxr_graph::app_state::AppState;
use webxr_graph::models::graph::GraphData;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use webxr_graph::config::Settings;
use webxr_graph::services::perplexity_service::RealApiClient;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let settings = Settings::new().unwrap();
    let api_client = Arc::new(RealApiClient::new());

    let app_state = web::Data::new(AppState {
        graph_data,
        file_cache,
        settings,
        api_client,
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/fetch-and-process", web::post().to(fetch_and_process_files))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

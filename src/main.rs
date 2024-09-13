use actix_web::{web, App, HttpServer};
use webxr_graph::handlers::file_handler::fetch_and_process_files;
use webxr_graph::app_state::AppState;
use webxr_graph::models::graph::GraphData;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let app_state = web::Data::new(AppState { file_cache, graph_data });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/fetch-and-process", web::post().to(fetch_and_process_files))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
use std::sync::Arc;
use tokio::sync::RwLock;
use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use log::{error, info};
use env_logger;
use dotenvy::dotenv;

use webxr::{
    AppState,
    Settings,
    handlers::{
        file_handler::{fetch_and_process_files, get_file_content, refresh_graph as file_refresh_graph},
        graph_handler::{get_graph_data, get_paginated_graph_data, refresh_graph as graph_refresh_graph},
        settings::visualization::config,
        socket_flow_handler::socket_flow_handler,
    },
    utils::gpu_compute::GPUCompute,
    services::{
        file_service::RealGitHubService,
        github_service::RealGitHubPRService,
    },
    models::graph::GraphData,
};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize environment
    dotenv().ok();
    env_logger::init();

    // Load settings
    let settings = Arc::new(RwLock::new(
        Settings::from_env().unwrap_or_else(|_| Settings::default())
    ));

    // Initialize services
    let github_service = Arc::new(RealGitHubService::new(
        settings.read().await.github.token.clone(),
        settings.read().await.github.owner.clone(),
        settings.read().await.github.repo.clone(),
        settings.read().await.github.base_path.clone(),
        settings.clone(),
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?);

    let github_pr_service = Arc::new(RealGitHubPRService::new(
        settings.read().await.github.token.clone(),
        settings.read().await.github.owner.clone(),
        settings.read().await.github.repo.clone(),
        settings.read().await.github.base_path.clone(),
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?);

    // Initialize GPU compute with empty graph
    let empty_graph = GraphData::default();
    let gpu_compute = match GPUCompute::new(&empty_graph).await {
        Ok(compute) => Some(compute),
        Err(e) => {
            error!("Failed to initialize GPU compute: {}", e);
            None
        }
    };

    // Create application state
    let state = AppState::new(
        settings.clone(),
        github_service,
        None, // perplexity_service
        None, // ragflow_service
        gpu_compute,
        String::new(), // some_string
        github_pr_service,
    ).await;

    let app_state = Arc::new(state);

    // Get static files path from environment or use default
    let static_files_path = std::env::var("STATIC_FILES_PATH")
        .unwrap_or_else(|_| "./static".to_string());

    // Configure and start server
    let static_path = static_files_path.to_string();
    let server = HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .max_age(3600)
            )
            .wrap(middleware::Logger::default())
            .service(
                web::scope("/api")
                    .route("/files/fetch", web::post().to(fetch_and_process_files))
                    .route("/files/content/{file_name}", web::get().to(get_file_content))
                    .route("/files/refresh", web::post().to(file_refresh_graph))
                    .route("/graph/data", web::get().to(get_graph_data))
                    .route("/graph/data/paginated", web::get().to(get_paginated_graph_data))
                    .route("/graph/refresh", web::post().to(graph_refresh_graph))
                    .service(
                        web::scope("/settings")
                            .configure(config)
                    )
            )
            .service(
                web::scope("/ws")
                    .route("", web::get().to(socket_flow_handler))
            )
            .service(Files::new("/", &static_path).index_file("index.html"))
    })
    .bind("0.0.0.0:8080")?
    .run();

    info!("Server running at http://localhost:8080");
    server.await
}

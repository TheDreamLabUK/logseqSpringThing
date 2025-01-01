use std::sync::Arc;
use tokio::sync::RwLock;
use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use log::{error, info};
use dotenvy::dotenv;

use webxr::{
    AppState,
    Settings,
    handlers::{
        file_handler::{fetch_and_process_files, get_file_content, refresh_graph as file_refresh_graph},
        graph_handler::{get_graph_data, get_paginated_graph_data, refresh_graph as graph_refresh_graph, update_graph},
        settings::{self, websocket, visualization},
        socket_flow_handler::socket_flow_handler,
    },
    utils::gpu_compute::GPUCompute,
    services::{
        file_service::FileService,
        github_service::RealGitHubPRService,
        graph_service::GraphService,
    },
    RealGitHubService,
};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize environment
    dotenv().ok();

    // Load settings
    let settings = Arc::new(RwLock::new(
        Settings::from_env().unwrap_or_else(|_| Settings::default())
    ));

    // Diagnostic: Print environment variables
    println!("GITHUB_TOKEN: {}", std::env::var("GITHUB_TOKEN").unwrap_or_else(|_| "Not found".to_string()));
    println!("GITHUB_OWNER: {}", std::env::var("GITHUB_OWNER").unwrap_or_else(|_| "Not found".to_string()));
    println!("GITHUB_REPO: {}", std::env::var("GITHUB_REPO").unwrap_or_else(|_| "Not found".to_string()));
    println!("GITHUB_BASE_PATH: {}", std::env::var("GITHUB_BASE_PATH").unwrap_or_else(|_| "Not found".to_string()));

    // Initialize services
    let settings_read = settings.read().await;
    let github_service = Arc::new(RealGitHubService::new(
        match settings_read.github.token.as_str() {
            "" => return Err(std::io::Error::new(std::io::ErrorKind::Other, "GITHUB_TOKEN not set")),
            token => token.to_string(),
        },
        match settings_read.github.owner.as_str() {
            "" => return Err(std::io::Error::new(std::io::ErrorKind::Other, "GITHUB_OWNER not set")),
            owner => owner.to_string(),
        },
        match settings_read.github.repo.as_str() {
            "" => return Err(std::io::Error::new(std::io::ErrorKind::Other, "GITHUB_REPO not set")),
            repo => repo.to_string(),
        },
        match settings_read.github.base_path.as_str() {
            "" => return Err(std::io::Error::new(std::io::ErrorKind::Other, "GITHUB_BASE_PATH not set")),
            base_path => base_path.to_string(),
        },
        settings.clone(),
    ).map_err(|e| {
        error!("Failed to initialize GitHub service: {}", e);
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })?);
    drop(settings_read);

    let settings_read = settings.read().await;
    let github_pr_service = Arc::new(RealGitHubPRService::new(
        settings_read.github.token.clone(),
        settings_read.github.owner.clone(),
        settings_read.github.repo.clone(),
        settings_read.github.base_path.clone(),
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?);
    drop(settings_read);

    // Initialize local storage and fetch initial data
    info!("Initializing local storage and fetching initial data");
    if let Err(e) = FileService::initialize_local_storage(&*github_service, settings.clone()).await {
        error!("Failed to initialize local storage: {}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
    }

    // Load metadata and build initial graph
    info!("Building initial graph from metadata");
    let metadata_store = FileService::load_or_create_metadata()
        .map_err(|e| {
            error!("Failed to load metadata: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;

    let initial_graph = GraphService::build_graph_from_metadata(&metadata_store)
        .await
        .map_err(|e| {
            error!("Failed to build initial graph: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;

    // Initialize GPU compute with initial graph
    let gpu_compute = match GPUCompute::create_for_app_state(&initial_graph).await {
        Ok(compute) => {
            info!("GPU compute initialized successfully");
            Some(compute)
        },
        Err(e) => {
            error!("Failed to initialize GPU compute: {}", e);
            None
        }
    };

    // Create application state with initial graph
    let state: AppState = AppState::new(
        settings.clone(),
        github_service.clone(),
        None, // perplexity_service
        None, // ragflow_service
        gpu_compute,
        String::new(), // some_string
        github_pr_service,
    ).await;

    let app_state = Arc::new(state);

    // Get static files path from environment or use default
    let static_files_path = std::env::var("STATIC_FILES_PATH")
        .unwrap_or_else(|_| "/app/static".to_string());

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
                    // Settings routes
                    .service(
                        web::scope("/settings")
                            .service(web::scope("/websocket").configure(websocket::config))
                            .service(web::scope("/visualization").configure(visualization::config))
                            .configure(settings::config)
                    )
                    // Graph routes
                    .service(
                        web::scope("/graph")
                            .route("/data", web::get().to(get_graph_data))
                            .route("/data/paginated", web::get().to(get_paginated_graph_data))
                            .route("/update", web::post().to(update_graph))
                            .route("/refresh", web::post().to(graph_refresh_graph))
                    )
                    // File routes
                    .service(
                        web::scope("/files")
                            .route("/fetch", web::post().to(fetch_and_process_files))
                            .route("/content/{file_name}", web::get().to(get_file_content))
                            .route("/refresh", web::post().to(file_refresh_graph))
                    )
            )
            .service(
                web::scope("/wss")
                    .route("", web::get().to(socket_flow_handler))
            )
            .service(Files::new("/", &static_path).index_file("index.html"))
    })
    .bind("0.0.0.0:3001")?
    .run();

    info!("Server running at http://localhost:3001");
    server.await
}

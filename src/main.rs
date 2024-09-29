use actix_files::Files;
use actix_web::{web, App, HttpServer, middleware};
use crate::handlers::{file_handler, graph_handler, ragflow_handler};
use crate::config::Settings;
use crate::app_state::AppState;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use crate::services::file_service::{GitHubService, RealGitHubService, FileService};
use crate::services::perplexity_service::PerplexityServiceImpl;
use crate::services::ragflow_service::RAGFlowService;
use crate::models::graph::GraphData;
use crate::utils::websocket_manager::WebSocketManager;
use crate::utils::gpu_compute::GPUCompute;
use crate::services::graph_service::GraphService;

mod app_state;
mod config;
mod handlers;
mod models;
mod services;
mod utils;

async fn initialize_graph_data(app_state: &web::Data<AppState>) -> std::io::Result<()> {
    log::info!("Initializing graph data...");
    
    match FileService::fetch_and_process_files(&*app_state.github_service, &app_state.settings).await {
        Ok(processed_files) => {
            log::info!("Successfully processed {} files", processed_files.len());

            // Update the file cache with processed content
            {
                let mut file_cache = app_state.file_cache.write().await;
                for processed_file in &processed_files {
                    file_cache.insert(processed_file.file_name.clone(), processed_file.content.clone());
                }
            }

            // Update graph data structure based on processed files
            match GraphService::build_graph(&app_state).await {
                Ok(graph_data) => {
                    let mut graph = app_state.graph_data.write().await;
                    *graph = graph_data;
                    log::info!("Graph data structure initialized successfully");
                    Ok(())
                },
                Err(e) => {
                    log::error!("Failed to build graph data: {}", e);
                    Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to build graph data: {}", e)))
                }
            }
        },
        Err(e) => {
            log::error!("Error processing files: {:?}", e);
            Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Error processing files: {:?}", e)))
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load environment variables
    dotenv::dotenv().ok();
    println!("Environment: {:?}", std::env::vars().collect::<Vec<_>>());

    // Set RUST_LOG to debug
    std::env::set_var("RUST_LOG", "debug");

    // Initialise logger
    env_logger::init();
    log::info!("Starting WebXR Graph Server");

    // Load settings
    let settings = match Settings::new() {
        Ok(s) => {
            log::debug!("Successfully loaded settings: {:?}", s);
            s
        },
        Err(e) => {
            log::error!("Failed to load settings: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to load settings: {:?}", e)));
        }
    };

    // Debug log for Perplexity API key
    log::debug!("Perplexity API Key: {}", settings.perplexity.perplexity_api_key);

    // Initialise shared application state
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let github_service: Arc<dyn GitHubService + Send + Sync> = Arc::new(RealGitHubService::new(settings.github.clone()));
    let perplexity_service = PerplexityServiceImpl::new();
    let ragflow_service = Arc::new(RAGFlowService::new(&settings));
    let websocket_manager = Arc::new(WebSocketManager::new());
    
    // Initialize GPUCompute
    let gpu_compute = match GPUCompute::new().await {
        Ok(gpu) => {
            log::info!("GPU initialization successful");
            Some(Arc::new(RwLock::new(gpu)))
        },
        Err(e) => {
            log::warn!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            None
        }
    };

    let app_state = web::Data::new(AppState::new(
        graph_data,
        file_cache,
        settings,
        github_service,
        perplexity_service,
        ragflow_service,
        websocket_manager,
        gpu_compute,
    ));

    // Initialize graph data
    initialize_graph_data(&app_state).await?;

    // Start HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(middleware::Logger::default())
            // Register API routes
            .service(
                web::scope("/api/files")
                    .route("/fetch", web::get().to(file_handler::fetch_and_process_files))
            )
            .service(
                web::scope("/api/graph")
                    .route("/data", web::get().to(graph_handler::get_graph_data))
            )
            .service(
                web::scope("/api/chat")
                    .route("/init", web::post().to(ragflow_handler::init_chat))
                    .route("/message/{conversation_id}", web::post().to(ragflow_handler::send_message))
                    .route("/history/{conversation_id}", web::get().to(ragflow_handler::get_chat_history))
            )
            // **Define the WebSocket route first**
            .route("/ws", web::get().to(WebSocketManager::handle_websocket))
            // **Then serve static files**
            .service(
                Files::new("/", "/app/data/public/dist").index_file("index.html")
            )
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
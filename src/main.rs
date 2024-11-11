use actix_files::Files;
use actix_web::{web, App, HttpServer, middleware, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::env;
use tokio::time::{interval, Duration};

use crate::app_state::AppState;
use crate::config::Settings;
use crate::handlers::{
    file_handler, 
    graph_handler, 
    ragflow_handler, 
    visualization_handler,
    perplexity_handler,
};
use crate::models::graph::GraphData;
use crate::services::file_service::{GitHubService, RealGitHubService, FileService};
use crate::services::perplexity_service::{PerplexityService, PerplexityServiceImpl};
use crate::services::ragflow_service::RAGFlowService;
use crate::services::speech_service::SpeechService;
use crate::services::graph_service::GraphService;
use crate::services::github_service::{GitHubPRService, RealGitHubPRService};
use crate::utils::websocket_manager::WebSocketManager;
use crate::utils::gpu_compute::GPUCompute;

mod app_state;
mod config;
mod handlers;
mod models;
mod services;
mod utils;

async fn initialize_graph_data(app_state: &web::Data<AppState>) -> std::io::Result<()> {
    log::info!("Starting graph data initialization...");
    
    let mut metadata_map = HashMap::new();
    log::info!("Fetching and processing files from GitHub...");
    match FileService::fetch_and_process_files(&*app_state.github_service, app_state.settings.clone(), &mut metadata_map).await {
        Ok(processed_files) => {
            log::info!("Successfully processed {} files", processed_files.len());
            log::debug!("Processed files: {:?}", processed_files.iter().map(|f| &f.file_name).collect::<Vec<_>>());

            let mut file_cache = app_state.file_cache.write().await;
            for processed_file in &processed_files {
                log::debug!("Caching file: {}", processed_file.file_name);
                file_cache.insert(processed_file.file_name.clone(), processed_file.content.clone());
            }
            drop(file_cache); // Explicitly drop the write lock

            log::info!("Building graph from processed files...");
            match GraphService::build_graph(&app_state).await {
                Ok(graph_data) => {
                    let mut graph = app_state.graph_data.write().await;
                    *graph = graph_data;
                    log::info!("Graph data structure initialized successfully");
                    log::debug!("Graph stats: {} nodes, {} edges", 
                        graph.nodes.len(), 
                        graph.edges.len()
                    );
                    Ok(())
                },
                Err(e) => {
                    log::error!("Failed to build graph data: {}", e);
                    log::error!("Error details: {:?}", e);
                    Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to build graph data: {}", e)))
                }
            }
        },
        Err(e) => {
            log::error!("Error processing files: {:?}", e);
            log::error!("Error details: {:?}", e);
            Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Error processing files: {:?}", e)))
        }
    }
}

async fn test_speech_service(app_state: web::Data<AppState>) -> HttpResponse {
    match app_state.speech_service.send_message("Hello, OpenAI!".to_string()).await {
        Ok(_) => HttpResponse::Ok().body("Message sent successfully"),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    }
}

// Simple health check endpoint that returns 200 OK when the service is running
async fn health_check() -> HttpResponse {
    HttpResponse::Ok().finish()
}

async fn randomize_nodes_periodically(app_state: web::Data<AppState>) {
    let mut interval = interval(Duration::from_secs(30));

    loop {
        interval.tick().await;
        
        log::debug!("Starting periodic graph rebuild...");
        // Recalculate graph data
        if let Err(e) = GraphService::build_graph(&app_state).await {
            log::error!("Failed to rebuild graph: {}", e);
            continue;
        }

        // Notify WebSocket clients about the updated graph data
        let graph_data = app_state.graph_data.read().await;
        if let Err(e) = app_state.websocket_manager.broadcast_graph_update(&graph_data).await {
            log::error!("Failed to broadcast graph update: {}", e);
        }
        log::debug!("Completed periodic graph rebuild");
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv::dotenv().ok();
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    log::info!("Starting WebXR Graph Server");

    log::info!("Loading settings...");
    let settings = match Settings::new() {
        Ok(s) => {
            log::info!("Successfully loaded settings");
            Arc::new(RwLock::new(s))
        },
        Err(e) => {
            log::error!("Failed to load settings: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize settings: {:?}", e)));
        }
    };

    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    
    log::info!("Initializing GitHub service...");
    let github_service: Arc<dyn GitHubService + Send + Sync> = {
        let settings_read = settings.read().await;
        match RealGitHubService::new(
            settings_read.github.github_access_token.clone(),
            settings_read.github.github_owner.clone(),
            settings_read.github.github_repo.clone(),
            settings_read.github.github_directory.clone(),
            settings.clone(),
        ) {
            Ok(service) => Arc::new(service),
            Err(e) => {
                log::error!("Failed to initialize GitHubService: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHubService: {:?}", e)));
            }
        }
    };

    log::info!("Initializing GitHub PR service...");
    let github_pr_service: Arc<dyn GitHubPRService + Send + Sync> = {
        let settings_read = settings.read().await;
        match RealGitHubPRService::new(
            settings_read.github.github_access_token.clone(),
            settings_read.github.github_owner.clone(),
            settings_read.github.github_repo.clone(),
            settings_read.github.github_directory.clone(),
        ) {
            Ok(service) => Arc::new(service),
            Err(e) => {
                log::error!("Failed to initialize GitHubPRService: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHubPRService: {:?}", e)));
            }
        }
    };
    
    let perplexity_service = Arc::new(PerplexityServiceImpl::new()) as Arc<dyn PerplexityService + Send + Sync>;
    
    log::info!("Initializing RAGFlow service...");
    let ragflow_service = match RAGFlowService::new(settings.clone()).await {
        Ok(service) => Arc::new(service),
        Err(e) => {
            log::error!("Failed to initialize RAGFlowService: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize RAGFlowService: {:?}", e)));
        }
    };

    // Create a single RAGFlow conversation
    log::info!("Creating RAGFlow conversation...");
    let ragflow_conversation_id = match ragflow_service.create_conversation("default_user".to_string()).await {
        Ok(id) => {
            log::info!("Created RAGFlow conversation with ID: {}", id);
            id
        },
        Err(e) => {
            log::error!("Failed to create RAGFlow conversation: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to create RAGFlow conversation: {:?}", e)));
        }
    };

    let websocket_manager = Arc::new(WebSocketManager::new());
    
    // Initialize with default graph data first
    log::info!("Initializing GPU compute...");
    let initial_graph_data = graph_data.read().await;
    let gpu_compute = match GPUCompute::new(&initial_graph_data).await {
        Ok(gpu) => {
            log::info!("GPU initialization successful");
            Some(Arc::new(RwLock::new(gpu)))
        },
        Err(e) => {
            log::warn!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            None
        }
    };
    drop(initial_graph_data); // Release the read lock

    log::info!("Initializing speech service...");
    let speech_service = Arc::new(SpeechService::new(websocket_manager.clone(), settings.clone()));
    if let Err(e) = speech_service.initialize().await {
        log::error!("Failed to initialize SpeechService: {:?}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize SpeechService: {:?}", e)));
    }

    let app_state = web::Data::new(AppState::new(
        graph_data,
        file_cache,
        settings.clone(),
        github_service,
        perplexity_service,
        ragflow_service.clone(),
        speech_service,
        websocket_manager.clone(),
        gpu_compute,
        ragflow_conversation_id,
        github_pr_service,
    ));

    log::info!("Initializing graph data...");
    if let Err(e) = initialize_graph_data(&app_state).await {
        log::error!("Failed to initialize graph data: {:?}", e);
        return Err(e);
    }

    log::info!("Initializing WebSocket manager...");
    if let Err(e) = websocket_manager.initialize(&ragflow_service).await {
        log::error!("Failed to initialize RAGflow conversation: {:?}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize RAGflow conversation: {:?}", e)));
    }

    // Spawn the randomization task
    let randomization_state = app_state.clone();
    tokio::spawn(async move {
        randomize_nodes_periodically(randomization_state).await;
    });

    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let bind_address = format!("0.0.0.0:{}", port);

    log::info!("Starting HTTP server on {}", bind_address);

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(middleware::Logger::default())
            .route("/health", web::get().to(health_check))
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
                    .route("/message", web::post().to(ragflow_handler::send_message))
                    .route("/history", web::get().to(ragflow_handler::get_chat_history))
            )
            .service(
                web::scope("/api/visualization")
                    .route("/settings", web::get().to(visualization_handler::get_visualization_settings))
            )
            .service(
                web::scope("/api/perplexity")
                    .route("/process", web::post().to(perplexity_handler::process_files))
            )
            .route("/ws", web::get().to(WebSocketManager::handle_websocket))
            .route("/test_speech", web::get().to(test_speech_service))
            .service(
                Files::new("/", "/app/data/public/dist").index_file("index.html")
            )
    })
    .bind(&bind_address)?
    .run()
    .await
}

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

/// Initialize graph data from cached metadata
/// This is called at startup to quickly get the graph running before GitHub updates
async fn initialize_cached_graph_data(app_state: &web::Data<AppState>) -> std::io::Result<()> {
    log::info!("Loading cached graph data...");
    
    // Load existing metadata from disk
    let metadata_map = match FileService::load_or_create_metadata() {
        Ok(map) => {
            log::info!("Loaded existing metadata with {} entries", map.len());
            map
        },
        Err(e) => {
            log::error!("Failed to load metadata: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to load metadata: {}", e)));
        }
    };

    // Build initial graph from cached metadata
    log::info!("Building graph from cached metadata...");
    match GraphService::build_graph_from_metadata(&metadata_map).await {
        Ok(graph_data) => {
            let mut graph = app_state.graph_data.write().await;
            *graph = graph_data;
            log::info!("Graph initialized from cache with {} nodes and {} edges", 
                graph.nodes.len(), 
                graph.edges.len()
            );
            Ok(())
        },
        Err(e) => {
            log::error!("Failed to build graph from cache: {}", e);
            Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        }
    }
}

/// Periodic graph update function
/// Checks for GitHub updates every 12 hours while preserving node positions
async fn update_graph_periodically(app_state: web::Data<AppState>) {
    let mut interval = interval(Duration::from_secs(43200)); // 12 hour interval

    loop {
        interval.tick().await;
        
        log::debug!("Starting periodic graph update...");
        
        // Load current metadata
        let mut metadata_map = match FileService::load_or_create_metadata() {
            Ok(map) => map,
            Err(e) => {
                log::error!("Failed to load metadata: {}", e);
                continue;
            }
        };

        // Check for GitHub updates
        match FileService::fetch_and_process_files(&*app_state.github_service, app_state.settings.clone(), &mut metadata_map).await {
            Ok(processed_files) => {
                if !processed_files.is_empty() {
                    log::info!("Found {} updated files, updating graph", processed_files.len());

                    // Update file cache with new/modified files
                    let mut file_cache = app_state.file_cache.write().await;
                    for processed_file in &processed_files {
                        file_cache.insert(processed_file.file_name.clone(), processed_file.content.clone());
                    }
                    drop(file_cache);

                    // Update graph while preserving node positions
                    let mut graph = app_state.graph_data.write().await;
                    let old_positions: HashMap<String, (f32, f32, f32)> = graph.nodes.iter()
                        .map(|node| (node.id.clone(), (node.x, node.y, node.z)))
                        .collect();
                    
                    // Update metadata
                    graph.metadata = metadata_map.clone();

                    // Build new graph preserving positions
                    if let Ok(mut new_graph) = GraphService::build_graph_from_metadata(&metadata_map).await {
                        // Preserve positions for existing nodes
                        for node in &mut new_graph.nodes {
                            if let Some(&(x, y, z)) = old_positions.get(&node.id) {
                                node.x = x;
                                node.y = y;
                                node.z = z;
                            }
                        }
                        *graph = new_graph.clone();
                        drop(graph); // Release the write lock before broadcasting

                        // Notify clients of the update
                        if let Err(e) = app_state.websocket_manager.broadcast_graph_update(&new_graph).await {
                            log::error!("Failed to broadcast graph update: {}", e);
                        }
                    }
                } else {
                    log::debug!("No updates found");
                }
            },
            Err(e) => log::error!("Failed to check for updates: {}", e)
        }

        log::debug!("Completed periodic graph update");
    }
}

/// Simple health check endpoint
async fn health_check() -> HttpResponse {
    HttpResponse::Ok().finish()
}

/// Test endpoint for speech service
async fn test_speech_service(app_state: web::Data<AppState>) -> HttpResponse {
    match app_state.speech_service.send_message("Hello, OpenAI!".to_string()).await {
        Ok(_) => HttpResponse::Ok().body("Message sent successfully"),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load environment variables
    if let Ok(vars) = envy::from_env::<HashMap<String, String>>() {
        for (key, value) in vars {
            env::set_var(key, value);
        }
    }

    // Initialize logging
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    log::info!("Starting WebXR Graph Server");

    // Load configuration
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

    // Initialize core data structures
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    
    // Initialize GitHub service
    log::info!("Initializing GitHub service...");
    let github_service: Arc<dyn GitHubService + Send + Sync> = {
        let settings_read = settings.read().await;
        match RealGitHubService::new(
            settings_read.github.access_token.clone(),
            settings_read.github.owner.clone(),
            settings_read.github.repo.clone(),
            settings_read.github.directory.clone(),
            settings.clone(),
        ) {
            Ok(service) => Arc::new(service),
            Err(e) => {
                log::error!("Failed to initialize GitHubService: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHubService: {:?}", e)));
            }
        }
    };

    // Initialize GitHub PR service
    log::info!("Initializing GitHub PR service...");
    let github_pr_service: Arc<dyn GitHubPRService + Send + Sync> = {
        let settings_read = settings.read().await;
        match RealGitHubPRService::new(
            settings_read.github.access_token.clone(),
            settings_read.github.owner.clone(),
            settings_read.github.repo.clone(),
            settings_read.github.directory.clone(),
        ) {
            Ok(service) => Arc::new(service),
            Err(e) => {
                log::error!("Failed to initialize GitHubPRService: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHubPRService: {:?}", e)));
            }
        }
    };
    
    // Initialize services
    let perplexity_service = Arc::new(PerplexityServiceImpl::new()) as Arc<dyn PerplexityService + Send + Sync>;
    
    log::info!("Initializing RAGFlow service...");
    let ragflow_service = match RAGFlowService::new(settings.clone()).await {
        Ok(service) => Arc::new(service),
        Err(e) => {
            log::error!("Failed to initialize RAGFlowService: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize RAGFlowService: {:?}", e)));
        }
    };

    // Create RAGFlow conversation
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
    let websocket_data = web::Data::new(websocket_manager.clone());
    
    // Initialize GPU compute with default graph
    log::info!("Initializing GPU compute...");
    let initial_graph_data = graph_data.read().await;
    let gpu_compute = match GPUCompute::new(&initial_graph_data).await {
        Ok(gpu) => {
            log::info!("GPU initialization successful with {} nodes", initial_graph_data.nodes.len());
            Some(Arc::new(RwLock::new(gpu)))
        },
        Err(e) => {
            log::warn!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            None
        }
    };
    drop(initial_graph_data);

    // Initialize speech service
    log::info!("Initializing speech service...");
    let speech_service = Arc::new(SpeechService::new(websocket_manager.clone(), settings.clone()));
    if let Err(e) = speech_service.initialize().await {
        log::error!("Failed to initialize SpeechService: {:?}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize SpeechService: {:?}", e)));
    }

    // Create application state
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

    // Initialize graph from cache for fast startup
    log::info!("Initializing graph with cached data...");
    if let Err(e) = initialize_cached_graph_data(&app_state).await {
        log::warn!("Failed to initialize from cache: {:?}, proceeding with empty graph", e);
    }

    // Start periodic update task
    let update_state = app_state.clone();
    tokio::spawn(async move {
        update_graph_periodically(update_state).await;
    });

    // Start HTTP server
    let port = env::var("PORT").unwrap_or_else(|_| "4000".to_string());
    let bind_address = format!("0.0.0.0:{}", port);
    log::info!("Starting HTTP server on {}", bind_address);

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(websocket_data.clone()) // Add WebSocketManager as app data
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

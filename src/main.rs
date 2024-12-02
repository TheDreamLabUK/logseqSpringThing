use actix_files::Files;
use actix_web::{web, App, HttpServer, middleware, HttpResponse, HttpRequest};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use tokio::time::{interval, Duration};
use dotenv::dotenv;

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
use crate::services::perplexity_service::PerplexityService;
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

    // Store metadata in app state
    {
        let mut app_metadata = app_state.metadata.write().await;
        *app_metadata = metadata_map.clone();
    }

    // Build initial graph from cached metadata
    log::info!("Building graph from cached metadata...");
    match GraphService::build_graph_from_metadata(&metadata_map).await {
        Ok(graph_data) => {
            let mut graph = app_state.graph_service.graph_data.write().await;
            *graph = graph_data.clone();
            
            // Ensure metadata is stored in graph_data
            graph.metadata = metadata_map;
            
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

                    // Update metadata in app state
                    {
                        let mut app_metadata = app_state.metadata.write().await;
                        *app_metadata = metadata_map.clone();
                    }

                    // Update graph while preserving node positions
                    let mut graph = app_state.graph_service.graph_data.write().await;
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
                                node.position = Some([x, y, z]);
                            }
                        }
                        *graph = new_graph.clone();
                        drop(graph); // Release the write lock before broadcasting

                        // Only broadcast if websocket manager is initialized
                        if let Some(ws_manager) = app_state.get_websocket_manager().await {
                            if let Err(e) = ws_manager.broadcast_graph_update(&new_graph).await {
                                log::error!("Failed to broadcast graph update: {}", e);
                            }
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
    // Load .env file first
    dotenv().ok();

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

    // Initialize services
    log::info!("Initializing services...");
    let github_service = {
        let settings_read = settings.read().await;
        Arc::new(RealGitHubService::new(
            settings_read.github.token.clone(),
            settings_read.github.owner.clone(),
            settings_read.github.repo.clone(),
            settings_read.github.base_path.clone(),
            settings.clone(),
        )?)
    };

    let github_pr_service = {
        let settings_read = settings.read().await;
        Arc::new(RealGitHubPRService::new(
            settings_read.github.token.clone(),
            settings_read.github.owner.clone(),
            settings_read.github.repo.clone(),
            settings_read.github.base_path.clone(),
        )?)
    };

    let perplexity_service = Arc::new(PerplexityService::new(settings.clone())?);
    let ragflow_service = Arc::new(RAGFlowService::new(settings.clone()).await?);
    let speech_service = Arc::new(SpeechService::new(settings.clone()));

    // Create RAGFlow conversation
    log::info!("Creating RAGFlow conversation...");
    let ragflow_conversation_id = ragflow_service.create_conversation("default_user".to_string()).await?;
    
    // Initialize GPU compute with default graph
    log::info!("Initializing GPU compute...");
    let gpu_compute = match GPUCompute::new(&GraphData::default()).await {
        Ok(gpu) => {
            log::info!("GPU initialization successful");
            Some(Arc::new(RwLock::new(gpu)))
        },
        Err(e) => {
            log::warn!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            None
        }
    };

    // Create application state without websocket manager
    let app_state = web::Data::new(AppState::new(
        settings.clone(),
        github_service,
        perplexity_service,
        ragflow_service,
        speech_service,
        gpu_compute,
        ragflow_conversation_id,
        github_pr_service,
    ));

    // Initialize graph from cache for fast startup
    log::info!("Initializing graph with cached data...");
    if let Err(e) = initialize_cached_graph_data(&app_state).await {
        log::warn!("Failed to initialize from cache: {:?}, proceeding with empty graph", e);
    }

    // Initialize WebSocket manager with debug settings and app_state after graph is initialized
    let websocket_manager = {
        let settings_read = settings.read().await;
        Arc::new(WebSocketManager::new(
            settings_read.debug.enable_websocket_debug,
            app_state.clone(),
        ))
    };

    // Set websocket manager in app state
    app_state.set_websocket_manager(websocket_manager.clone()).await;
    let websocket_data = web::Data::new(websocket_manager.clone());

    // Start periodic update task
    let update_state = app_state.clone();
    tokio::spawn(async move {
        update_graph_periodically(update_state).await;
    });

    // Start HTTP server
    let bind_address = "0.0.0.0:3000";
    log::info!("Starting HTTP server on {}", bind_address);

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(websocket_data.clone())
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
                    .service(perplexity_handler::handle_perplexity)
            )
            .route("/ws", web::get().to(|req: HttpRequest, stream: web::Payload, websocket_manager: web::Data<Arc<WebSocketManager>>| WebSocketManager::handle_websocket(req, stream, websocket_manager)))
            .route("/test_speech", web::get().to(test_speech_service))
            .service(
                Files::new("/", "/app/data/public/dist").index_file("index.html")
            )
    })
    .bind(bind_address)?
    .run()
    .await
}

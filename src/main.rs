use actix_files::Files;
use actix_web::{web, App, HttpServer, middleware, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use tokio::time::{interval, Duration};
use dotenvy::dotenv;
use std::error::Error;
use cudarc::driver::DriverError;
use log::{error, warn, debug};

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
use crate::services::file_service::{RealGitHubService, FileService};
use crate::services::perplexity_service::PerplexityService;
use crate::services::ragflow_service::{RAGFlowService, RAGFlowError};
use crate::services::graph_service::GraphService;
use crate::services::github_service::RealGitHubPRService;
use crate::utils::socket_flow_handler::ws_handler;
use crate::utils::gpu_compute::GPUCompute;
use crate::utils::debug_logging::init_debug_settings;

mod app_state;
mod config;
mod handlers;
mod models;
mod services;
mod utils;

#[derive(Debug)]
pub struct AppError(Box<dyn Error + Send + Sync>);

impl From<Box<dyn Error + Send + Sync>> for AppError {
    fn from(err: Box<dyn Error + Send + Sync>) -> Self {
        AppError(err)
    }
}

impl From<RAGFlowError> for AppError {
    fn from(err: RAGFlowError) -> Self {
        AppError(Box::new(err))
    }
}

impl From<AppError> for std::io::Error {
    fn from(err: AppError) -> Self {
        if let Some(io_err) = err.0.downcast_ref::<std::io::Error>() {
            std::io::Error::new(io_err.kind(), io_err.to_string())
        } else if let Some(driver_err) = err.0.downcast_ref::<DriverError>() {
            std::io::Error::new(std::io::ErrorKind::Other, driver_err.to_string())
        } else {
            std::io::Error::new(std::io::ErrorKind::Other, err.0.to_string())
        }
    }
}

fn to_io_error(e: impl std::fmt::Display) -> Box<dyn Error + Send + Sync> {
    Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
}

async fn initialize_cached_graph_data(app_state: &web::Data<AppState>) -> Result<(), Box<dyn Error + Send + Sync>> {
    let metadata_map = FileService::load_or_create_metadata()
        .map_err(|e| {
            log_error!("Failed to load metadata: {}", e);
            to_io_error(e)
        })?;

    log_data!("Loaded metadata with {} entries", metadata_map.len());
    
    {
        let mut app_metadata = app_state.metadata.write().await;
        *app_metadata = metadata_map.clone();
    }

    log_data!("Building graph from metadata...");
    let graph_data = GraphService::build_graph_from_metadata(&metadata_map).await
        .map_err(|e| {
            log_error!("Failed to build graph from metadata: {}", e);
            to_io_error(e)
        })?;

    {
        let mut graph = app_state.graph_service.graph_data.write().await;
        *graph = graph_data.clone();
        graph.metadata = metadata_map;
        
        log_data!("Graph initialized with {} nodes and {} edges", 
            graph.nodes.len(), 
            graph.edges.len()
        );
    }

    Ok(())
}

async fn update_graph_periodically(app_state: web::Data<AppState>) {
    let mut interval = interval(Duration::from_secs(43200));

    loop {
        interval.tick().await;
        
        let mut metadata_map = match FileService::load_or_create_metadata() {
            Ok(map) => map,
            Err(e) => {
                log_error!("Failed to load metadata: {}", e);
                continue;
            }
        };

        match FileService::fetch_and_process_files(&*app_state.github_service, app_state.settings.clone(), &mut metadata_map).await {
            Ok(processed_files) => {
                if !processed_files.is_empty() {
                    log_data!("Found {} updated files, updating graph", processed_files.len());

                    {
                        let mut app_metadata = app_state.metadata.write().await;
                        *app_metadata = metadata_map.clone();
                    }

                    let mut graph = app_state.graph_service.graph_data.write().await;
                    let old_positions: HashMap<String, (f32, f32, f32)> = graph.nodes.iter()
                        .map(|node| (node.id.clone(), (node.x, node.y, node.z)))
                        .collect();
                    
                    graph.metadata = metadata_map.clone();

                    if let Ok(mut new_graph) = GraphService::build_graph_from_metadata(&metadata_map).await {
                        for node in &mut new_graph.nodes {
                            if let Some(&(x, y, z)) = old_positions.get(&node.id) {
                                node.x = x;
                                node.y = y;
                                node.z = z;
                                node.position = Some([x, y, z]);
                            }
                        }
                        *graph = new_graph.clone();
                    }
                }
            },
            Err(e) => log_error!("Failed to check for updates: {}", e)
        }
    }
}

async fn health_check() -> HttpResponse {
    HttpResponse::Ok().finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();

    // Load settings first to get the log level
    let settings = match Settings::new() {
        Ok(s) => Arc::new(RwLock::new(s)),
        Err(e) => {
            eprintln!("Failed to load settings: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize settings: {:?}", e)));
        }
    };

    // Set log level from settings and initialize debug settings
    let (log_level, debug_enabled, websocket_debug, data_debug) = {
        let settings_read = settings.read().await;
        (
            settings_read.default.log_level.clone(),
            settings_read.server_debug.enabled,
            settings_read.server_debug.enable_websocket_debug,
            settings_read.server_debug.enable_data_debug
        )
    };
    
    std::env::set_var("RUST_LOG", log_level);
    env_logger::init();
    
    // Initialize our debug logging system
    init_debug_settings(debug_enabled, websocket_debug, data_debug);

    log_data!("Initializing services...");
    
    let settings_read = settings.read().await;
    let github_service = {
        Arc::new(RealGitHubService::new(
            settings_read.github.token.clone(),
            settings_read.github.owner.clone(),
            settings_read.github.repo.clone(),
            settings_read.github.base_path.clone(),
            settings.clone(),
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?)
    };

    let github_pr_service = {
        Arc::new(RealGitHubPRService::new(
            settings_read.github.token.clone(),
            settings_read.github.owner.clone(),
            settings_read.github.repo.clone(),
            settings_read.github.base_path.clone(),
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?)
    };

    let perplexity_service = Arc::new(PerplexityService::new(settings.clone())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?);
    
    let ragflow_service = Arc::new(RAGFlowService::new(settings.clone()).await
        .map_err(AppError::from)?);

    log_data!("Creating RAGFlow conversation...");
    let ragflow_conversation_id = ragflow_service.create_conversation("default_user".to_string()).await
        .map_err(AppError::from)?;

    log_data!("Initializing GPU compute...");
    let gpu_compute = match GPUCompute::new(&GraphData::default()).await {
       Ok(gpu) => {
            log_data!("GPU initialization successful");
            Some(gpu)
        },
        Err(e) => {
            log_warn!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            None
        }
    };

    let app_state = web::Data::new(AppState::new(
        settings.clone(),
        github_service,
        perplexity_service,
        ragflow_service,
        None,
        gpu_compute,
        ragflow_conversation_id,
        github_pr_service,
    ));

    log_data!("Initializing graph with cached data...");
    if let Err(e) = initialize_cached_graph_data(&app_state).await {
        log_error!("Failed to initialize graph from cache: {}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize graph: {}", e)));
    }

    let update_state = app_state.clone();
    let update_handle = tokio::spawn(async move {
        update_graph_periodically(update_state).await;
    });

    let bind_address = "0.0.0.0:3000";
    log_data!("Starting HTTP server on {}", bind_address);

    let server = HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(middleware::Logger::default())
            .route("/health", web::get().to(health_check))
            .route("/wss", web::get().to(ws_handler))  // WebSocket handler
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
            .service(
                Files::new("/", "/app/data/public/dist").index_file("index.html")
            )
    })
    .bind(bind_address)?
    .run();

    // Run both servers and handle shutdown
    tokio::select! {
        _ = server => {
            log_data!("HTTP server stopped");
        }
        _ = update_handle => {
            log_data!("Update task stopped");
        }
    }

    Ok(())
}

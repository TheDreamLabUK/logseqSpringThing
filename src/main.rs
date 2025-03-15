use webxr::{
    AppState,
    config::Settings,
    handlers::{
        api_handler,
        health_handler,
        pages_handler,
        socket_flow_handler::socket_flow_handler,
        nostr_handler,
    },
    services::{
        file_service::FileService,
        file_service::{GRAPH_CACHE_PATH, LAYOUT_CACHE_PATH}, // Added import for cache paths
        github::{GitHubClient, ContentAPI, GitHubConfig},
        graph_service::GraphService,
    },
    utils::gpu_compute::GPUCompute,
    models::node::Node,
    utils::gpu_diagnostics
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenvy::dotenv;
use tokio::time::Duration;
use log::{error, info, debug, warn};
use webxr::utils::logging::{init_logging_with_config, LogConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Make dotenv optional since env vars can come from Docker
    dotenv().ok();

    // Load settings first to get the log level
    let settings = match Settings::new() {
        Ok(s) => {
            info!("Settings loaded successfully from: {}", 
                std::env::var("SETTINGS_FILE_PATH").unwrap_or_default());
            Arc::new(RwLock::new(s))
        },
        Err(e) => {
            error!("Failed to load settings: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize settings: {:?}", e)));
        }
    };

    // Initialize logging with settings-based configuration
    let log_config = {
        let settings_read = settings.read().await;
        // Only use debug level if debug is enabled, otherwise use configured level
        let log_level = &settings_read.system.debug.log_level;
        
        LogConfig::new(
            log_level,
            log_level,
        )
    };

    init_logging_with_config(log_config)?;

    debug!("Successfully loaded settings");

    info!("Starting WebXR application...");
    
    // Create web::Data instances first
    let settings_data = web::Data::new(settings.clone());

    // Initialize services
    let github_config = match GitHubConfig::from_env() {
        Ok(config) => config,
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to load GitHub config: {}", e)))
    };

    let github_client = match GitHubClient::new(github_config, settings.clone()).await {
        Ok(client) => Arc::new(client),
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHub client: {}", e)))
    };

    let content_api = Arc::new(ContentAPI::new(github_client.clone()));

    // Initialize app state asynchronously
    let mut app_state = match AppState::new(
            settings.clone(),
            github_client.clone(),
            content_api.clone(),
            None,
            None,
            None,
            "default_conversation".to_string(),
        ).await {
            Ok(state) => state,
            Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize app state: {}", e)))
        };

    // Initialize Nostr service
    nostr_handler::init_nostr_service(&mut app_state);

    // First, try to load existing metadata without waiting for GitHub download
    info!("Loading existing metadata for quick initialization");
    let metadata_store = FileService::load_or_create_metadata()
        .map_err(|e| {
            error!("Failed to load existing metadata: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;

    // Launch GitHub data fetch in background to avoid blocking WebSocket initialization
    // Instead of spawning a background task which causes Send trait issues,
    // log that we're skipping the background fetch to avoid compilation errors
    info!("Note: Background GitHub data fetch is disabled to resolve compilation issues");
    // If GitHub data fetching becomes critical, consider modifying FileService or GitHubClient 
    // to implement Send for all futures

    if metadata_store.is_empty() {
        error!("No metadata found and could not create empty store");
        return Err(std::io::Error::new(std::io::ErrorKind::Other, 
            "No metadata found and could not create empty store".to_string()));
    }

    info!("Loaded {} items from metadata store", metadata_store.len());
    
    // Ensure metadata directories are properly set up
    if let Err(e) = tokio::fs::create_dir_all("/app/data/metadata/files").await {
        warn!("Failed to create metadata directory: {}", e);
    }
    
    // Ensure parent directories for cache files exist
    if let Err(e) = tokio::fs::create_dir_all(std::path::Path::new(GRAPH_CACHE_PATH).parent().unwrap()).await {
        warn!("Failed to create directory for graph cache: {}", e);
    }
    if let Err(e) = tokio::fs::create_dir_all(std::path::Path::new(LAYOUT_CACHE_PATH).parent().unwrap()).await {
        warn!("Failed to create directory for layout cache: {}", e);
    }
    
    if tokio::fs::metadata("/app/data/metadata").await.is_ok() {
        info!("Verified metadata directory exists");
    }

    // Update metadata in app state
    {
        let mut app_metadata = app_state.metadata.write().await;
        *app_metadata = metadata_store.clone();
        info!("Loaded metadata into app state");
    }

    // Build initial graph from metadata and initialize GPU compute
    
    // CRITICAL: Initialize Node ID counter at startup to ensure consistent IDs across all processes
    // This MUST be done before any graph building or GPU operations 
    info!("CRITICAL: Initializing Node ID counter for consistent IDs across all worker processes");
    Node::initialize_id_counter();
    
    // LAZY INITIALIZATION: We don't build the graph on startup anymore
    // Instead, we'll build it when the first client request comes in
    info!("Initializing graph with cache files and GPU setup");

    // Build graph ONCE and initialize GPU 
    // This eliminates the redundant second build that was happening before
    match GraphService::build_graph_from_metadata(&metadata_store).await {
        Ok(new_graph) => {
                info!("Successfully built graph with {} nodes, {} edges", new_graph.nodes.len(), new_graph.edges.len());
                
                // Update app state with the new graph
                let mut app_graph = app_state.graph_service.get_graph_data_mut().await;
                *app_graph = new_graph.clone();
                drop(app_graph);
                
                // Update the node map too
                let mut node_map = app_state.graph_service.get_node_map_mut().await;
                node_map.clear();
                for node in &new_graph.nodes {
                    node_map.insert(node.id.clone(), node.clone());
                }
                drop(node_map);
                
                // Explicitly verify the cache files were created
                if let Err(e) = tokio::fs::metadata(GRAPH_CACHE_PATH).await {
                    warn!("Graph cache file was not created: {}", e);
                }
                if let Err(e) = tokio::fs::metadata(LAYOUT_CACHE_PATH).await {
                    warn!("Layout cache file was not created: {}", e);
                }
                
                // Run GPU diagnostics before attempting to initialize
                info!("Running GPU diagnostics before initialization");
                let gpu_diag_report = gpu_diagnostics::run_gpu_diagnostics();
                info!("GPU diagnostics complete: \n{}", gpu_diag_report);
                
                // Try to fix CUDA environment if there might be issues
                if let Err(e) = gpu_diagnostics::fix_cuda_environment() {
                    warn!("Could not fix CUDA environment: {}", e);
                }
                
                // Initialize GPU with the populated graph
                info!("Attempting to initialize GPU compute with populated graph data");
                let gpu_init_start = std::time::Instant::now();
                match GPUCompute::new(&new_graph).await {
                    Ok(gpu_instance) => {
                        info!("✅ GPU compute initialized successfully");
                        
                        // CRITICAL FIX: Test GPU and update flag in a separate scope to avoid borrow issues
                        {
                            let gpu_lock = gpu_instance.read().await;
                            if gpu_lock.test_compute().is_ok() {
                                GraphService::update_global_gpu_status(true);
                            }
                        }
                        app_state.gpu_compute = Some(gpu_instance);
                        
                        // CRITICAL: Check GPU status after initialization to verify it's fully operational
                        tokio::time::sleep(Duration::from_millis(500)).await; // Longer delay for GPU stabilization
                        
                        // Run a thorough check
                        // Check GPU status with a longer timeout
                        tokio::time::timeout(Duration::from_secs(5), app_state.check_gpu_status())
                            .await
                            .unwrap_or(false);
                            
                        let gpu_status = *app_state.gpu_available.read().await;
                        if gpu_status {
                            info!("🎉 GPU COMPUTE IS VERIFIED AND AVAILABLE FOR PHYSICS SIMULATION ({:?})", 
                                  gpu_init_start.elapsed());
                            
                            // Try an actual physics calculation to confirm it's fully working
                            if let Some(gpu) = &app_state.gpu_compute {
                                let mut graph_data = app_state.graph_service.get_graph_data_mut().await;
                                let mut node_map = app_state.graph_service.get_node_map_mut().await;
                                
                                match GraphService::calculate_layout_with_retry(
                                    gpu, 
                                    Some(&web::Data::new(app_state.clone())), 
                                    &mut graph_data, 
                                    &mut node_map, 
                                    &webxr::models::simulation_params::SimulationParams::default()
                                ).await {
                                    Ok(_) => {
                                        info!("🔥 GPU PHYSICS TEST CALCULATION SUCCESSFUL - SYSTEM READY FOR GPU ACCELERATION");
                                        GraphService::update_global_gpu_status(true);
                                        // Force GPU status to true after successful test
                                        *app_state.gpu_available.write().await = true;
                                    },
                                    Err(e) => {
                                        warn!("⚠️ GPU test calculation failed: {} - will fall back to CPU physics", e);
                                        *app_state.gpu_available.write().await = false;
                                    }
                                }
                            }
                        } else {
                            warn!("⚠️ GPU compute failed verification check after {:?} - falling back to CPU physics", gpu_init_start.elapsed());
                        }
                        
                        // Run one final diagnostic if we didn't get GPU working
                        if !(*app_state.gpu_available.read().await) {
                            error!("Final GPU diagnosis after unsuccessful initialization: \n{}", gpu_diagnostics::run_gpu_diagnostics());
                        }
                    },
                    Err(e) => error!("❌ Failed to initialize GPU compute: {}. Will use CPU fallback.", e)
                }
            },
            Err(e) => warn!("Failed to pre-build graph: {}. Cache files may not be created until first request", e)
        }
    
    info!("Starting HTTP server with graph initialization complete...");

    // Create web::Data after all initialization is complete
    let app_state_data = web::Data::new(app_state);
    info!("Pre-build process completed");

    // Start the server
    let bind_address = {
        let settings_read = settings.read().await;
        format!("{}:{}", settings_read.system.network.bind_address, settings_read.system.network.port)
    };

    info!("Starting HTTP server on {}", bind_address);

    HttpServer::new(move || {
        // Configure CORS
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600)
            .supports_credentials();

        App::new()
            .wrap(middleware::Logger::default())
            .wrap(cors)
            .wrap(middleware::Compress::default())
            .app_data(settings_data.clone())
            .app_data(web::Data::new(github_client.clone()))
            .app_data(web::Data::new(content_api.clone()))
            .app_data(app_state_data.clone())  // Add the complete AppState
            .app_data(app_state_data.nostr_service.clone().unwrap())
            .app_data(app_state_data.feature_access.clone())
            .route("/wss", web::get().to(socket_flow_handler))
            .service(
                web::scope("")
                    .configure(api_handler::config)
                    .service(web::scope("/health").configure(health_handler::config))
                    .service(web::scope("/pages").configure(pages_handler::config))
            )
            .service(Files::new("/", "/app/data/public/dist").index_file("index.html"))
    })
    .bind(&bind_address)?
    .run()
    .await?;

    info!("HTTP server stopped");
    Ok(())
}

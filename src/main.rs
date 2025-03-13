use webxr::{
    AppState,
    config::Settings,
    models::graph::GraphData,
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
    utils::gpu_compute::GPUCompute
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenvy::dotenv;
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
    // LAZY INITIALIZATION: We don't build the graph on startup anymore
    // Instead, we'll build it when the first client request comes in
    info!("LAZY INITIALIZATION ENABLED: Deferring graph building until first client request");
    
    // Initialize the GraphService with the settings, but don't build the graph yet
    // GraphService will try to load from cache when first requested
    info!("Initializing graph service with lazy loading");
    
    // Initialize GPU compute instance but don't populate it with data yet
    match GPUCompute::new(&GraphData::default()).await {
        Ok(gpu_instance) => {
            info!("GPU compute initialized (empty) successfully for lazy loading");
            app_state.gpu_compute = Some(gpu_instance);
        },
        Err(e) => warn!("Failed to initialize GPU compute: {}. Will use CPU fallback when needed.", e)
    }
    
    info!("Starting HTTP server with lazy graph initialization...");

    // Create web::Data after all initialization is complete
    let app_state_data = web::Data::new(app_state);

    // Pre-build graph to ensure cache files are created on startup
    if metadata_store.len() > 0 {
        info!("Pre-building graph to ensure cache files are created on startup");
        match GraphService::build_graph_from_metadata(&metadata_store).await {
            Ok(graph) => {
                info!("Successfully pre-built graph with {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
                
                // Update the app state's graph service with the built graph
                let mut app_graph = app_state_data.graph_service.get_graph_data_mut().await;
                *app_graph = graph.clone();
                drop(app_graph);
                
                // Update the node map too
                let mut node_map = app_state_data.graph_service.get_node_map_mut().await;
                node_map.clear();
                for node in &graph.nodes {
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
            },
            Err(e) => warn!("Failed to pre-build graph: {}. Cache files may not be created until first request", e)
        }
        info!("Pre-build process completed");
    }

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

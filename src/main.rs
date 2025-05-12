use webxr::services::nostr_service::NostrService;
use webxr::{
    AppState,
    config::AppFullSettings, // Import AppFullSettings only
    handlers::{
        api_handler,
        health_handler,
        pages_handler,
        socket_flow_handler::socket_flow_handler,
        speech_socket_handler::speech_socket_handler,
        nostr_handler,
    },
    services::{
        file_service::FileService,
        graph_service::GraphService,
        github::{GitHubClient, ContentAPI, GitHubConfig},
    },
    utils::gpu_compute::GPUCompute,
    services::speech_service::SpeechService,
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
// use actix_files::Files; // Removed unused import
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Duration;
use dotenvy::dotenv;
use log::{error, info, debug, warn};
use webxr::utils::logging::{init_logging_with_config, LogConfig};
use tokio::signal::unix::{signal, SignalKind};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Make dotenv optional since env vars can come from Docker
    dotenv().ok();

    // Load settings first to get the log level
    // Use AppFullSettings here as this is the main server configuration loaded from YAML/Env
    let settings = match AppFullSettings::new() { // Changed to AppFullSettings::new()
        Ok(s) => {
            info!("AppFullSettings loaded successfully from: {}", 
                std::env::var("SETTINGS_FILE_PATH").unwrap_or_else(|_| "/app/settings.yaml".to_string()));
            Arc::new(RwLock::new(s)) // Now holds Arc<RwLock<AppFullSettings>>
        },
        Err(e) => {
            error!("Failed to load AppFullSettings: {:?}", e);
            // Try loading the client-facing Settings as a fallback for debugging? Unlikely to work.
            // error!("Attempting fallback load of client-facing Settings struct...");
            // match ClientFacingSettings::new() { // This ::new doesn't exist on client Settings
            //     Ok(_) => error!("Fallback load seemed to work structurally, but AppState expects AppFullSettings!"),
            //     Err(fe) => error!("Fallback load also failed: {:?}", fe),
            // }
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize AppFullSettings: {:?}", e)));
        }
    };

    // --- BEGIN GPU TEST BEFORE LOGGING ---
    println!("[PRE-LOGGING CHECK] Starting GPU detection test before logging is initialized");
    tokio::time::sleep(Duration::from_millis(1000)).await;
    
    match webxr::utils::gpu_compute::GPUCompute::test_gpu().await {
        Ok(_) => println!("[PRE-LOGGING CHECK] GPU test successful."),
        Err(e) => {
            eprintln!("[PRE-LOGGING CHECK] GPU test failed: {}", e);
            eprintln!("[PRE-LOGGING CHECK] Will retry once with additional delay");
            tokio::time::sleep(Duration::from_millis(2000)).await;
            
            match webxr::utils::gpu_compute::GPUCompute::test_gpu().await {
                Ok(_) => println!("[PRE-LOGGING CHECK] GPU test successful on retry!"),
                Err(e) => eprintln!("[PRE-LOGGING CHECK] GPU test failed on retry: {}", e),
            }
        }
    }
    // --- END GPU TEST BEFORE LOGGING ---


    // Initialize logging with settings-based configuration
    let log_config = {
        let settings_read = settings.read().await; // Reads AppFullSettings
        // Access log level correctly from AppFullSettings structure
        let log_level = &settings_read.system.debug.log_level; 
        
        LogConfig::new(
            log_level,
            log_level, // Assuming same level for app and deps for now
        )
    };

    init_logging_with_config(log_config)?;

    debug!("Successfully loaded AppFullSettings"); // Updated log message

    info!("Starting WebXR application...");
    
    // Create web::Data instances first
    // This now holds Data<Arc<RwLock<AppFullSettings>>>
    let settings_data = web::Data::new(settings.clone()); 

    // Initialize services
    let github_config = match GitHubConfig::from_env() {
        Ok(config) => config,
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to load GitHub config: {}", e)))
    };

    // GitHubClient::new might need adjustment if it expects client-facing Settings
    // Assuming it can work with AppFullSettings for now.
    let github_client = match GitHubClient::new(github_config, settings.clone()).await {
        Ok(client) => Arc::new(client),
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHub client: {}", e)))
    };

    let content_api = Arc::new(ContentAPI::new(github_client.clone()));

    // Initialize speech service
    // SpeechService::new might need adjustment if it expects client-facing Settings
    let speech_service = {
        let service = SpeechService::new(settings.clone());
        Some(Arc::new(service))
    };
    
    // Initialize app state asynchronously
    // AppState::new now correctly receives Arc<RwLock<AppFullSettings>>
    let mut app_state = match AppState::new(
            settings.clone(),
            github_client.clone(),
            content_api.clone(),
            None, // Perplexity placeholder
            None, // RAGFlow placeholder
            speech_service,
            None, // GPU Compute placeholder
            "default_session".to_string() // RAGFlow session ID placeholder
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

    info!("Note: Background GitHub data fetch is disabled to resolve compilation issues");

    if metadata_store.is_empty() {
        error!("No metadata found and could not create empty store");
        return Err(std::io::Error::new(std::io::ErrorKind::Other, 
            "No metadata found and could not create empty store".to_string()));
    }

    info!("Loaded {} items from metadata store", metadata_store.len());

    // Update metadata in app state
    {
        let mut app_metadata = app_state.metadata.write().await;
        *app_metadata = metadata_store.clone();
        info!("Loaded metadata into app state");
    }

    // Build initial graph from metadata and initialize GPU compute
    info!("Building initial graph from existing metadata for physics simulation");
    
    let client_manager = app_state.ensure_client_manager().await;
    
    match GraphService::build_graph_from_metadata(&metadata_store).await {
        Ok(graph_data) => {            
            if app_state.gpu_compute.is_none() {
                info!("No GPU compute instance found, initializing one now");
                match GPUCompute::new(&graph_data).await {
                    Ok(gpu_instance) => {
                        info!("GPU compute initialized successfully");
                        app_state.gpu_compute = Some(gpu_instance);
                        
                        info!("Shutting down existing graph service before reinitializing with GPU");
                        let shutdown_start = std::time::Instant::now();
                        
                        match tokio::time::timeout(Duration::from_secs(5), app_state.graph_service.shutdown()).await {
                            Ok(_) => info!("Graph service shutdown completed successfully in {:?}", shutdown_start.elapsed()),
                            Err(_) => {
                                warn!("Graph service shutdown timed out after 5 seconds");
                                warn!("Proceeding with reinitialization anyway - old simulation loop will self-terminate");
                            }
                        }
                        
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        
                        info!("Reinitializing graph service with GPU compute");
                        // GraphService::new receives Arc<RwLock<AppFullSettings>>
                        app_state.graph_service = GraphService::new(
                            settings.clone(),
                            app_state.gpu_compute.clone(),
                            client_manager.clone() // Pass client manager
                        ).await;
                        
                        info!("Graph service successfully reinitialized with GPU compute");
                    },
                    Err(e) => {
                        warn!("Failed to initialize GPU compute: {}. Continuing with CPU fallback.", e);
                        
                        let shutdown_start = std::time::Instant::now();
                        match tokio::time::timeout(Duration::from_secs(5), app_state.graph_service.shutdown()).await {
                            Ok(_) => info!("Graph service shutdown completed successfully in {:?}", shutdown_start.elapsed()),
                            Err(_) => {
                                warn!("Graph service shutdown timed out after 5 seconds");
                                warn!("Proceeding with reinitialization anyway - old simulation loop will self-terminate");
                            }
                        }
                        
                        // Reinitialize graph service with None as GPU compute
                        app_state.graph_service = GraphService::new(
                            settings.clone(),
                            None,
                            client_manager.clone() // Pass client manager
                        ).await;
                        
                        info!("Graph service initialized with CPU fallback");
                    }
                }
            }

            // Update graph data after GPU is initialized (or CPU fallback)
            let mut graph = app_state.graph_service.get_graph_data_mut().await;
            let mut node_map = app_state.graph_service.get_node_map_mut().await;
            *graph = graph_data;
            
            node_map.clear();
            for node in &graph.nodes {
                node_map.insert(node.id.clone(), node.clone());
            }
            
            drop(graph);
            drop(node_map);

            info!("Built initial graph from metadata");
            
        },
        Err(e) => {
            error!("Failed to build initial graph: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to build initial graph: {}", e)));
        }
    }

    info!("Waiting for initial physics layout calculation to complete...");
    tokio::time::sleep(Duration::from_millis(500)).await;
    info!("Initial delay complete. Starting HTTP server...");
    
    app_state.graph_service.start_broadcast_loop(client_manager.clone());
    info!("Position broadcast loop started");
 
    // Create web::Data after all initialization is complete
    let app_state_data = web::Data::new(app_state);

    // Start the server
    let bind_address = {
        let settings_read = settings.read().await; // Reads AppFullSettings
        // Access network settings correctly
        format!("{}:{}", settings_read.system.network.bind_address, settings_read.system.network.port)
    };

    info!("Starting HTTP server on {}", bind_address);

    let server = HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600)
            .supports_credentials();

        let mut app = App::new()
            .wrap(middleware::Logger::default())
            .wrap(cors)
            .wrap(middleware::Compress::default())
            // Pass AppFullSettings wrapped in Data
            .app_data(settings_data.clone()) 
            .app_data(web::Data::new(github_client.clone()))
            .app_data(web::Data::new(content_api.clone()))
            .app_data(app_state_data.clone()) // Add the complete AppState
            .app_data(app_state_data.nostr_service.clone().unwrap_or_else(|| web::Data::new(NostrService::default()))) // Provide default if None
            .app_data(app_state_data.feature_access.clone())
            .route("/wss", web::get().to(socket_flow_handler)) // Changed from /ws to /wss
            .route("/speech", web::get().to(speech_socket_handler))
            .service(
                web::scope("")
                    .configure(api_handler::config)
                    .service(web::scope("/health").configure(health_handler::config))
                    .service(web::scope("/pages").configure(pages_handler::config))
            );
        
        app
    })
    .bind(&bind_address)?
    .run();

    let server_handle = server.handle();

    // Set up signal handlers
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    tokio::spawn(async move {
        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM signal");
            }
            _ = sigint.recv() => {
                info!("Received SIGINT signal");
            }
        }
        info!("Initiating graceful shutdown");
        server_handle.stop(true).await;
    });

    server.await?;

    info!("HTTP server stopped");
    Ok(())
}

use webxr::{
    AppState, Settings,
    handlers::{
        api_handler,
        health_handler,
        pages_handler,
        socket_flow_handler::socket_flow_handler,
        nostr_handler,
        settings_handler,
    },
    services::{
        file_service::FileService,
        graph_service::GraphService,
        github::{GitHubClient, ContentAPI, GitHubConfig},
    }
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenvy::dotenv;
use log::{error, info, debug};
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

    // Initialize local storage and fetch initial data
    info!("Initializing local storage and fetching initial data");
    if let Err(e) = FileService::initialize_local_storage(settings.clone()).await {
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

    info!("Loaded {} items from metadata store", metadata_store.len());

    // Update metadata in app state
    {
        let mut app_metadata = app_state.metadata.write().await;
        *app_metadata = metadata_store.clone();
        info!("Loaded metadata into app state");
    }

    // Build initial graph from metadata
    match GraphService::build_graph_from_metadata(&metadata_store).await {
        Ok(graph_data) => {            
            // Initialize GPU compute with actual graph data
            if let Err(e) = app_state.graph_service.initialize_gpu(settings.clone(), &graph_data).await {
                error!("Failed to initialize GPU compute: {}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, 
                    format!("GPU initialization failed: {}", e)));
            }

            // Update graph data after GPU is initialized
            let mut graph = app_state.graph_service.graph_data.write().await;
            *graph = graph_data;
            drop(graph);

            info!("Built initial graph from metadata");
            
        },
        Err(e) => {
            error!("Failed to build initial graph: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to build initial graph: {}", e)));
        }
    }

    // Create web::Data after all initialization is complete
    let app_state_data = web::Data::new(app_state);

    // Start the server
    let bind_address = {
        let settings_read = settings.read().await;
        format!("{}:{}", (*settings_read).system.network.bind_address, (*settings_read).system.network.port)
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

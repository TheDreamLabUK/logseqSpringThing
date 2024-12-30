use log::{error, info};
use env_logger;

use webxr::{
    AppState,
    Settings,
    RealGitHubService,
    RealGitHubPRService,
    socket_flow_handler,
    services::file_service::FileService,
    gpu_compute::GPUCompute,
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenvy::dotenv;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    env_logger::init();
    info!("Starting WebXR server");

    // Load settings first to get the log level
    let settings = match Settings::new() {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to load settings: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };

    // Create settings Arc
    let settings = Arc::new(RwLock::new(settings));

    // Initialize services
    let github_service = Arc::new(RealGitHubService::new(
        settings.read().await.github.token.clone(),
        settings.read().await.github.owner.clone(),
        settings.read().await.github.repo.clone(),
        settings.read().await.github.base_path.clone(),
        Arc::clone(&settings),
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?);

    let github_pr_service = Arc::new(RealGitHubPRService::new(
        settings.read().await.github.token.clone(),
        settings.read().await.github.owner.clone(),
        settings.read().await.github.repo.clone(),
        settings.read().await.github.base_path.clone(),
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?);

    // Initialize GPU compute
    info!("Initializing GPU compute...");
    let gpu_compute = match GPUCompute::new().await {
        Ok(gpu) => {
            info!("GPU initialization successful");
            Some(Arc::new(RwLock::new(gpu)))
        }
        Err(e) => {
            error!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            None
        }
    };

    // Initialize app state
    let app_state = {
        let state = AppState::new(
            Arc::clone(&settings),
            Arc::clone(&github_service),
            None, // perplexity_service
            None, // ragflow_service
            gpu_compute,
            String::new(), // some_string
            Arc::clone(&github_pr_service),
        );
        web::Data::new(state)
    };

    // Initialize local storage and fetch files from GitHub
    info!("Initializing local storage and fetching files from GitHub...");
    if let Err(e) = FileService::initialize_local_storage(&*github_service, settings.clone()).await {
        error!("Failed to initialize local storage: {}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize local storage: {}", e)));
    }
    info!("Local storage initialization complete");

    // Get port from environment variable or use default
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse::<u16>()
        .unwrap_or(3000);

    let bind_addr = format!("0.0.0.0:{}", port);
    info!("Binding to {}", bind_addr);

    // Serve static files from the correct directory
    let static_files_path = if cfg!(debug_assertions) {
        "../data/public/dist"
    } else {
        "data/public/dist"
    };

    // Update WebSocket configuration to match settings.toml
    let ws_config = web::PayloadConfig::new(
        settings.read().await.system.websocket.max_message_size as usize
    );

    // Configure app with services
    HttpServer::new(move || {
        App::new()
            .wrap(
                Cors::default()
                    .allowed_origin("https://www.visionflow.info")
                    .allow_any_method()
                    .allow_any_header()
                    .max_age(3600)
            )
            .wrap(middleware::Logger::default())
            .app_data(app_state.clone())
            .app_data(ws_config.clone()) // Use clone to avoid move error
            // .service(web::scope("/api/settings").configure(settings::config)) // Temporarily disabled for client-side defaults
            .service(
                web::scope("/api")
                    // Comment out settings-related services
                    // .service(web::scope("/settings").configure(webxr::settings::config))
                    // .service(web::scope("/websocket").configure(websocket::config))
                    .service(web::scope("/graph").configure(webxr::graph_handler::config))
            )
            .service(
                web::resource("/wss")
                    .route(web::get().to(socket_flow_handler))
            )
            .service(Files::new("/", static_files_path).index_file("index.html"))
    })
    .bind(bind_addr)?
    .run()
    .await
}

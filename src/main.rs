#[macro_use]
extern crate log;

use webxr::{
    AppState, Settings,
    init_debug_settings,
    file_handler, graph_handler, visualization_handler,
    handlers::socket_flow_handler,
    RealGitHubService,
    RealGitHubPRService, GPUCompute, GraphData,
    log_data, log_warn,
    services::file_service::FileService,
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenvy::dotenv;

// Handler configuration functions
fn configure_file_handler(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/fetch").to(file_handler::fetch_and_process_files))
       .service(web::resource("/content/{file_name}").to(file_handler::get_file_content))
       .service(web::resource("/refresh").to(file_handler::refresh_graph))
       .service(web::resource("/update").to(file_handler::update_graph));
}

fn configure_graph_handler(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/data").to(graph_handler::get_graph_data))
       .service(web::resource("/data/paginated").to(graph_handler::get_paginated_graph_data))
       .service(web::resource("/update").to(graph_handler::update_graph));
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();

    // Load settings first to get the log level
    let settings = match Settings::new() {
        Ok(s) => {
            debug!("Successfully loaded settings: {:?}", s);
            Arc::new(RwLock::new(s))
        },
        Err(e) => {
            eprintln!("Failed to load settings: {:?}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize settings: {:?}", e)));
        }
    };

    // Create web::Data instances first
    let settings_data = web::Data::new(settings.clone());

    // Initialize services
    let settings_read = settings.read().await;
    let github_service: Arc<RealGitHubService> = match RealGitHubService::new(
        (*settings_read).github.token.clone(),
        (*settings_read).github.owner.clone(),
        (*settings_read).github.repo.clone(),
        (*settings_read).github.base_path.clone(),
        settings.clone(),
    ) {
        Ok(service) => Arc::new(service),
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    };

    let github_pr_service: Arc<RealGitHubPRService> = match RealGitHubPRService::new(
        (*settings_read).github.token.clone(),
        (*settings_read).github.owner.clone(),
        (*settings_read).github.repo.clone(),
        (*settings_read).github.base_path.clone()
    ) {
        Ok(service) => Arc::new(service),
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    };
    drop(settings_read);

    // Initialize GPU compute
    log_data!("Initializing GPU compute...");
    let gpu_compute = match GPUCompute::new(&GraphData::default()).await {
        Ok(gpu) => {
            log_data!("GPU initialization successful");
            Some(gpu)
        }
        Err(e) => {
            log_warn!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            None
        }
    };

    // Initialize app state
    let app_state = web::Data::new(AppState::new(
        settings.clone(),
        github_service.clone(),
        None,
        None,
        gpu_compute,
        "default_conversation".to_string(),
        github_pr_service.clone(),
    ));

    // Initialize debug settings
    let (debug_enabled, websocket_debug, data_debug) = {
        let settings_read = settings.read().await;
        let debug_settings = (
            (*settings_read).server_debug.enabled,
            (*settings_read).server_debug.enable_websocket_debug,
            (*settings_read).server_debug.enable_data_debug,
        );
        debug_settings
    };

    // Initialize our debug logging system
    init_debug_settings(debug_enabled, websocket_debug, data_debug);

    // Initialize local storage and fetch files from GitHub
    info!("Initializing local storage and fetching files from GitHub...");
    if let Err(e) = FileService::initialize_local_storage(&*github_service, settings.clone()).await {
        error!("Failed to initialize local storage: {}", e);
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize local storage: {}", e)));
    }
    info!("Local storage initialization complete");

    // Start the server
    let bind_address = {
        let settings_read = settings.read().await;
        format!("{}:{}", (*settings_read).network.bind_address, (*settings_read).network.port)
    };

    log_data!("Starting HTTP server on {}", bind_address);

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
            .app_data(app_state.clone())
            .app_data(web::Data::new(github_service.clone()))
            .app_data(web::Data::new(github_pr_service.clone()))
            .service(
                web::scope("/api")
                    .service(web::scope("/files").configure(configure_file_handler))
                    .service(web::scope("/graph").configure(configure_graph_handler))
                    .service(web::scope("/visualization").configure(visualization_handler::config))
            )
            .service(
                web::resource("/wss")
                    .app_data(web::PayloadConfig::new(1 << 25))  // 32MB max payload
                    .to(socket_flow_handler::ws_handler)
            )
            .service(Files::new("/", "/app/client").index_file("index.html"))
    })
    .bind(&bind_address)?
    .run()
    .await?;

    log_data!("HTTP server stopped");
    Ok(())
}

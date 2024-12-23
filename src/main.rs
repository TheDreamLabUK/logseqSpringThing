use log::{debug, info};
use env_logger;
use webxr::{
    Settings,
    handlers::{
        file_handler,
        graph_handler,
        settings,
        socket_flow_handler::socket_flow_handler,
    },
    AppState,
    services::{
        file_service::RealGitHubService,
        github_service::RealGitHubPRService,
    },
};

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use actix_files::Files;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenvy::dotenv;
use std::env;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    env_logger::init();
    info!("Starting LogseqXR server");

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

    // Load environment variables
    let github_token = env::var("GITHUB_TOKEN")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("GITHUB_TOKEN not set: {}", e)))?;
    let github_owner = env::var("GITHUB_OWNER")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("GITHUB_OWNER not set: {}", e)))?;
    let github_repo = env::var("GITHUB_REPO")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("GITHUB_REPO not set: {}", e)))?;
    let github_base_path = env::var("GITHUB_BASE_PATH").unwrap_or_else(|_| String::from(""));

    // Initialize GitHub services
    let github_service = Arc::new(RealGitHubService::new(
        github_token.clone(),
        github_owner.clone(),
        github_repo.clone(),
        github_base_path.clone(),
        settings.clone(),
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHub service: {}", e)))?);

    let github_pr_service = Arc::new(RealGitHubPRService::new(
        github_token,
        github_owner,
        github_repo,
        github_base_path,
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to initialize GitHub PR service: {}", e)))?);
    
    // Optional services
    let perplexity_service = None; // Initialize if needed
    let ragflow_service = None; // Initialize if needed
    let gpu_compute = None; // Initialize if needed
    
    // Create AppState with initialized services
    let app_state = web::Data::new(AppState::new(
        settings.clone(),
        github_service,
        perplexity_service,
        ragflow_service,
        gpu_compute,
        String::from("default"), // ragflow_conversation_id
        github_pr_service,
    ));

    // Use PORT env var with fallback to 3001 as specified in docs
    let port = env::var("PORT").unwrap_or_else(|_| "3001".to_string());
    let bind_addr = format!("0.0.0.0:{}", port);
    info!("Binding to {}", bind_addr);

    // Configure app with services
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::default()
                .allow_any_origin()
                .allow_any_method()
                .allow_any_header()
                .max_age(3600))
            .wrap(middleware::Logger::default())
            .app_data(app_state.clone())
            .service(
                web::scope("/api")
                    .service(web::scope("/files").configure(file_handler::config))
                    .service(web::scope("/graph").configure(graph_handler::config))
                    .service(web::scope("/visualization").configure(settings::config))
            )
            .service(
                web::resource("/wss")
                    .app_data(web::PayloadConfig::new(1 << 25))  // 32MB max payload
                    .route(web::get().to(socket_flow_handler))
            )
            .service(Files::new("/", "static").index_file("index.html"))
    })
    .bind(bind_addr)?
    .run()
    .await
}

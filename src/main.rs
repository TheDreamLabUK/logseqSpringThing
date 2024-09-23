// src/main.rs

use actix::prelude::*;
use actix_files as fs;
use actix_web::{web, App, HttpServer, HttpRequest, HttpResponse, Error};
use actix_web::middleware::Logger;
use actix_web_actors::ws;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use env_logger::Env;

mod handlers;
mod app_state;
mod models;
mod config;
mod services;
mod utils; // Added utils module

// Import necessary components from modules
use crate::handlers::file_handler::fetch_and_process_files;
use crate::handlers::graph_handler::{get_graph_data, refresh_graph};
use crate::app_state::AppState;
use crate::models::graph::GraphData;
use crate::config::Settings;
use crate::services::perplexity_service::RealApiClient;
use crate::utils::websocket_manager::handle_websocket; // Import the websocket handler

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger with environment variables and default log level
    env_logger::init_from_env(Env::default().default_filter_or("debug"));

    // Initialize shared application state
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let settings = Settings::new().expect("Failed to load settings");
    let api_client = Arc::new(RealApiClient::new());

    let app_state = web::Data::new(AppState {
        graph_data,
        file_cache,
        settings,
        api_client,
    });

    let bind_address = "0.0.0.0";
    let port = 8080;

    println!("Starting server at http://{}:{}", bind_address, port);

    // Start HTTP server with configured routes and middleware
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(Logger::default())
            .route("/ws/", web::get().to(handle_websocket)) // Updated WebSocket route
            .service(fs::Files::new("/", "./data/public").index_file("index.html")) // Serve static files
            .route("/fetch-and-process", web::post().to(fetch_and_process_files)) // Additional routes
            .route("/graph-data", web::get().to(get_graph_data))
            .route("/refresh-graph", web::post().to(refresh_graph))
    })
    .bind(format!("{}:{}", bind_address, port))?
    .run()
    .await
}

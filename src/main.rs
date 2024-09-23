use actix_web::{web, App, HttpServer, HttpRequest, HttpResponse, Error};
use actix_web_actors::ws;
use actix_files as fs;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use env_logger::Env;
use std::env;
use rustls::ServerConfig;
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::fs::File;
use std::io::BufReader;

mod websocket;
mod handlers;
mod app_state;
mod models;
mod config;
mod services;

use crate::handlers::file_handler::fetch_and_process_files;
use crate::handlers::graph_handler::{get_graph_data, refresh_graph};
use crate::app_state::AppState;
use crate::models::graph::GraphData;
use crate::websocket::WebSocketSession;
use crate::config::Settings;
use crate::services::perplexity_service::RealApiClient;

async fn websocket_route(req: HttpRequest, stream: web::Payload, app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    println!("WebSocket route hit");
    println!("Request headers: {:?}", req.headers());
    let app_state_arc = app_state.into_inner();
    ws::start(WebSocketSession::new(app_state_arc), &req, stream)
}

fn load_rustls_config() -> Option<ServerConfig> {
    let cert_file = File::open("/etc/nginx/ssl/selfsigned.crt");
    let key_file = File::open("/etc/nginx/ssl/selfsigned.key");

    match (cert_file, key_file) {
        (Ok(cert_file), Ok(key_file)) => {
            let mut cert_reader = BufReader::new(cert_file);
            let mut key_reader = BufReader::new(key_file);

            // Parse and load the certificates and private key
            let cert_chain = certs(&mut cert_reader).unwrap().into_iter().map(rustls::Certificate).collect();
            let mut keys: Vec<rustls::PrivateKey> = pkcs8_private_keys(&mut key_reader)
                .unwrap()
                .into_iter()
                .map(rustls::PrivateKey)
                .collect();

            // Create ServerConfig
            let config = ServerConfig::builder()
                .with_safe_defaults()
                .with_no_client_auth()
                .with_single_cert(cert_chain, keys.remove(0))
                .unwrap();

            Some(config)
        },
        _ => {
            println!("SSL certificate files not found. Running in HTTP mode.");
            None
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("debug"));

    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let settings = Settings::new().unwrap();
    let api_client = Arc::new(RealApiClient::new());

    let app_state = web::Data::new(AppState {
        graph_data,
        file_cache,
        settings,
        api_client,
    });

    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let bind_address = env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".to_string());

    let rustls_config = load_rustls_config();

    let server = HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(fs::Files::new("/", "./data/public").index_file("index.html"))
            .route("/ws", web::get().to(websocket_route))
            .route("/fetch-and-process", web::post().to(fetch_and_process_files))
            .route("/graph-data", web::get().to(get_graph_data))
            .route("/refresh-graph", web::post().to(refresh_graph))
    });

    let server = if let Some(config) = rustls_config {
        println!("Starting server with HTTPS at https://{}:{}", bind_address, port);
        server.bind_rustls(format!("{}:{}", bind_address, port), config)?
    } else {
        println!("Starting server with HTTP at http://{}:{}", bind_address, port);
        server.bind(format!("{}:{}", bind_address, port))?
    };

    server.run().await
}

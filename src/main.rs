use actix_web::{web, App, HttpServer, HttpRequest, HttpResponse, Error};
use actix_web_actors::ws;
use actix_files as fs;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use env_logger::Env;
use rustls::{ServerConfig, Certificate, PrivateKey};
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

fn load_ssl_config() -> ServerConfig {
    let cert_file = &mut BufReader::new(File::open("cert.pem").unwrap());
    let key_file = &mut BufReader::new(File::open("key.pem").unwrap());
    
    let cert_chain = certs(cert_file)
        .unwrap()
        .into_iter()
        .map(Certificate)
        .collect();
    let mut keys: Vec<PrivateKey> = pkcs8_private_keys(key_file)
        .unwrap()
        .into_iter()
        .map(PrivateKey)
        .collect();

    ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(cert_chain, keys.remove(0))
        .unwrap()
}

async fn websocket_route(req: HttpRequest, stream: web::Payload, app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    ws::start(WebSocketSession::new(app_state.get_ref().clone()), &req, stream)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

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

    let ssl_config = load_ssl_config();

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(fs::Files::new("/", "./public").index_file("index.html"))
            .service(fs::Files::new("/js", "./public/js").show_files_listing())
            .route("/fetch-and-process", web::post().to(fetch_and_process_files))
            .route("/graph-data", web::get().to(get_graph_data))
            .route("/refresh-graph", web::post().to(refresh_graph))
            .route("/ws", web::get().to(websocket_route))
    })
    .bind_rustls("0.0.0.0:8443", ssl_config)?
    .run()
    .await
}

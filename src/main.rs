use crate::{
    config::Settings,
    handlers::{
        file_handler,
        graph_handler,
        settings,
        socket_flow_handler::{self, socket_flow_handler},
    },
    services::file_service::FileService,
    app_state::AppState,
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
       .service(
           web::resource("/update")
               .route(web::post().to(graph_handler::update_graph))
       );
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
    let app_state = web::Data::new(AppState::new());

    // Configure app with services
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::default()
                .allow_any_origin()
                .allow_any_method()
                .allow_any_header()
                .max_age(3600))
            .wrap(middleware::Logger::default())
            .app_data(settings_data.clone())
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
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

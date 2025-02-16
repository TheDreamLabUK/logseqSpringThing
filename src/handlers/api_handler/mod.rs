pub mod files;
pub mod graph;
pub mod visualization;

// Re-export specific types and functions
// Re-export specific types and functions
pub use files::{
    fetch_and_process_files,
    get_file_content,
};

pub use graph::{
    get_graph_data,
    get_paginated_graph_data,
    refresh_graph,
    update_graph,
};

pub use visualization::get_visualization_settings;

use actix_web::web;

// Configure all API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .configure(files::config)
            .configure(graph::config)
            .configure(visualization::config)
            .configure(crate::handlers::nostr_handler::config)
            .configure(crate::handlers::settings_handler::config)
    );
}

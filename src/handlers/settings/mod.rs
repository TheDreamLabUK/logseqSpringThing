pub mod websocket;
pub mod visualization;
pub mod common;

pub use websocket::*;
pub use visualization::*;
pub use common::*;

use actix_web::web;

// Register all settings handlers
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(visualization::config)
       .service(websocket::config);
}

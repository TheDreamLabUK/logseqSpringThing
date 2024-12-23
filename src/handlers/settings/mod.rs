mod websocket;
mod visualization;
mod common;

pub use websocket::*;
pub use visualization::*;
pub use common::*;

use actix_web::{web, HttpResponse};
use crate::config::Settings;
use std::sync::Arc;
use tokio::sync::RwLock;

// Register all settings handlers
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/settings")
            .service(
                web::scope("/visualization")
                    .configure(visualization::config)
            )
            .service(
                web::scope("/websocket")
                    .configure(websocket::config)
            )
    );
}

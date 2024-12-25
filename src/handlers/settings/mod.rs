pub mod websocket;
pub mod visualization;
pub mod common;

pub use websocket::*;
pub use visualization::*;
pub use common::*;

use actix_web::{web, get, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use log::debug;
use crate::config::Settings;

#[get("")]
async fn get_all_settings(settings: web::Data<Arc<RwLock<Settings>>>) -> HttpResponse {
    debug!("Getting all settings");
    let settings_guard = settings.read().await;
    HttpResponse::Ok().json(&*settings_guard)
}

// Register all settings handlers
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(get_all_settings);
    visualization::config(cfg);
    websocket::config(cfg);
}

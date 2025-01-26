use std::sync::Arc;
use tokio::sync::RwLock;
use crate::config::Settings;

#[derive(Debug)]  // Only Debug derive, remove Clone
pub struct AppState {
    pub settings: Arc<RwLock<Settings>>,
}

impl AppState {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings: Arc::new(RwLock::new(settings)),
        }
    }

    pub fn clone_settings(&self) -> Arc<RwLock<Settings>> {
        self.settings.clone()
    }
} 
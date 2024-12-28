use tokio::time::{interval, Duration};
use std::sync::Arc;
use crate::state::AppState;

pub struct BackupService {
    state: Arc<AppState>,
    interval: Duration,
}

impl BackupService {
    pub fn new(state: Arc<AppState>, interval_hours: u64) -> Self {
        Self {
            state,
            interval: Duration::from_secs(interval_hours * 3600),
        }
    }

    pub async fn run(&self) {
        let mut interval = interval(self.interval);

        loop {
            interval.tick().await;
            
            match self.state.create_backup().await {
                Ok(path) => log::info!("Created automatic backup at {:?}", path),
                Err(e) => log::error!("Failed to create automatic backup: {}", e),
            }
        }
    }
} 